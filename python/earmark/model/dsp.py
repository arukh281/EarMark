"""WOLA analysis/synthesis, the ERB filterbank and causal normalisation.

Every signal-level number comes from :mod:`earmark.constants` (the M0 contract):
a 320-sample periodic sqrt-Hann window, a 160-sample hop, an unscaled forward DFT and
a ``1 / N_FFT`` inverse, uncentred framing, 32 contiguous rectangular ERB bands and a
causal exponential mean with ``NORM_ALPHA`` per hop.

Conventions that are *not* in the contract but that the engine must copy exactly are
defined here as module constants (the export manifest records them):

* ``ERB_NORM_INIT_DB`` / ``UNIT_NORM_INIT``: initial running means. They are the
  DeepFilterNet initial values shifted to this contract's unscaled forward DFT
  (``SPECTRUM_SCALE_DB = 20 log10(N_FFT)``), spread linearly from the lowest to the
  highest band or bin.
* ERB features: ``db = 10 log10(mean band power + POWER_EPS)``, the running mean ``m``
  is updated first, then ``feature = (db - m) / ERB_FEATURE_SCALE_DB``.
* Unit-normalised low band: the running mean ``s`` of ``|X|`` is updated first, then
  ``feature = X / sqrt(s)``.

All functions here are pure: they never mutate their inputs, and every recurrent
quantity is passed in and returned explicitly so the offline (sequence) and streaming
(one hop) paths stay bit-for-bit comparable.
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from typing import Final

import torch
from torch import Tensor

from earmark import constants as C

#: Offset in dB between DeepFilterNet's 1/N-normalised STFT and this contract's unscaled DFT.
SPECTRUM_SCALE_DB: Final[float] = 20.0 * math.log10(C.N_FFT)

#: Initial ERB running means in dB for (lowest band, highest band); linear in between.
ERB_NORM_INIT_DB: Final[tuple[float, float]] = (-60.0 + SPECTRUM_SCALE_DB, -90.0 + SPECTRUM_SCALE_DB)

#: Initial running means of |X| for (bin 0, bin DF_BINS - 1); linear in between.
UNIT_NORM_INIT: Final[tuple[float, float]] = (1e-3 * C.N_FFT, 1e-4 * C.N_FFT)

#: Divisor applied to mean-removed ERB log-power features.
ERB_FEATURE_SCALE_DB: Final[float] = 40.0

#: Added to band power before the logarithm.
POWER_EPS: Final[float] = 1e-10

#: Frames per chunk in :func:`exp_mean_scan` (keeps alpha ** chunk well conditioned).
SCAN_CHUNK: Final[int] = 64


# ----------------------------------------------------------------------------- dtypes


def fp32_region(device: torch.device) -> AbstractContextManager[object]:
    """Context that disables autocast so DSP runs at full precision.

    n_fft = 320 is not a power of two, which half-precision cuFFT rejects; the spectral
    front and back ends of the model therefore always run in fp32 (or fp64).
    """
    try:
        return torch.autocast(device_type=device.type, enabled=False)
    except RuntimeError:  # pragma: no cover - device without autocast support
        return nullcontext()


def dsp_dtype(dtype: torch.dtype) -> torch.dtype:
    """Real dtype used for DSP: float64 stays float64, everything else becomes float32."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def complex_dtype(real: torch.dtype) -> torch.dtype:
    """Complex dtype matching a real DSP dtype."""
    return torch.complex128 if real == torch.float64 else torch.complex64


# ------------------------------------------------------------------------------- WOLA


def sqrt_hann_window(
    device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Periodic sqrt-Hann window ``w[n] = sin(pi n / WINDOW_LENGTH)``, computed in float64."""
    n = torch.arange(C.WINDOW_LENGTH, dtype=torch.float64)
    return torch.sin(math.pi * n / C.WINDOW_LENGTH).to(device=device, dtype=dtype)


def num_frames(num_samples: int) -> int:
    """Contract frames in a signal: ``(N - WINDOW_LENGTH) // HOP_LENGTH + 1`` (0 if too short)."""
    if num_samples < C.WINDOW_LENGTH:
        return 0
    return (num_samples - C.WINDOW_LENGTH) // C.HOP_LENGTH + 1


def frame_signal(x: Tensor) -> Tensor:
    """Split ``x`` of shape ``[B, N]`` into contract frames ``[B, T, WINDOW_LENGTH]``.

    Frame ``t`` covers samples ``[t * HOP_LENGTH, t * HOP_LENGTH + WINDOW_LENGTH)``;
    trailing samples that do not fill a whole frame are ignored.
    """
    if x.dim() != 2:
        raise ValueError(f"expected [batch, samples], got shape {tuple(x.shape)}")
    if x.shape[-1] < C.WINDOW_LENGTH:
        raise ValueError(f"need at least {C.WINDOW_LENGTH} samples, got {x.shape[-1]}")
    return x.unfold(-1, C.WINDOW_LENGTH, C.HOP_LENGTH)


def analysis_frame(samples: Tensor, window: Tensor | None = None) -> Tensor:
    """Windowed forward DFT of one or more frames ``[..., WINDOW_LENGTH]`` -> ``[..., N_BINS]``."""
    if window is None:
        window = sqrt_hann_window(samples.device, samples.dtype)
    return torch.fft.rfft(samples * window, n=C.N_FFT, dim=-1, norm="backward")


def synthesis_frame(spec: Tensor, window: Tensor | None = None) -> Tensor:
    """Inverse DFT and synthesis window: ``[..., N_BINS]`` -> ``[..., WINDOW_LENGTH]``."""
    frame = torch.fft.irfft(spec, n=C.N_FFT, dim=-1, norm="backward")[..., : C.WINDOW_LENGTH]
    if window is None:
        window = sqrt_hann_window(frame.device, frame.dtype)
    return frame * window


def stft(x: Tensor) -> Tensor:
    """Contract STFT of ``x`` ``[B, N]`` -> complex ``[B, T, N_BINS]`` (no centring, no scaling)."""
    real = dsp_dtype(x.dtype)
    with fp32_region(x.device):
        frames = frame_signal(x.to(real))
        return analysis_frame(frames, sqrt_hann_window(x.device, real))


def overlap_add(frames: Tensor, tail: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Overlap-add windowed frames ``[B, T, WINDOW_LENGTH]`` at 50 % overlap.

    Returns ``(wav, new_tail)``. ``wav`` has ``T * HOP_LENGTH`` samples: block ``t`` is the
    first half of frame ``t`` plus the second half of frame ``t - 1`` (``tail`` for
    ``t = 0``, zeros if ``None``). ``new_tail`` is the second half of the last frame,
    which becomes final only once the next frame arrives.
    """
    batch, n_frames, _ = frames.shape
    first = frames[..., : C.HOP_LENGTH]
    second = frames[..., C.HOP_LENGTH :]
    if tail is None:
        tail = frames.new_zeros(batch, C.HOP_LENGTH)
    previous = torch.cat([tail.unsqueeze(1), second[:, :-1]], dim=1)
    wav = (first + previous).reshape(batch, n_frames * C.HOP_LENGTH)
    return wav, second[:, -1]


def istft(spec: Tensor) -> Tensor:
    """WOLA synthesis of ``[B, T, N_BINS]`` -> ``[B, (T + 1) * HOP_LENGTH]``.

    Inverts :func:`stft` exactly on samples ``[HOP_LENGTH, T * HOP_LENGTH)``, where two
    frames overlap; the first and last hop are covered by one frame only.
    """
    real = torch.float64 if spec.dtype == torch.complex128 else torch.float32
    with fp32_region(spec.device):
        frames = synthesis_frame(spec, sqrt_hann_window(spec.device, real))
        wav, tail = overlap_add(frames)
        return torch.cat([wav, tail], dim=-1)


def pad_to_hop(x: Tensor) -> tuple[Tensor, int]:
    """Right-pad ``[B, N]`` with zeros to a whole number of hops; returns ``(padded, N)``."""
    n = x.shape[-1]
    extra = (-n) % C.HOP_LENGTH
    if extra:
        x = torch.nn.functional.pad(x, (0, extra))
    return x, n


# -------------------------------------------------------------------------------- ERB


def erb_band_edges() -> list[int]:
    """Band edges in bins: band ``b`` covers ``[edges[b], edges[b + 1])``."""
    edges = [0]
    for width in C.ERB_WIDTHS:
        edges.append(edges[-1] + width)
    return edges


def erb_band_index(device: torch.device | None = None) -> Tensor:
    """Band index of each STFT bin, ``[N_BINS]`` int64."""
    widths = torch.tensor(C.ERB_WIDTHS, dtype=torch.long)
    return torch.repeat_interleave(torch.arange(C.ERB_BANDS), widths).to(device)


def erb_matrix(device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Band-averaging matrix ``[N_BINS, ERB_BANDS]`` with ``1 / width`` inside each band."""
    matrix = torch.zeros(C.N_BINS, C.ERB_BANDS, dtype=torch.float64)
    edges = erb_band_edges()
    for band in range(C.ERB_BANDS):
        matrix[edges[band] : edges[band + 1], band] = 1.0 / C.ERB_WIDTHS[band]
    return matrix.to(device=device, dtype=dtype)


def erb_power(spec: Tensor, matrix: Tensor | None = None) -> Tensor:
    """Mean power per ERB band: complex ``[..., N_BINS]`` -> real ``[..., ERB_BANDS]``."""
    power = spec.real.square() + spec.imag.square()
    if matrix is None:
        matrix = erb_matrix(spec.device, power.dtype)
    return power @ matrix


def erb_expand(gains: Tensor, index: Tensor | None = None) -> Tensor:
    """Expand per-band values ``[..., ERB_BANDS]`` to bins ``[..., N_BINS]`` (rectangular)."""
    if index is None:
        index = erb_band_index(gains.device)
    return gains.index_select(-1, index)


def erb_log_power(spec: Tensor, matrix: Tensor | None = None) -> Tensor:
    """ERB log-power in dB, ``10 log10(mean band power + POWER_EPS)``."""
    return 10.0 * torch.log10(erb_power(spec, matrix) + POWER_EPS)


# ---------------------------------------------------------------------- normalisation


def erb_norm_init(device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Initial ERB running means ``[ERB_BANDS]`` in dB."""
    lo, hi = ERB_NORM_INIT_DB
    return torch.linspace(lo, hi, C.ERB_BANDS, dtype=torch.float64).to(device=device, dtype=dtype)


def unit_norm_init(device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Initial running means of ``|X|`` for the low band, ``[DF_BINS]``."""
    lo, hi = UNIT_NORM_INIT
    return torch.linspace(lo, hi, C.DF_BINS, dtype=torch.float64).to(device=device, dtype=dtype)


def exp_mean_step(x: Tensor, mean: Tensor, alpha: float = C.NORM_ALPHA) -> Tensor:
    """One causal exponential-mean update: ``alpha * mean + (1 - alpha) * x``."""
    return alpha * mean + (1.0 - alpha) * x


def exp_mean_scan(
    x: Tensor, init: Tensor, alpha: float = C.NORM_ALPHA, chunk: int = SCAN_CHUNK
) -> tuple[Tensor, Tensor]:
    """Running means of ``x`` ``[B, T, F]`` from ``init`` ``[B, F]`` (or ``[F]``).

    Equivalent to applying :func:`exp_mean_step` frame by frame (``m_t`` includes
    ``x_t``) but vectorised: each chunk of ``chunk`` frames is one small lower-triangular
    matmul plus the decayed carry-in. Returns ``(means [B, T, F], last [B, F])``.
    """
    batch, n_frames, features = x.shape
    mean = init.expand(batch, features)
    if n_frames == 0:
        return x, mean
    length = min(chunk, n_frames)
    idx = torch.arange(length, dtype=torch.float64)
    lag = idx[:, None] - idx[None, :]
    toeplitz = torch.where(lag >= 0, (1.0 - alpha) * alpha ** lag.clamp(min=0), 0.0)
    toeplitz = toeplitz.to(device=x.device, dtype=x.dtype)
    decay = (alpha ** (idx + 1.0)).to(device=x.device, dtype=x.dtype)
    outputs: list[Tensor] = []
    for start in range(0, n_frames, length):
        block = x[:, start : start + length]
        size = block.shape[1]
        means = torch.einsum("ji,bif->bjf", toeplitz[:size, :size], block)
        means = means + decay[:size, None] * mean[:, None, :]
        outputs.append(means)
        mean = means[:, -1]
    return torch.cat(outputs, dim=1), mean


def erb_features(
    spec: Tensor, state: Tensor, matrix: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Causally normalised ERB log-power for a sequence.

    ``spec`` is complex ``[B, T, N_BINS]``, ``state`` the running means ``[B, ERB_BANDS]``.
    Returns ``(features [B, T, ERB_BANDS], new_state)``.
    """
    db = erb_log_power(spec, matrix)
    means, last = exp_mean_scan(db, state)
    return (db - means) / ERB_FEATURE_SCALE_DB, last


def erb_features_step(
    spec: Tensor, state: Tensor, matrix: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """One-hop version of :func:`erb_features`: ``spec`` ``[B, N_BINS]`` -> ``[B, ERB_BANDS]``."""
    db = erb_log_power(spec, matrix)
    mean = exp_mean_step(db, state)
    return (db - mean) / ERB_FEATURE_SCALE_DB, mean


def unit_norm_features(spec_low: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Unit-normalised complex low band for a sequence.

    ``spec_low`` is complex ``[B, T, DF_BINS]``, ``state`` the running means of ``|X|``
    ``[B, DF_BINS]``. Returns ``(features complex [B, T, DF_BINS], new_state)``.
    """
    means, last = exp_mean_scan(spec_low.abs(), state)
    return spec_low / means.sqrt(), last


def unit_norm_features_step(spec_low: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """One-hop version of :func:`unit_norm_features` (``[B, DF_BINS]`` complex)."""
    mean = exp_mean_step(spec_low.abs(), state)
    return spec_low / mean.sqrt(), mean


# ------------------------------------------------------------------------ deep filter


def deep_filter(
    spec_low: Tensor, coefs: Tensor, history: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Order-``DF_ORDER`` complex deep filter with zero lookahead.

    ``Y[t, k] = sum_i coefs[t, i, k] * X[t - i, k]`` for ``i < DF_ORDER``.

    Args:
        spec_low: complex ``[B, T, K]`` input spectrum (the ERB-gained low band).
        coefs: complex ``[B, T, DF_ORDER, K]``; tap ``i`` multiplies frame ``t - i``.
        history: complex ``[B, DF_ORDER - 1, K]``, the previous frames oldest first
            (``t - 2``, ``t - 1``); zeros if ``None``.

    Returns:
        ``(filtered [B, T, K], new_history [B, DF_ORDER - 1, K])``.
    """
    batch, n_frames, bins = spec_low.shape
    order = coefs.shape[2]
    if history is None:
        history = spec_low.new_zeros(batch, order - 1, bins)
    padded = torch.cat([history, spec_low], dim=1)
    out = coefs[:, :, 0] * spec_low
    for tap in range(1, order):
        start = order - 1 - tap
        out = out + coefs[:, :, tap] * padded[:, start : start + n_frames]
    return out, padded[:, n_frames:]


__all__ = [
    "ERB_FEATURE_SCALE_DB",
    "ERB_NORM_INIT_DB",
    "POWER_EPS",
    "SCAN_CHUNK",
    "SPECTRUM_SCALE_DB",
    "UNIT_NORM_INIT",
    "analysis_frame",
    "complex_dtype",
    "deep_filter",
    "dsp_dtype",
    "erb_band_edges",
    "erb_band_index",
    "erb_expand",
    "erb_features",
    "erb_features_step",
    "erb_log_power",
    "erb_matrix",
    "erb_norm_init",
    "erb_power",
    "exp_mean_scan",
    "exp_mean_step",
    "fp32_region",
    "frame_signal",
    "istft",
    "num_frames",
    "overlap_add",
    "pad_to_hop",
    "sqrt_hann_window",
    "stft",
    "synthesis_frame",
    "unit_norm_features",
    "unit_norm_features_step",
    "unit_norm_init",
]
