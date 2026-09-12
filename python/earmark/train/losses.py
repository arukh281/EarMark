"""Training losses for Earmark (M3 in the plan).

The total loss is a weighted sum of per-example terms, averaged over the batch:

* ``mr_stft`` - multi-resolution STFT loss (spectral convergence plus log-magnitude L1,
  Yamamoto et al. 2020) over three window lengths. Only on *reference-active* examples,
  where the training reference holds speech; spectral convergence is undefined otherwise.
* ``spectral`` - compressed complex spectral loss with exponent ``c = 0.3``: a magnitude
  term on ``|X|^c`` plus a complex term on ``|X|^c e^{j angle X}`` (DeepFilterNet style).
  All examples; with a silent reference it simply pulls the output towards silence.
* ``asym`` - the VoiceFilter-Lite asymmetric loss (Wang et al. 2020). The compressed
  magnitude error ``x = |S|^c - |Y|^c`` is multiplied by ``alpha = 10`` where ``x > 0``,
  that is where the output is quieter than the reference (over-suppressing the target).
  The formula is VoiceFilter-Lite's exactly: ``mean(g(x)^2)`` with ``g(x) = alpha x`` for
  ``x > 0`` and ``x`` otherwise. All examples.
* ``sisdr`` - ``SISDR_MAX_DB - SI-SDR`` in dB with a soft cap (thresholded SI-SDR), so it
  is non-negative and 0 for a perfect estimate. Reference-active examples only.
* ``absent`` - log-energy penalty on reference-silent examples (target absent; with the
  NULL embedding, no speech at all): output energy relative to the mixture in dB, above a
  floor, so it is 0 once the output is ``-absent_floor_db`` dB below the input.
* ``vad`` - BCE-with-logits of the personal-VAD head against the mixer's frame labels,
  weight ``lambda = 0.1``. All examples (with the NULL embedding the labels cover all speech).

Precision: every term runs in fp32 (fp64 inputs stay fp64) with autocast disabled.
Half-precision cuFFT rejects ``n_fft = 320``, and ``|X|^0.3`` needs fp32 range near zero.

Alignment: the network's output lags its input by one hop (``OUTPUT_DELAY_SAMPLES``), and
model frame ``t`` analyses contract frame ``t - 1`` (``MODEL_FRAME_OFFSET``). The loss
compares ``out.wav[:, 160:]`` with ``batch["target"][:, :-160]`` and pairs
``out.vad_logit[:, 1:]`` with ``batch["vad"]``, as the mixer documents.

Batch keys used: ``mixture``, ``target`` and ``vad`` (see :mod:`earmark.data.mixer`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, NamedTuple, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from earmark.model.dsp import dsp_dtype, fp32_region
from earmark.model.earmark_net import MODEL_FRAME_OFFSET, OUTPUT_DELAY_SAMPLES

#: One STFT resolution: ``(n_fft, hop, window_length)``.
Resolution = tuple[int, int, int]

#: Loss terms in reporting order.
TERMS: Final[tuple[str, ...]] = ("mr_stft", "spectral", "asym", "sisdr", "absent", "vad")

#: Added to bin powers before a square root or a fractional power (keeps gradients finite).
POWER_EPS: Final[float] = 1e-12

#: Added to signal energies in energy ratios.
ENERGY_EPS: Final[float] = 1e-8

#: Added to energies inside SI-SDR; small enough that the soft cap, not the epsilon,
#: decides the value for any audible reference.
SISDR_EPS: Final[float] = 1e-12


@dataclass(frozen=True)
class LossConfig:
    """Weights and settings of every loss term (``c``, ``alpha`` and ``lambda`` are the M3 plan's).

    The weights were set on synthetic mixer batches (harmonic "talkers" in white and
    brown noise, 3 x 32 examples) so that no term drowns the others; revisit them on
    the mini-full run with real speech. Moving
    from the unprocessed mixture to one with the residual 10 dB lower lowers the
    weighted MR-STFT, spectral and asymmetric terms by about 0.35 each and the SI-SDR
    term by about 0.24. A target over-suppressed by 6 dB costs about 2.5 times one left
    6 dB too loud; with equal weights MR-STFT alone would dominate and rank them the
    other way round. The VAD BCE starts at about ``0.1 * 0.69``.
    """

    mr_stft_weight: float = 0.25
    mr_stft_resolutions: tuple[Resolution, ...] = ((256, 64, 256), (512, 128, 512), (1024, 256, 1024))
    spectral_weight: float = 4.0
    #: STFT of the compressed spectral and asymmetric terms: the model's 20 ms window, 4x overlap.
    spectral_resolution: Resolution = (320, 80, 320)
    compression: float = 0.3
    #: Weight of the complex part relative to the magnitude part of the spectral term.
    complex_factor: float = 1.0
    asym_weight: float = 8.0
    asym_alpha: float = 10.0
    sisdr_weight: float = 0.03
    #: Soft cap of the thresholded SI-SDR; the ``sisdr`` term is ``sisdr_max_db - SI-SDR``.
    sisdr_max_db: float = 50.0
    absent_weight: float = 0.03
    #: The ``absent`` term is 0 when the output is this far (dB) below the mixture.
    absent_floor_db: float = -60.0
    vad_weight: float = 0.1
    #: A reference whose mean power is below this (dBFS) counts as silent.
    active_threshold_db: float = -80.0
    #: Floor added to bin powers inside the log-magnitude term (about -90 dB).
    log_floor: float = 1e-9

    def __post_init__(self) -> None:
        for name in (
            "mr_stft_weight", "spectral_weight", "complex_factor", "asym_weight", "sisdr_weight",
            "absent_weight", "vad_weight",
        ):  # fmt: skip
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 < self.compression <= 1.0:
            raise ValueError("compression must be in (0, 1]")
        if self.asym_alpha < 1.0:
            raise ValueError("asym_alpha must be at least 1 (1 makes the term symmetric)")
        if self.sisdr_max_db <= 0 or self.absent_floor_db >= 0 or self.log_floor <= 0:
            raise ValueError("sisdr_max_db must be positive, absent_floor_db negative, log_floor positive")
        if not self.mr_stft_resolutions:
            raise ValueError("mr_stft_resolutions must not be empty")
        for res in (*self.mr_stft_resolutions, self.spectral_resolution):
            n_fft, hop, win = res
            if not (0 < hop <= win <= n_fft):
                raise ValueError(f"resolution {res} must satisfy 0 < hop <= window <= n_fft")

    def weights(self) -> dict[str, float]:
        """Weight of each term, keyed like :data:`TERMS`."""
        return {
            "mr_stft": self.mr_stft_weight,
            "spectral": self.spectral_weight,
            "asym": self.asym_weight,
            "sisdr": self.sisdr_weight,
            "absent": self.absent_weight,
            "vad": self.vad_weight,
        }


class LossOutput(NamedTuple):
    """Result of :class:`EarmarkLoss`."""

    total: Tensor  #: scalar, differentiable, fp32
    terms: dict[str, Tensor]  #: unweighted batch mean of each term (detached scalars)
    stats: dict[str, Tensor]  #: detached diagnostics: SI-SDR(i) on active and energy on silent examples


class _HasWavAndVad(Protocol):
    @property
    def wav(self) -> Tensor: ...

    @property
    def vad_logit(self) -> Tensor: ...


# ------------------------------------------------------------------------------ helpers


def align(wav: Tensor, reference: Tensor, delay: int = OUTPUT_DELAY_SAMPLES) -> tuple[Tensor, Tensor]:
    """Remove the model's output delay: ``(wav[..., delay:], reference[..., :-delay])``."""
    if wav.shape != reference.shape:
        raise ValueError(f"output {tuple(wav.shape)} and reference {tuple(reference.shape)} differ")
    if delay == 0:
        return wav, reference
    return wav[..., delay:], reference[..., : reference.shape[-1] - delay]


def vad_pairs(vad_logit: Tensor, labels: Tensor, offset: int = MODEL_FRAME_OFFSET) -> tuple[Tensor, Tensor]:
    """Model VAD logits paired frame for frame with contract-frame labels."""
    logits = vad_logit[:, offset:]
    if logits.shape != labels.shape:
        raise ValueError(
            f"VAD logits {tuple(vad_logit.shape)} minus {offset} frame(s) do not match labels "
            f"{tuple(labels.shape)}"
        )
    return logits, labels


def reference_active(reference: Tensor, threshold_db: float = LossConfig.active_threshold_db) -> Tensor:
    """``[B]`` bool: the reference's mean power exceeds ``threshold_db`` dBFS."""
    return reference.square().mean(-1) > 10.0 ** (threshold_db / 10.0)


def hann_window(win: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Periodic Hann window."""
    return torch.hann_window(win, periodic=True, device=device, dtype=dtype)


def stft(x: Tensor, resolution: Resolution, window: Tensor | None = None) -> Tensor:
    """Power-normalised complex STFT ``[B, F, T]`` (Hann, centred, zero-padded).

    Divided by ``sqrt(sum(window^2))`` so ``|X|^2`` is about the signal power per bin,
    whatever the resolution.
    """
    n_fft, hop, win = resolution
    if window is None:
        window = hann_window(win, x.device, x.dtype)
    spec = torch.stft(
        x, n_fft, hop_length=hop, win_length=win, window=window, center=True,
        pad_mode="constant", normalized=False, onesided=True, return_complex=True,
    )  # fmt: skip
    return spec / window.square().sum().sqrt()


def _power(spec: Tensor) -> Tensor:
    return spec.real.square() + spec.imag.square() + POWER_EPS


# ------------------------------------------------------------------------- loss terms


def mr_stft_loss(
    est: Tensor,
    ref: Tensor,
    resolutions: tuple[Resolution, ...] = LossConfig.mr_stft_resolutions,
    *,
    log_floor: float = LossConfig.log_floor,
) -> Tensor:
    """Multi-resolution STFT loss per example ``[B]``: mean over resolutions of
    spectral convergence ``|| |S| - |Y| ||_F / || |S| ||_F`` plus log-magnitude L1."""
    total = est.new_zeros(est.shape[0])
    for res in resolutions:
        window = hann_window(res[2], est.device, est.dtype)
        p_est, p_ref = _power(stft(est, res, window)), _power(stft(ref, res, window))
        m_est, m_ref = p_est.sqrt(), p_ref.sqrt()
        num = (m_ref - m_est).square().sum(dim=(-2, -1))
        den = p_ref.sum(dim=(-2, -1))
        convergence = ((num + POWER_EPS) / (den + POWER_EPS)).sqrt()
        log_mag = 0.5 * (torch.log(p_est + log_floor) - torch.log(p_ref + log_floor)).abs().mean(dim=(-2, -1))
        total = total + convergence + log_mag
    return total / len(resolutions)


def compressed_spectral_terms(
    est: Tensor,
    ref: Tensor,
    resolution: Resolution = LossConfig.spectral_resolution,
    *,
    compression: float = LossConfig.compression,
    complex_factor: float = LossConfig.complex_factor,
    alpha: float = LossConfig.asym_alpha,
) -> tuple[Tensor, Tensor]:
    """Compressed complex spectral loss and the VoiceFilter-Lite asymmetric loss, ``[B]`` each.

    With ``c = compression``, ``Y`` the estimate's and ``S`` the reference's STFT:

    * spectral: ``mean((|Y|^c - |S|^c)^2) + complex_factor * mean(|Y_c - S_c|^2)`` where
      ``X_c = |X|^c e^{j angle X}``;
    * asymmetric: ``mean(g(|S|^c - |Y|^c)^2)`` with ``g(x) = alpha x`` for ``x > 0``
      (over-suppression) and ``g(x) = x`` otherwise.
    """
    window = hann_window(resolution[2], est.device, est.dtype)
    spec_est, spec_ref = stft(est, resolution, window), stft(ref, resolution, window)
    p_est, p_ref = _power(spec_est), _power(spec_ref)
    c_est, c_ref = p_est ** (compression / 2), p_ref ** (compression / 2)
    magnitude = (c_est - c_ref).square().mean(dim=(-2, -1))
    # X_c = X |X|^(c - 1): the phase of X with the compressed magnitude.
    scale_est, scale_ref = p_est ** ((compression - 1) / 2), p_ref ** ((compression - 1) / 2)
    diff_re = spec_est.real * scale_est - spec_ref.real * scale_ref
    diff_im = spec_est.imag * scale_est - spec_ref.imag * scale_ref
    complex_term = (diff_re.square() + diff_im.square()).mean(dim=(-2, -1))
    x = c_ref - c_est
    asym = torch.where(x > 0, alpha * x, x).square().mean(dim=(-2, -1))
    return magnitude + complex_factor * complex_term, asym


def si_sdr(est: Tensor, ref: Tensor, *, max_db: float | None = None) -> Tensor:
    """Scale-invariant SDR in dB per example ``[B]`` (zero-mean signals).

    With ``max_db`` it is the thresholded form ``10 log10(|t|^2 / (|e|^2 + tau |t|^2))``,
    ``tau = 10^(-max_db / 10)``, which saturates smoothly at ``max_db``.
    """
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    gain = (est * ref).sum(-1, keepdim=True) / (ref.square().sum(-1, keepdim=True) + SISDR_EPS)
    target = gain * ref
    t_energy = target.square().sum(-1)
    e_energy = (est - target).square().sum(-1)
    tau = 0.0 if max_db is None else 10.0 ** (-max_db / 10.0)
    return 10.0 * torch.log10((t_energy + SISDR_EPS) / (e_energy + tau * t_energy + SISDR_EPS))


def absent_energy_db(est: Tensor, mixture: Tensor, *, floor_db: float = LossConfig.absent_floor_db) -> Tensor:
    """Output energy relative to the mixture, in dB above ``floor_db`` (``>= 0``), ``[B]``."""
    floor = 10.0 ** (floor_db / 10.0)
    ratio = est.square().mean(-1) / (mixture.square().mean(-1) + ENERGY_EPS)
    return 10.0 * torch.log10(ratio + floor) - 10.0 * math.log10(floor)


def vad_bce(vad_logit: Tensor, labels: Tensor) -> Tensor:
    """Mean BCE-with-logits per example ``[B]`` over frames."""
    return F.binary_cross_entropy_with_logits(vad_logit, labels, reduction="none").mean(-1)


def _scatter(values: Tensor, index: Tensor, size: int) -> Tensor:
    """Per-example vector of length ``size``: ``values`` at ``index``, zeros elsewhere."""
    return values.new_zeros(size).index_copy(0, index, values)


# --------------------------------------------------------------------------- the loss


class EarmarkLoss:
    """The M3 training loss; call it as ``loss_fn(out, batch) -> LossOutput``.

    ``out`` is an :class:`~earmark.model.earmark_net.EarmarkOutput` (anything with ``wav``
    ``[B, N]`` and ``vad_logit`` ``[B, N / 160]``); ``batch`` holds the mixer's
    ``mixture``, ``target`` and ``vad``.
    """

    def __init__(self, config: LossConfig = LossConfig()) -> None:
        self.config = config
        self.weights = config.weights()

    def __call__(self, out: _HasWavAndVad, batch: Mapping[str, Tensor]) -> LossOutput:
        cfg = self.config
        wav = out.wav
        with fp32_region(wav.device):
            real = dsp_dtype(wav.dtype)
            est, ref = align(wav.to(real), batch["target"].to(real))
            _, mix = align(wav.to(real), batch["mixture"].to(real))
            logits, labels = vad_pairs(out.vad_logit.to(real), batch["vad"].to(real))
            size = est.shape[0]
            active = reference_active(ref, cfg.active_threshold_db)
            act_idx = active.nonzero().squeeze(1)
            sil_idx = (~active).nonzero().squeeze(1)

            per_term: dict[str, Tensor] = {}
            if act_idx.numel():
                a_est, a_ref = est.index_select(0, act_idx), ref.index_select(0, act_idx)
                per_term["mr_stft"] = _scatter(
                    mr_stft_loss(a_est, a_ref, cfg.mr_stft_resolutions, log_floor=cfg.log_floor), act_idx, size
                )
                sdr = si_sdr(a_est, a_ref, max_db=cfg.sisdr_max_db)
                per_term["sisdr"] = _scatter(cfg.sisdr_max_db - sdr, act_idx, size)
            else:
                per_term["mr_stft"] = est.new_zeros(size)
                per_term["sisdr"] = est.new_zeros(size)
            per_term["spectral"], per_term["asym"] = compressed_spectral_terms(
                est, ref, cfg.spectral_resolution, compression=cfg.compression,
                complex_factor=cfg.complex_factor, alpha=cfg.asym_alpha,
            )  # fmt: skip
            if sil_idx.numel():
                energy = absent_energy_db(
                    est.index_select(0, sil_idx), mix.index_select(0, sil_idx), floor_db=cfg.absent_floor_db
                )
                per_term["absent"] = _scatter(energy, sil_idx, size)
            else:
                per_term["absent"] = est.new_zeros(size)
            per_term["vad"] = vad_bce(logits, labels)

            per_example = sum(self.weights[name] * per_term[name] for name in TERMS)
            total = per_example.mean()
            terms = {name: per_term[name].mean().detach() for name in TERMS}
            stats = self._stats(est, ref, mix, act_idx, sil_idx, per_term["absent"])
        return LossOutput(total=total, terms=terms, stats=stats)

    @torch.no_grad()
    def _stats(
        self, est: Tensor, ref: Tensor, mix: Tensor, act_idx: Tensor, sil_idx: Tensor, absent: Tensor
    ) -> dict[str, Tensor]:
        nan = est.new_tensor(float("nan"))
        stats = {
            "n_active": est.new_tensor(float(act_idx.numel())),
            "n_silent": est.new_tensor(float(sil_idx.numel())),
            "si_sdr_db": nan,
            "si_sdr_in_db": nan,
            "si_sdri_db": nan,
            "absent_db": nan,
        }
        if act_idx.numel():
            a_ref = ref.index_select(0, act_idx)
            out_db = si_sdr(est.index_select(0, act_idx), a_ref)
            in_db = si_sdr(mix.index_select(0, act_idx), a_ref)
            stats.update(si_sdr_db=out_db.mean(), si_sdr_in_db=in_db.mean(), si_sdri_db=(out_db - in_db).mean())
        if sil_idx.numel():
            stats["absent_db"] = (absent.index_select(0, sil_idx) + self.config.absent_floor_db).mean()
        return {k: v.detach() for k, v in stats.items()}


__all__ = [
    "ENERGY_EPS",
    "POWER_EPS",
    "SISDR_EPS",
    "TERMS",
    "EarmarkLoss",
    "LossConfig",
    "LossOutput",
    "Resolution",
    "absent_energy_db",
    "align",
    "compressed_spectral_terms",
    "hann_window",
    "mr_stft_loss",
    "reference_active",
    "si_sdr",
    "stft",
    "vad_bce",
    "vad_pairs",
]
