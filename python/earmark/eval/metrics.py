"""Objective metrics for Earmark evaluation: 16 kHz, mono, time-aligned signals.

Every function takes 1-D float arrays of equal length, clean reference first and system output
second, matching the argument order of :func:`pesq.pesq` and :func:`pystoi.stoi`. Aligning the
output (compensating a system's latency) is the adapter's job, not the metric's.

Metrics
-------
* :func:`pesq_wb` - PESQ wide-band (ITU-T P.862.2) MOS-LQO via the ``pesq`` package.
* :func:`stoi` / :func:`estoi` - (extended) short-time objective intelligibility via ``pystoi``.
* :func:`si_sdr` / :func:`si_sdr_improvement` - scale-invariant SDR (Le Roux et al. 2019).
* :func:`composite` - CSIG/CBAK/COVL (Hu and Loizou 2008) with wide-band PESQ, as used for the
  published VoiceBank+DEMAND tables.
* :func:`tsos` - target-speaker over-suppression (Eskimez et al. 2021, arXiv 2110.09625).
* :func:`interferer_suppression_db` - output-vs-input energy reduction in interferer-only regions.

Frame conventions follow the signal contract: frame ``t`` covers samples
``[t * HOP_LENGTH, t * HOP_LENGTH + WINDOW_LENGTH)`` (no centring), so frame masks here line up
with model VAD frames at ``FRAME_RATE_HZ``. This module never imports torch, so worker processes
that only score audio start quickly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from earmark import constants as C

__all__ = [
    "TSOS_COMPRESSION",
    "TSOS_GAMMA",
    "CompositeScores",
    "MetricError",
    "TsosResult",
    "activity_from_energy",
    "activity_mask",
    "composite",
    "estoi",
    "frame_energy",
    "frame_signal",
    "interferer_region",
    "interferer_suppression_db",
    "llr_frames",
    "longest_run",
    "num_frames",
    "os_flags_from_frames",
    "over_suppressed_frames",
    "pesq_wb",
    "segmental_snr_frames",
    "si_sdr",
    "si_sdr_improvement",
    "stoi",
    "tsos",
    "tsos_from_flags",
    "wss_frames",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: Spectral compression exponent ``p`` of the TSOS measure (Eskimez et al. use 0.3).
TSOS_COMPRESSION: float = 0.3
#: Over-suppression threshold ``gamma`` of the TSOS measure (Eskimez et al. use 0.1).
TSOS_GAMMA: float = 0.1

_EPS = 1e-12


class MetricError(ValueError):
    """A metric is undefined for the given input (for example a silent reference)."""


# --------------------------------------------------------------------------------------------
# Input handling and framing


def _as_1d(x: ArrayLike, name: str) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or inf")
    return a


def _pair(reference: ArrayLike, estimate: ArrayLike) -> tuple[FloatArray, FloatArray]:
    r = _as_1d(reference, "reference")
    e = _as_1d(estimate, "estimate")
    if r.shape != e.shape:
        raise ValueError(f"reference and estimate lengths differ: {r.size} vs {e.size}")
    return r, e


def num_frames(n_samples: int, frame_length: int = C.WINDOW_LENGTH, hop: int = C.HOP_LENGTH) -> int:
    """Number of complete, uncentred frames in ``n_samples`` samples."""
    return 0 if n_samples < frame_length else 1 + (n_samples - frame_length) // hop


def frame_signal(x: ArrayLike, frame_length: int = C.WINDOW_LENGTH, hop: int = C.HOP_LENGTH) -> FloatArray:
    """Read-only ``(frames, frame_length)`` view of complete uncentred frames (tail dropped)."""
    a = _as_1d(x, "signal")
    n = num_frames(a.size, frame_length, hop)
    if n == 0:
        return np.zeros((0, frame_length), dtype=np.float64)
    return sliding_window_view(a[: (n - 1) * hop + frame_length], frame_length)[::hop]


def frame_energy(x: ArrayLike) -> FloatArray:
    """Rectangular energy (sum of squares) of each contract frame."""
    frames = frame_signal(x)
    return np.einsum("ij,ij->i", frames, frames)


def _hangover(active: BoolArray, hangover_frames: int) -> BoolArray:
    """Keep each active frame's label on for ``hangover_frames`` more frames (causal)."""
    if hangover_frames <= 0 or active.size == 0:
        return active.copy()
    kernel = np.ones(hangover_frames + 1, dtype=np.int64)
    return np.convolve(active.astype(np.int64), kernel)[: active.size] > 0


def activity_from_energy(
    energy: ArrayLike,
    threshold_db: float = C.VAD_THRESHOLD_DB,
    hangover_frames: int = C.VAD_HANGOVER_FRAMES,
) -> BoolArray:
    """Activity labels from per-frame energies, relative to the loudest frame.

    A frame is active when ``10 log10(E_t / max_t E_t) > threshold_db`` (the contract uses
    -40 dB relative to the utterance peak), then the label is held for ``hangover_frames``
    frames after every active frame. An all-silent signal has no active frames.
    """
    e = np.asarray(energy, dtype=np.float64)
    if e.size == 0 or float(e.max()) <= 0.0:
        return np.zeros(e.shape, dtype=bool)
    level_db = 10.0 * np.log10(np.maximum(e, 1e-300) / float(e.max()))
    return _hangover(level_db > threshold_db, hangover_frames)


def activity_mask(
    signal: ArrayLike,
    threshold_db: float = C.VAD_THRESHOLD_DB,
    hangover_frames: int = C.VAD_HANGOVER_FRAMES,
) -> BoolArray:
    """Contract-style activity labels (one per frame) from a clean or direct-path signal."""
    return activity_from_energy(frame_energy(signal), threshold_db, hangover_frames)


def longest_run(mask: ArrayLike) -> int:
    """Length of the longest run of consecutive ``True`` values."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0
    padded = np.concatenate(([False], m, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return int(np.max(edges[1::2] - edges[0::2]))


# --------------------------------------------------------------------------------------------
# SI-SDR


def si_sdr(reference: ArrayLike, estimate: ArrayLike, *, zero_mean: bool = True) -> float:
    """Scale-invariant SDR in dB: ``10 log10(|a s|^2 / |a s - y|^2)`` with the optimal ``a``.

    ``zero_mean=True`` removes each signal's mean first (the usual SI-SNR convention).
    Raises :class:`MetricError` when the reference is silent.
    """
    r, e = _pair(reference, estimate)
    if zero_mean:
        r = r - r.mean()
        e = e - e.mean()
    rr = float(np.dot(r, r))
    if rr <= _EPS:
        raise MetricError("reference is silent; SI-SDR is undefined")
    target = (float(np.dot(e, r)) / rr) * r
    noise = e - target
    return 10.0 * math.log10((float(np.dot(target, target)) + _EPS) / (float(np.dot(noise, noise)) + _EPS))


def si_sdr_improvement(
    reference: ArrayLike, estimate: ArrayLike, mixture: ArrayLike, *, zero_mean: bool = True
) -> float:
    """SI-SDRi in dB: SI-SDR of the output minus SI-SDR of the unprocessed mixture."""
    return si_sdr(reference, estimate, zero_mean=zero_mean) - si_sdr(reference, mixture, zero_mean=zero_mean)


# --------------------------------------------------------------------------------------------
# PESQ and STOI


def pesq_wb(reference: ArrayLike, estimate: ArrayLike, sample_rate: int = C.SAMPLE_RATE) -> float:
    """PESQ wide-band MOS-LQO (P.862.2), roughly 1.04 to 4.64.

    Raises :class:`MetricError` if PESQ finds no utterance (for example silent input).
    """
    from pesq import PesqError, pesq  # local import keeps module import light

    if sample_rate != 16000:
        raise ValueError(f"PESQ-WB needs 16 kHz audio, got {sample_rate} Hz")
    r, e = _pair(reference, estimate)
    if not np.any(r) or not np.any(e):
        raise MetricError("PESQ is undefined for all-zero input")
    try:
        return float(pesq(sample_rate, r, e, "wb"))
    except PesqError as exc:  # NoUtterancesError, BufferTooShortError, ...
        raise MetricError(f"PESQ failed: {type(exc).__name__}: {exc}") from exc


def stoi(reference: ArrayLike, estimate: ArrayLike, sample_rate: int = C.SAMPLE_RATE) -> float:
    """Classic STOI (Taal et al. 2011) in [0, 1]; this is the '0.921' of the VB-DEMAND tables."""
    from pystoi import stoi as _stoi

    r, e = _pair(reference, estimate)
    return float(_stoi(r, e, sample_rate, extended=False))


def estoi(reference: ArrayLike, estimate: ArrayLike, sample_rate: int = C.SAMPLE_RATE) -> float:
    """Extended STOI (Jensen and Taal 2016)."""
    from pystoi import stoi as _stoi

    r, e = _pair(reference, estimate)
    return float(_stoi(r, e, sample_rate, extended=True))


# --------------------------------------------------------------------------------------------
# Composite measures (CSIG, CBAK, COVL)
#
# Vectorised port of Loizou's composite.m (as used by SEGAN, MetricGAN and CMGAN for the
# VoiceBank+DEMAND tables): 30 ms Hann frames with a 7.5 ms hop, WSS over 25 critical bands,
# LLR with LPC order 16 at 16 kHz, segmental SNR clipped to [-10, 35] dB, and the lowest 95%
# of frame WSS/LLR values averaged. Frames are indexed 0 .. (n - win) // hop - 1, as in the
# reference implementation (which drops the last complete frame).

_CRIT_CENTRE_HZ = np.array(
    [50.0, 120.0, 190.0, 260.0, 330.0, 400.0, 470.0, 540.0, 617.372, 703.378, 798.717, 904.128,
     1020.38, 1148.30, 1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 2211.08, 2446.71, 2701.97,
     2978.04, 3276.17, 3597.63]
)  # fmt: skip
_CRIT_BANDWIDTH_HZ = np.array(
    [70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 77.3724, 86.0056, 95.3398, 105.411, 116.256,
     127.914, 140.423, 153.823, 168.154, 183.457, 199.776, 217.153, 235.631, 255.255, 276.072,
     298.126, 321.465, 346.136]
)  # fmt: skip


def _composite_frames(x: FloatArray, sample_rate: int) -> FloatArray:
    win = int(round(30 * sample_rate / 1000))
    hop = win // 4
    n = max((x.size - win) // hop, 0)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(1, win + 1) / (win + 1)))
    if n == 0:
        return np.zeros((0, win))
    frames = sliding_window_view(x[: (n - 1) * hop + win], win)[::hop]
    return frames * window


def _mean_lowest(values: FloatArray, fraction: float = 0.95) -> float:
    v = np.sort(values[np.isfinite(values)])
    if v.size == 0:
        return float("nan")
    count = max(int(math.floor(v.size * fraction + 0.5)), 1)
    return float(v[:count].mean())


def wss_frames(reference: ArrayLike, estimate: ArrayLike, sample_rate: int = C.SAMPLE_RATE) -> FloatArray:
    """Per-frame weighted spectral slope distance (Klatt 1982), as in composite.m."""
    r, e = _pair(reference, estimate)
    fr = _composite_frames(r, sample_rate)
    fe = _composite_frames(e, sample_rate)
    if fr.shape[0] == 0:
        return np.zeros(0)
    win = fr.shape[1]
    n_fft = int(2 ** math.ceil(math.log2(2 * win)))
    half = n_fft // 2
    max_freq = sample_rate / 2.0
    num_crit = _CRIT_CENTRE_HZ.size
    min_factor = math.exp(-30.0 / (2.0 * 2.303))
    j = np.arange(half)
    f0 = np.floor(_CRIT_CENTRE_HZ / max_freq * half)
    bw = _CRIT_BANDWIDTH_HZ / max_freq * half
    norm = np.log(_CRIT_BANDWIDTH_HZ[0]) - np.log(_CRIT_BANDWIDTH_HZ)
    filt = np.exp(-11.0 * ((j[None, :] - f0[:, None]) / bw[:, None]) ** 2 + norm[:, None])
    filt = np.where(filt > min_factor, filt, 0.0)

    def band_db(frames: FloatArray) -> FloatArray:
        spec = np.abs(np.fft.fft(frames, n_fft, axis=1)[:, :half]) ** 2
        return 10.0 * np.log10(np.maximum(spec @ filt.T, 1e-10))

    er, ee = band_db(fr), band_db(fe)
    sr, se = np.diff(er, axis=1), np.diff(ee, axis=1)  # (frames, 24)
    k = num_crit - 1
    idx = np.arange(k)

    def nearest_peak(energy: FloatArray, slope: FloatArray) -> FloatArray:
        # Right search: first n >= i with slope[n] <= 0 (or k); value energy[n - 1].
        right_pos = np.where(slope <= 0, idx, k)
        right = np.minimum.accumulate(right_pos[:, ::-1], axis=1)[:, ::-1]
        # Left search: last n <= i with slope[n] > 0 (or -1); value energy[n + 1].
        left_pos = np.where(slope > 0, idx, -1)
        left = np.maximum.accumulate(left_pos, axis=1)
        pick = np.where(slope > 0, right - 1, left + 1)
        return np.take_along_axis(energy, pick, axis=1)

    pr, pe = nearest_peak(er, sr), nearest_peak(ee, se)
    k_max, k_locmax = 20.0, 1.0
    w_r = (k_max / (k_max + er.max(axis=1, keepdims=True) - er[:, :k])) * (k_locmax / (k_locmax + pr - er[:, :k]))
    w_e = (k_max / (k_max + ee.max(axis=1, keepdims=True) - ee[:, :k])) * (k_locmax / (k_locmax + pe - ee[:, :k]))
    w = 0.5 * (w_r + w_e)
    return np.sum(w * (sr - se) ** 2, axis=1) / np.sum(w, axis=1)


def _autocorr(frames: FloatArray, order: int) -> FloatArray:
    n = frames.shape[1]
    return np.stack([np.einsum("ij,ij->i", frames[:, : n - k], frames[:, k:]) for k in range(order + 1)], axis=1)


def _levinson(r: FloatArray) -> FloatArray:
    """LPC polynomials ``[1, -a_1, ..., -a_P]`` from autocorrelations (frames, P + 1)."""
    frames, p1 = r.shape
    order = p1 - 1
    a = np.zeros((frames, order))
    err = r[:, 0].copy()
    for i in range(order):
        acc = r[:, i + 1] - np.einsum("ij,ij->i", a[:, :i], r[:, i:0:-1]) if i else r[:, 1].copy()
        k = acc / err
        prev = a[:, :i].copy()
        a[:, i] = k
        if i:
            a[:, :i] = prev - k[:, None] * prev[:, ::-1]
        err = (1.0 - k * k) * err
    return np.concatenate([np.ones((frames, 1)), -a], axis=1)


def llr_frames(reference: ArrayLike, estimate: ArrayLike, sample_rate: int = C.SAMPLE_RATE) -> FloatArray:
    """Per-frame log-likelihood ratio (Itakura) as in composite.m; silent frames give NaN."""
    r, e = _pair(reference, estimate)
    fr = _composite_frames(r, sample_rate)
    fe = _composite_frames(e, sample_rate)
    if fr.shape[0] == 0:
        return np.zeros(0)
    order = 10 if sample_rate < 10000 else 16
    rr, re = _autocorr(fr, order), _autocorr(fe, order)
    ok = (rr[:, 0] > 0) & (re[:, 0] > 0)
    out = np.full(fr.shape[0], np.nan)
    if not ok.any():
        return out
    rr, re = rr[ok], re[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        a_r, a_e = _levinson(rr), _levinson(re)
        lag = np.abs(np.arange(order + 1)[:, None] - np.arange(order + 1)[None, :])
        toeplitz_r = rr[:, lag]  # (frames, P+1, P+1)
        num = np.einsum("fi,fij,fj->f", a_e, toeplitz_r, a_e)
        den = np.einsum("fi,fij,fj->f", a_r, toeplitz_r, a_r)
        out[ok] = np.log(num / den)
    return out


def segmental_snr_frames(
    reference: ArrayLike, estimate: ArrayLike, sample_rate: int = C.SAMPLE_RATE
) -> FloatArray:
    """Per-frame segmental SNR in dB, clipped to [-10, 35] as in composite.m."""
    r, e = _pair(reference, estimate)
    fr = _composite_frames(r, sample_rate)
    fe = _composite_frames(e, sample_rate)
    eps = np.finfo(np.float64).eps
    signal = np.sum(fr**2, axis=1)
    noise = np.sum((fr - fe) ** 2, axis=1)
    return np.clip(10.0 * np.log10(signal / (noise + eps) + eps), -10.0, 35.0)


@dataclass(frozen=True)
class CompositeScores:
    """Hu and Loizou composite measures plus their ingredients."""

    csig: float
    cbak: float
    covl: float
    pesq_wb: float
    llr: float
    wss: float
    segsnr: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def composite(
    reference: ArrayLike,
    estimate: ArrayLike,
    sample_rate: int = C.SAMPLE_RATE,
    *,
    pesq_score: float | None = None,
) -> CompositeScores:
    """CSIG, CBAK and COVL (each clipped to [1, 5]) using wide-band PESQ.

    Pass ``pesq_score`` to reuse an already computed PESQ-WB value.
    """
    r, e = _pair(reference, estimate)
    p = pesq_wb(r, e, sample_rate) if pesq_score is None else float(pesq_score)
    wss = _mean_lowest(wss_frames(r, e, sample_rate))
    llr = _mean_lowest(llr_frames(r, e, sample_rate))
    seg = segmental_snr_frames(r, e, sample_rate)
    segsnr = float(seg.mean()) if seg.size else float("nan")
    csig = float(np.clip(3.093 - 1.029 * llr + 0.603 * p - 0.009 * wss, 1.0, 5.0))
    cbak = float(np.clip(1.634 + 0.478 * p - 0.007 * wss + 0.063 * segsnr, 1.0, 5.0))
    covl = float(np.clip(1.594 + 0.805 * p - 0.512 * llr - 0.007 * wss, 1.0, 5.0))
    return CompositeScores(csig=csig, cbak=cbak, covl=covl, pesq_wb=p, llr=llr, wss=wss, segsnr=segsnr)


# --------------------------------------------------------------------------------------------
# Target-speaker over-suppression (TSOS)


def _sqrt_hann_periodic(length: int = C.WINDOW_LENGTH) -> FloatArray:
    return np.sin(np.pi * np.arange(length) / length)


def over_suppressed_frames(
    reference: ArrayLike,
    estimate: ArrayLike,
    *,
    p: float = TSOS_COMPRESSION,
    gamma: float = TSOS_GAMMA,
) -> BoolArray:
    """Per-frame over-suppression flags, Eskimez et al. (2021) eq. (2)-(3), scale-invariant form.

    With compressed magnitudes ``A = |S|^p`` (clean) and ``B = |S_hat|^p`` (output) on the
    contract STFT, frame ``t`` is over-suppressed when
    ``sum_f max(A - B, 0)^2 > gamma * sum_f A^2``.

    The paper writes the right-hand side as ``gamma * sum_f |S|^p``; that mixes powers of the
    magnitude, so its verdict would change with the signal's absolute level. Squaring the
    compressed magnitude keeps the test level-independent. With ``p = 0.3`` and
    ``gamma = 0.1``, a frame whose target is attenuated uniformly by more than about 11 dB
    counts as over-suppressed. Frames where the reference is digital silence are never flagged.
    """
    fr = frame_signal(reference)
    fe = frame_signal(estimate)
    if fr.shape != fe.shape:
        raise ValueError("reference and estimate lengths differ")
    return os_flags_from_frames(fr, fe, p=p, gamma=gamma)


def os_flags_from_frames(
    reference_frames: FloatArray,
    estimate_frames: FloatArray,
    *,
    p: float = TSOS_COMPRESSION,
    gamma: float = TSOS_GAMMA,
) -> BoolArray:
    """:func:`over_suppressed_frames` on pre-cut ``(frames, WINDOW_LENGTH)`` arrays (for streaming)."""
    window = _sqrt_hann_periodic()
    a = np.abs(np.fft.rfft(reference_frames * window, n=C.N_FFT, axis=1)) ** p
    b = np.abs(np.fft.rfft(estimate_frames * window, n=C.N_FFT, axis=1)) ** p
    loss = np.sum(np.maximum(a - b, 0.0) ** 2, axis=1)
    return loss > gamma * np.sum(a * a, axis=1)


@dataclass(frozen=True)
class TsosResult:
    """Frame-level TSOS summary over target-active frames.

    ``percent`` is the headline number (H2 pre-registers ``<= 2%``); ``total_os_s`` and
    ``max_os_s`` are the total and longest over-suppressed durations the paper also defines.
    """

    percent: float
    active_frames: int
    os_frames: int
    total_os_s: float
    max_os_s: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def tsos_from_flags(os_flags: ArrayLike, active: ArrayLike) -> TsosResult:
    """Summarise per-frame over-suppression flags over the target-active frames."""
    flags = np.asarray(os_flags, dtype=bool)
    act = np.asarray(active, dtype=bool)
    if flags.shape != act.shape:
        raise ValueError(f"flag and activity masks differ in shape: {flags.shape} vs {act.shape}")
    hit = flags & act
    n_active = int(act.sum())
    n_os = int(hit.sum())
    percent = 100.0 * n_os / n_active if n_active else float("nan")
    return TsosResult(
        percent=percent,
        active_frames=n_active,
        os_frames=n_os,
        total_os_s=n_os / C.FRAME_RATE_HZ,
        max_os_s=longest_run(hit) / C.FRAME_RATE_HZ,
    )


def tsos(
    reference: ArrayLike,
    estimate: ArrayLike,
    *,
    active: ArrayLike | None = None,
    p: float = TSOS_COMPRESSION,
    gamma: float = TSOS_GAMMA,
) -> TsosResult:
    """Target-speaker over-suppression over target-active frames.

    ``active`` is a boolean frame mask (one per contract frame); by default it is
    :func:`activity_mask` of the clean reference (-40 dB of peak, 50 ms hangover), so TSOS is
    measured on target-present segments only, as H2 requires.
    """
    r, e = _pair(reference, estimate)
    flags = over_suppressed_frames(r, e, p=p, gamma=gamma)
    act = activity_mask(r) if active is None else np.asarray(active, dtype=bool)
    return tsos_from_flags(flags, act)


# --------------------------------------------------------------------------------------------
# Interferer suppression


def interferer_region(reference: ArrayLike | None, interferer: ArrayLike) -> BoolArray:
    """Frames where the interferer is active and the target is not (target-absent: interferer only)."""
    region = activity_mask(interferer)
    if reference is not None:
        target = activity_mask(reference)
        if target.shape != region.shape:
            raise ValueError("reference and interferer lengths differ")
        region = region & ~target
    return region


def interferer_suppression_db(mixture: ArrayLike, estimate: ArrayLike, region: ArrayLike) -> float:
    """Energy reduction (dB) from input to output over the frames in ``region``.

    ``10 log10(sum E_mix / sum E_out)`` over interferer-only frames: 0 dB means the
    interferer passes untouched; larger is more suppression. An all-zero output is floored at
    120 dB. Returns NaN when the region is empty or silent.
    """
    m, e = _pair(mixture, estimate)
    mask = np.asarray(region, dtype=bool)
    em_all, ee_all = frame_energy(m), frame_energy(e)
    if mask.shape != em_all.shape:
        raise ValueError(f"region has {mask.size} frames, signals have {em_all.size}")
    em = float(em_all[mask].sum())
    if em <= 0.0:
        return float("nan")
    ee = max(float(ee_all[mask].sum()), em * 1e-12)
    return 10.0 * math.log10(em / ee)
