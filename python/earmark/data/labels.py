"""Frame-level speech-activity labels: the signal contract's VAD rule.

Everything that turns audio or timings into per-frame activity labels goes through this
module, so the training mixer, the Earmark-Synth benchmark and the real-recording parsers
share one definition (docs/CONTRACT.md, "VAD labels"):

* Frames follow the STFT framing. Frame ``t`` covers samples
  ``[t * HOP_LENGTH, t * HOP_LENGTH + WINDOW_LENGTH)`` with no centring, so a signal of
  ``n >= WINDOW_LENGTH`` samples has ``1 + (n - WINDOW_LENGTH) // HOP_LENGTH`` frames.
* A frame's energy is that of the analysis-windowed frame, ``sum((w * x) ** 2)``, with the
  contract's periodic sqrt-Hann window ``w``. By Parseval this is the energy the model's
  STFT sees in that frame.
* A frame is active when the direct-path target energy is strictly greater than the
  utterance's peak frame energy scaled by ``VAD_THRESHOLD_DB`` (-40 dB).
* Activity is then held for ``VAD_HANGOVER_FRAMES`` frames after the last active frame:
  ``label[t] = any(active[t - k] for k in 0..VAD_HANGOVER_FRAMES)``.

Timings (LibriCSS segments, hand labels) map to frames by the frame centre: frame ``t`` is
inside ``[start, end)`` when ``(t * HOP_LENGTH + WINDOW_LENGTH / 2) / SAMPLE_RATE`` is.

NumPy variants (suffix ``_np``) serve offline parsers and the metrics; they are defined
once, in :mod:`earmark.data.activity` (which never imports torch), and re-exported here.
The torch variants run on any device and serve the training mixer; tests pin them to the
NumPy ones.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from earmark import constants as C
from earmark.data.activity import (
    activity_from_energy_np,
    active_power_np,
    apply_hangover_np,
    db_to_power_ratio,
    frame_energy_np,
    num_frames,
    vad_labels_np,
    window_squared_np,
)

__all__ = [
    "activity_from_energy_np",
    "active_power",
    "active_power_np",
    "active_sample_mask",
    "apply_hangover",
    "apply_hangover_np",
    "db_to_power_ratio",
    "frame_energy",
    "frame_energy_np",
    "frames_to_intervals",
    "intervals_to_frames",
    "num_frames",
    "vad_labels",
    "vad_labels_np",
    "window_squared_np",
]


def _window_squared(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n = torch.arange(C.WINDOW_LENGTH, dtype=torch.float64)
    return (torch.sin(math.pi * n / C.WINDOW_LENGTH) ** 2).to(device=device, dtype=dtype)


def frame_energy(x: torch.Tensor) -> torch.Tensor:
    """Windowed energy of every contract frame of ``x`` (shape ``[..., T]`` -> ``[..., F]``)."""
    frames = num_frames(x.shape[-1])
    if frames == 0:
        return x.new_zeros(x.shape[:-1] + (0,))
    view = x.unfold(-1, C.WINDOW_LENGTH, C.HOP_LENGTH)
    return (view.square() * _window_squared(x.device, x.dtype)).sum(-1)


def apply_hangover(active: torch.Tensor, frames: int = C.VAD_HANGOVER_FRAMES) -> torch.Tensor:
    """Torch version of :func:`apply_hangover_np`; returns a bool tensor on the same device."""
    a = active.to(torch.bool)
    if frames <= 0 or a.shape[-1] == 0:
        return a.clone()
    lead = a.shape[:-1]
    x = a.to(torch.float32).reshape(-1, 1, a.shape[-1])
    x = F.pad(x, (frames, 0))
    y = F.max_pool1d(x, kernel_size=frames + 1, stride=1)
    return y.reshape(*lead, a.shape[-1]) > 0.5


def vad_labels(
    direct: torch.Tensor,
    peak_energy: torch.Tensor | None = None,
    *,
    threshold_db: float = C.VAD_THRESHOLD_DB,
    hangover_frames: int = C.VAD_HANGOVER_FRAMES,
) -> torch.Tensor:
    """Torch version of :func:`vad_labels_np` (bool ``[..., F]`` on ``direct``'s device)."""
    energy = frame_energy(direct)
    if peak_energy is None:
        peak = energy.amax(dim=-1, keepdim=True) if energy.shape[-1] else energy
    else:
        peak = peak_energy.to(energy.dtype).unsqueeze(-1)
    active = energy > peak * db_to_power_ratio(threshold_db)
    return apply_hangover(active, hangover_frames)


def active_sample_mask(frame_active: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Samples covered by at least one active frame (bool ``[..., F]`` -> ``[..., T]``).

    Frame ``t`` covers hops ``t`` and ``t + 1`` (``HOP_LENGTH`` samples each), so a hop
    is active when either frame that covers it is. Samples past the last full frame are
    inactive.
    """
    fa = frame_active.to(torch.bool)
    frames = fa.shape[-1]
    hops = fa.new_zeros(fa.shape[:-1] + (frames + 1,))
    if frames:
        hops[..., :-1] |= fa
        hops[..., 1:] |= fa
    mask = hops.repeat_interleave(C.HOP_LENGTH, dim=-1)
    covered = mask.shape[-1]
    if covered >= num_samples:
        return mask[..., :num_samples]
    tail = fa.new_zeros(fa.shape[:-1] + (num_samples - covered,))
    return torch.cat([mask, tail], dim=-1)


def active_power(x: torch.Tensor, *, threshold_db: float = C.VAD_THRESHOLD_DB) -> torch.Tensor:
    """Mean power of ``x`` over its own active frames (``[..., T]`` -> ``[...]``).

    Frames are active under the contract rule with ``x``'s own peak frame as the reference
    and no hangover; the power is the mean square over the samples those frames cover. A
    silent signal has power 0. This is the level definition behind every SNR and SIR the
    mixer and the benchmark generator draw.
    """
    energy = frame_energy(x)
    if energy.shape[-1] == 0:
        return x.square().mean(dim=-1)
    peak = energy.amax(dim=-1, keepdim=True)
    frame_active = (energy > peak * db_to_power_ratio(threshold_db)) & (peak > 0)
    mask = active_sample_mask(frame_active, x.shape[-1]).to(x.dtype)
    num = (x.square() * mask).sum(dim=-1)
    den = mask.sum(dim=-1)
    return torch.where(den > 0, num / den.clamp_min(1.0), torch.zeros_like(num))


def intervals_to_frames(
    intervals: Iterable[tuple[float, float]],
    n_frames: int,
    *,
    sample_rate: int = C.SAMPLE_RATE,
    offset_s: float = 0.0,
) -> np.ndarray:
    """Mark frames whose centre lies in any ``[start_s, end_s)`` interval (bool ``[F]``).

    ``offset_s`` is subtracted from every interval first (for example the clap offset of a
    recording whose timeline starts later than the labels' timeline).
    """
    centres = (np.arange(n_frames) * C.HOP_LENGTH + C.WINDOW_LENGTH / 2) / sample_rate
    active = np.zeros(n_frames, dtype=bool)
    for start, end in intervals:
        s, e = float(start) - offset_s, float(end) - offset_s
        if e > s:
            active |= (centres >= s) & (centres < e)
    return active


def frames_to_intervals(
    active: np.ndarray, *, sample_rate: int = C.SAMPLE_RATE
) -> list[tuple[float, float]]:
    """Inverse of :func:`intervals_to_frames`: runs of active frames as ``(start_s, end_s)``.

    A run of frames ``[t0, t1)`` maps to ``[(t0 * HOP + HOP / 2) / sr, (t1 * HOP + HOP / 2) / sr)``,
    the span of frame centres it contains, so converting back gives the same frames.
    """
    a = np.asarray(active, dtype=bool)
    if a.ndim != 1:
        raise ValueError("frames_to_intervals expects a 1-D label array")
    edges = np.diff(np.concatenate([[False], a, [False]]).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    half = (C.WINDOW_LENGTH - C.HOP_LENGTH) / 2
    return [
        ((t0 * C.HOP_LENGTH + half) / sample_rate, (t1 * C.HOP_LENGTH + half) / sample_rate)
        for t0, t1 in zip(starts.tolist(), ends.tolist(), strict=True)
    ]
