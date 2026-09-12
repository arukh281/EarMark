"""The signal contract's frame-activity rule in NumPy only (no torch import).

This is the single NumPy implementation of the contract's VAD/activity rule
(docs/CONTRACT.md, "VAD labels"). :mod:`earmark.data.labels` re-exports it next to its torch
twins for the mixer and the data parsers, and :mod:`earmark.eval.metrics` uses it for
TSOS target-present frames and interferer-only regions. Scoring worker processes import
this module without paying for a torch import.

* Frame ``t`` covers samples ``[t * HOP_LENGTH, t * HOP_LENGTH + WINDOW_LENGTH)`` (no
  centring).
* A frame's energy is that of the analysis-windowed frame, ``sum((w * x) ** 2)``, with the
  contract's periodic sqrt-Hann window ``w``.
* A frame is active when its energy is strictly greater than the reference (peak) frame
  energy scaled by ``VAD_THRESHOLD_DB``; activity is then held for
  ``VAD_HANGOVER_FRAMES`` frames.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from earmark import constants as C

__all__ = [
    "activity_from_energy_np",
    "active_power_np",
    "apply_hangover_np",
    "db_to_power_ratio",
    "frame_energy_from_frames",
    "frame_energy_np",
    "num_frames",
    "vad_labels_np",
    "window_squared_np",
]


def db_to_power_ratio(db: float) -> float:
    """Convert a level difference in dB to a power ratio."""
    return 10.0 ** (db / 10.0)


def num_frames(num_samples: int) -> int:
    """Number of contract frames in a signal of ``num_samples`` (0 if shorter than a window)."""
    if num_samples < C.WINDOW_LENGTH:
        return 0
    return 1 + (num_samples - C.WINDOW_LENGTH) // C.HOP_LENGTH


def window_squared_np() -> NDArray[np.float64]:
    """Square of the contract window, ``sin(pi * n / WINDOW_LENGTH) ** 2`` (float64)."""
    n = np.arange(C.WINDOW_LENGTH, dtype=np.float64)
    return np.sin(np.pi * n / C.WINDOW_LENGTH) ** 2


def frame_energy_from_frames(frames: ArrayLike) -> NDArray[np.float64]:
    """Windowed energy of already-framed samples (``[..., F, WINDOW_LENGTH]`` -> ``[..., F]``).

    :func:`frame_energy_np` and the streaming scorer both call this, so a frame's energy is
    the same number whichever path computed it.
    """
    f = np.asarray(frames, dtype=np.float64)
    return np.einsum("...fw,w->...f", f * f, window_squared_np())


def frame_energy_np(x: ArrayLike) -> NDArray[np.float64]:
    """Windowed energy of every contract frame of ``x`` (shape ``[..., T]`` -> ``[..., F]``)."""
    x = np.asarray(x, dtype=np.float64)
    frames = num_frames(x.shape[-1])
    if frames == 0:
        return np.zeros(x.shape[:-1] + (0,), dtype=np.float64)
    view = np.lib.stride_tricks.sliding_window_view(x, C.WINDOW_LENGTH, axis=-1)
    return frame_energy_from_frames(view[..., :: C.HOP_LENGTH, :])


def apply_hangover_np(active: ArrayLike, frames: int = C.VAD_HANGOVER_FRAMES) -> NDArray[np.bool_]:
    """Hold activity for ``frames`` frames after each active frame (along the last axis)."""
    a = np.asarray(active, dtype=bool)
    if frames <= 0 or a.shape[-1] == 0:
        return a.copy()
    pad = np.zeros(a.shape[:-1] + (frames,), dtype=bool)
    padded = np.concatenate([pad, a], axis=-1)
    return np.lib.stride_tricks.sliding_window_view(padded, frames + 1, axis=-1).any(-1)


def activity_from_energy_np(
    energy: ArrayLike,
    peak_energy: float | ArrayLike | None = None,
    *,
    threshold_db: float = C.VAD_THRESHOLD_DB,
    hangover_frames: int = C.VAD_HANGOVER_FRAMES,
) -> NDArray[np.bool_]:
    """Frame labels from frame energies ``[..., F]`` (bool ``[..., F]``).

    ``peak_energy`` is the reference frame energy (one value per leading index); ``None``
    uses the peak of ``energy`` itself. A frame is active when ``energy > peak * ratio``,
    so an all-silent signal has no active frames.
    """
    e = np.asarray(energy, dtype=np.float64)
    if peak_energy is None:
        peak = e.max(axis=-1, keepdims=True) if e.shape[-1] else e
    else:
        peak = np.asarray(peak_energy, dtype=np.float64)[..., None]
    active = e > peak * db_to_power_ratio(threshold_db)
    return apply_hangover_np(active, hangover_frames)


def vad_labels_np(
    direct: ArrayLike,
    peak_energy: float | ArrayLike | None = None,
    *,
    threshold_db: float = C.VAD_THRESHOLD_DB,
    hangover_frames: int = C.VAD_HANGOVER_FRAMES,
) -> NDArray[np.bool_]:
    """Frame labels from a direct-path signal ``[..., T]`` (bool ``[..., F]``).

    ``peak_energy`` is the utterance's peak frame energy (one value per leading index).
    When it is ``None`` the peak of ``direct`` itself is used, which is right whenever
    ``direct`` holds the whole utterance.
    """
    return activity_from_energy_np(
        frame_energy_np(direct), peak_energy, threshold_db=threshold_db, hangover_frames=hangover_frames
    )


def active_power_np(x: ArrayLike, *, threshold_db: float = C.VAD_THRESHOLD_DB) -> NDArray[np.float64]:
    """Mean power of ``x`` over its own active frames (float64; a 0-d array for 1-D input).

    Frames are active under the contract rule with ``x``'s own peak frame as the reference
    and no hangover; the power is the mean square over the samples those frames cover. A
    silent signal has power 0. This is the level definition behind every SNR and SIR.
    """
    x = np.asarray(x, dtype=np.float64)
    energy = frame_energy_np(x)
    if energy.shape[-1] == 0:
        return np.mean(x * x, axis=-1)
    peak = energy.max(axis=-1, keepdims=True)
    frame_active = (energy > peak * db_to_power_ratio(threshold_db)) & (peak > 0)
    hops = np.zeros(frame_active.shape[:-1] + (frame_active.shape[-1] + 1,), dtype=bool)
    hops[..., :-1] |= frame_active
    hops[..., 1:] |= frame_active
    mask = np.repeat(hops, C.HOP_LENGTH, axis=-1)[..., : x.shape[-1]]
    if mask.shape[-1] < x.shape[-1]:
        pad = np.zeros(mask.shape[:-1] + (x.shape[-1] - mask.shape[-1],), dtype=bool)
        mask = np.concatenate([mask, pad], axis=-1)
    num = (x * x * mask).sum(axis=-1)
    den = mask.sum(axis=-1)
    return np.where(den > 0, num / np.maximum(den, 1), 0.0)
