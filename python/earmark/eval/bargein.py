"""Barge-in scoring: the one event definition used everywhere, plus the numbers built on it.

Event definition (from the signal contract)
-------------------------------------------
A *barge-in onset* is a run of at least ``BARGEIN_MIN_ACTIVE_FRAMES`` (200 ms) consecutive
active VAD frames that directly follows a run of at least ``BARGEIN_MIN_SILENCE_FRAMES``
(300 ms) consecutive inactive frames. The stream is treated as preceded by silence
(``initial_silence=True``), because a voice agent starts listening with the user quiet.

Each :class:`Onset` has a ``start`` (first active frame of the run) and a ``decision`` frame
(the frame on which the 200 ms criterion is met, ``start + min_active - 1``). A causal
detector can act at the end of the decision frame, so a perfect detector's onset delay is
exactly ``min_active`` frames = 200 ms. :class:`OnsetDetector` is the streaming form with O(1)
state; :func:`detect_onsets` is the vectorised form, and both give identical events.

Scores (paired, per the evaluation plan)
----------------------------------------
* **False barge-ins per minute**: detected onsets that match no reference onset and whose
  decision frame falls where the reference target is silent, divided by the minutes of
  target silence.
* **Onset recall**: reference onsets matched by a detection whose decision frame lies in
  ``[start, start + max_delay_frames)``; **onset delay** is ``decision + 1 - start`` frames.
* **Frame recall**: fraction of reference-active frames the detector marks active. Thresholds
  are frozen on dev at a matched recall (95% frame recall by default) with
  :func:`matched_threshold` and then applied unchanged to test.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from earmark import constants as C

__all__ = [
    "DEFAULT_MAX_DELAY_FRAMES",
    "BargeInCounts",
    "FrozenThreshold",
    "Onset",
    "OnsetDetector",
    "binarize",
    "detect_onsets",
    "frame_recall_threshold",
    "matched_threshold",
    "onset_recall_threshold",
    "pool_counts",
    "score_bargein",
]

#: Detections confirmed more than 1 s after the reference onset do not count as hits.
DEFAULT_MAX_DELAY_FRAMES: int = C.FRAME_RATE_HZ

BoolArray = NDArray[np.bool_]


class Onset(NamedTuple):
    """One barge-in event, in frame indices."""

    start: int
    decision: int


def _as_bool(x: ArrayLike, name: str) -> BoolArray:
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    return a.astype(bool, copy=False)


def binarize(scores: ArrayLike, threshold: float) -> BoolArray:
    """Frame decisions from VAD probabilities: active when ``score >= threshold``."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {s.shape}")
    return s >= threshold


def detect_onsets(
    active: ArrayLike,
    *,
    min_active_frames: int = C.BARGEIN_MIN_ACTIVE_FRAMES,
    min_silence_frames: int = C.BARGEIN_MIN_SILENCE_FRAMES,
    initial_silence: bool = True,
) -> list[Onset]:
    """All barge-in onsets in a boolean frame sequence (see the module docstring)."""
    a = _as_bool(active, "active")
    if min_active_frames < 1 or min_silence_frames < 0:
        raise ValueError("min_active_frames must be >= 1 and min_silence_frames >= 0")
    if a.size == 0:
        return []
    change = np.flatnonzero(a[1:] != a[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [a.size])))
    values = a[starts]
    onsets: list[Onset] = []
    for i in np.flatnonzero(values & (lengths >= min_active_frames)):
        if i == 0:
            armed = initial_silence
        else:  # runs alternate, so run i - 1 is silence
            armed = lengths[i - 1] >= min_silence_frames or (i == 1 and initial_silence)
        if armed:
            start = int(starts[i])
            onsets.append(Onset(start, start + min_active_frames - 1))
    return onsets


class OnsetDetector:
    """Streaming barge-in detector: push one frame decision at a time, O(1) state.

    This is the reference for the engine and the web Gate markers; it fires on exactly the
    frames :func:`detect_onsets` reports as ``decision``.
    """

    __slots__ = ("_active_run", "_armed", "_frame", "_run_start", "_silence", "min_active_frames", "min_silence_frames")

    def __init__(
        self,
        *,
        min_active_frames: int = C.BARGEIN_MIN_ACTIVE_FRAMES,
        min_silence_frames: int = C.BARGEIN_MIN_SILENCE_FRAMES,
        initial_silence: bool = True,
    ) -> None:
        if min_active_frames < 1 or min_silence_frames < 0:
            raise ValueError("min_active_frames must be >= 1 and min_silence_frames >= 0")
        self.min_active_frames = min_active_frames
        self.min_silence_frames = min_silence_frames
        self._silence = min_silence_frames if initial_silence else 0  # saturating counter
        self._active_run = 0
        self._armed = False
        self._run_start = -1
        self._frame = -1

    def push(self, active: bool) -> Onset | None:
        """Consume the next frame; return the onset if this frame is a decision frame."""
        self._frame += 1
        if not active:
            self._active_run = 0
            self._silence = min(self._silence + 1, self.min_silence_frames)
            return None
        if self._active_run == 0:
            self._armed = self._silence >= self.min_silence_frames
            self._run_start = self._frame
        self._active_run += 1
        self._silence = 0
        if self._armed and self._active_run == self.min_active_frames:
            return Onset(self._run_start, self._frame)
        return None


@dataclass
class BargeInCounts:
    """Additive barge-in tallies for one item; sum items with :func:`pool_counts`."""

    reference_onsets: int = 0
    detected_onsets: int = 0
    hits: int = 0
    false_barge_ins: int = 0
    silent_frames: int = 0
    active_frames: int = 0
    detected_active_frames: int = 0
    delays_frames: list[int] = field(default_factory=list)
    frame_rate_hz: int = C.FRAME_RATE_HZ

    @property
    def silent_minutes(self) -> float:
        return self.silent_frames / (60.0 * self.frame_rate_hz)

    @property
    def false_per_minute(self) -> float:
        """False barge-ins per minute of target silence (NaN without any silence)."""
        return self.false_barge_ins / self.silent_minutes if self.silent_frames else float("nan")

    @property
    def onset_recall(self) -> float:
        return self.hits / self.reference_onsets if self.reference_onsets else float("nan")

    @property
    def frame_recall(self) -> float:
        return self.detected_active_frames / self.active_frames if self.active_frames else float("nan")

    @property
    def median_delay_ms(self) -> float:
        if not self.delays_frames:
            return float("nan")
        return float(np.median(self.delays_frames)) * 1000.0 / self.frame_rate_hz

    def summary(self) -> dict[str, float | int]:
        """JSON-friendly headline numbers plus the raw tallies needed to pool or bootstrap."""
        return {
            "false_barge_ins_per_min": self.false_per_minute,
            "onset_recall": self.onset_recall,
            "frame_recall": self.frame_recall,
            "median_onset_delay_ms": self.median_delay_ms,
            "reference_onsets": self.reference_onsets,
            "detected_onsets": self.detected_onsets,
            "hits": self.hits,
            "false_barge_ins": self.false_barge_ins,
            "silent_minutes": self.silent_minutes,
            "active_frames": self.active_frames,
            "detected_active_frames": self.detected_active_frames,
        }


def score_bargein(
    detected_active: ArrayLike,
    reference_active: ArrayLike,
    *,
    max_delay_frames: int = DEFAULT_MAX_DELAY_FRAMES,
    min_active_frames: int = C.BARGEIN_MIN_ACTIVE_FRAMES,
    min_silence_frames: int = C.BARGEIN_MIN_SILENCE_FRAMES,
    initial_silence: bool = True,
) -> BargeInCounts:
    """Compare detector frame decisions with reference target activity for one item."""
    det = _as_bool(detected_active, "detected_active")
    ref = _as_bool(reference_active, "reference_active")
    if det.shape != ref.shape:
        raise ValueError(f"detected and reference masks differ in length: {det.size} vs {ref.size}")
    kw = {"min_active_frames": min_active_frames, "min_silence_frames": min_silence_frames, "initial_silence": initial_silence}
    ref_onsets = detect_onsets(ref, **kw)
    det_onsets = detect_onsets(det, **kw)

    matched = [False] * len(det_onsets)
    delays: list[int] = []
    j0 = 0
    for ro in ref_onsets:
        while j0 < len(det_onsets) and det_onsets[j0].decision < ro.start:
            j0 += 1
        for j in range(j0, len(det_onsets)):
            d = det_onsets[j].decision
            if d >= ro.start + max_delay_frames:
                break
            if not matched[j]:
                matched[j] = True
                delays.append(d + 1 - ro.start)
                break
    false = sum(1 for m, o in zip(matched, det_onsets, strict=True) if not m and not ref[o.decision])
    return BargeInCounts(
        reference_onsets=len(ref_onsets),
        detected_onsets=len(det_onsets),
        hits=len(delays),
        false_barge_ins=false,
        silent_frames=int((~ref).sum()),
        active_frames=int(ref.sum()),
        detected_active_frames=int((det & ref).sum()),
        delays_frames=delays,
    )


def pool_counts(items: Iterable[BargeInCounts]) -> BargeInCounts:
    """Sum per-item tallies (rates are ratios of sums, not means of per-item rates)."""
    total = BargeInCounts()
    for c in items:
        total.reference_onsets += c.reference_onsets
        total.detected_onsets += c.detected_onsets
        total.hits += c.hits
        total.false_barge_ins += c.false_barge_ins
        total.silent_frames += c.silent_frames
        total.active_frames += c.active_frames
        total.detected_active_frames += c.detected_active_frames
        total.delays_frames.extend(c.delays_frames)
    return total


# --------------------------------------------------------------------------------------------
# Thresholds matched on dev


@dataclass(frozen=True)
class FrozenThreshold:
    """A VAD threshold chosen on dev; record it before the pre-registration commit."""

    threshold: float
    level: Literal["frame", "onset"]
    target_recall: float
    achieved_recall: float
    n_positives: int
    split: str = "dev"

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "threshold": self.threshold,
            "level": self.level,
            "target_recall": self.target_recall,
            "achieved_recall": self.achieved_recall,
            "n_positives": self.n_positives,
            "split": self.split,
        }


def _pairs(scores: Sequence[ArrayLike], labels: Sequence[ArrayLike]) -> list[tuple[NDArray[np.float64], BoolArray]]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have one entry per item")
    out = []
    for s, lab in zip(scores, labels, strict=True):
        sa = np.asarray(s, dtype=np.float64)
        la = _as_bool(lab, "labels")
        if sa.shape != la.shape:
            raise ValueError("each score array must match its label array")
        out.append((sa, la))
    return out


def frame_recall_threshold(
    scores: Sequence[ArrayLike], labels: Sequence[ArrayLike], target_recall: float = 0.95
) -> FrozenThreshold:
    """Largest threshold whose pooled frame recall on reference-active frames is >= target."""
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    pairs = _pairs(scores, labels)
    pos = np.concatenate([s[lab] for s, lab in pairs]) if pairs else np.zeros(0)
    if pos.size == 0:
        raise ValueError("no reference-active frames to match recall on")
    k = math.ceil(target_recall * pos.size - 1e-9)
    threshold = float(np.sort(pos)[::-1][k - 1])
    achieved = float(np.mean(pos >= threshold))
    return FrozenThreshold(threshold, "frame", target_recall, achieved, int(pos.size))


def onset_recall_threshold(
    scores: Sequence[ArrayLike],
    labels: Sequence[ArrayLike],
    target_recall: float = 0.95,
    *,
    n_candidates: int = 512,
    max_delay_frames: int = DEFAULT_MAX_DELAY_FRAMES,
) -> FrozenThreshold:
    """Largest candidate threshold whose pooled onset recall is >= target.

    Onset recall is not strictly monotone in the threshold (lower thresholds can merge runs),
    so candidates (quantiles of all scores) are scanned from high to low.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    pairs = _pairs(scores, labels)
    all_scores = np.concatenate([s for s, _ in pairs])
    candidates = np.unique(np.quantile(all_scores, np.linspace(0.0, 1.0, n_candidates + 1)))[::-1]
    n_ref = sum(len(detect_onsets(lab)) for _, lab in pairs)
    if n_ref == 0:
        raise ValueError("no reference onsets to match recall on")
    for t in candidates:
        counts = pool_counts(score_bargein(s >= t, lab, max_delay_frames=max_delay_frames) for s, lab in pairs)
        if counts.onset_recall >= target_recall:
            return FrozenThreshold(float(t), "onset", target_recall, counts.onset_recall, n_ref)
    t = float(candidates[-1])
    counts = pool_counts(score_bargein(s >= t, lab, max_delay_frames=max_delay_frames) for s, lab in pairs)
    return FrozenThreshold(t, "onset", target_recall, counts.onset_recall, n_ref)


def matched_threshold(
    scores: Sequence[ArrayLike],
    labels: Sequence[ArrayLike],
    target_recall: float = 0.95,
    *,
    level: Literal["frame", "onset"] = "frame",
) -> FrozenThreshold:
    """Freeze a VAD threshold on dev at a matched recall (95% frame recall by default)."""
    if level == "frame":
        return frame_recall_threshold(scores, labels, target_recall)
    if level == "onset":
        return onset_recall_threshold(scores, labels, target_recall)
    raise ValueError(f"level must be 'frame' or 'onset', got {level!r}")
