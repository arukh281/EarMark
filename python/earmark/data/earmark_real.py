"""Earmark-Real: recording layout, clap alignment, close-talk VAD labels and transcripts.

About 30 minutes of real people on a laptop microphone in real rooms, each with a phone
held close to the talker's mouth as a close-talk reference. Layout of one release::

    earmark_real/
      speakers.json          {speaker_id: {"role": "author" | "volunteer",
                                            "accent": str, "consent_release": bool}}
      takes/<take_id>/
        meta.json            {"speaker_id", "scenario", "interferers": [...], "room", ...}
        laptop.wav           far-field laptop microphone (any rate; resampled to 16 kHz)
        closetalk.wav        phone near the talker's mouth (any rate)
        transcript.tsv       hand-corrected: start_s <TAB> end_s <TAB> speaker <TAB> text
        labels.txt           optional Audacity label track that overrides the VAD labels

Both recordings start with a hand clap. The close-talk track is aligned to the laptop
track by that clap (:func:`align_by_clap`), and its energy gives the target's frame labels
(:func:`closetalk_labels`), unless ``labels.txt`` holds hand-corrected labels. The
author's own takes are dev and volunteers' takes are test; only takes whose speaker
gave release consent belong to the published set.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from earmark import constants as C
from earmark.data.labels import apply_hangover_np, frame_energy_np, intervals_to_frames, num_frames
from earmark.data.shards import resample_to_contract, to_mono
from earmark.data.splits import EARMARK_REAL, speaker_key

__all__ = [
    "ROLES",
    "SCENARIOS",
    "PreparedTake",
    "Segment",
    "Take",
    "align_by_clap",
    "apply_lag",
    "closetalk_labels",
    "estimate_drift",
    "find_clap",
    "labels_from_segments",
    "load_audacity_labels",
    "load_speakers",
    "load_takes",
    "load_transcript",
    "prepare_take",
    "read_audio_16k",
]

ROLES: tuple[str, ...] = ("author", "volunteer")
SCENARIOS: tuple[str, ...] = ("second_person", "podcast_tv", "agent_voice", "cafe", "quiet")
_ENVELOPE_HOP_S = 0.001


@dataclass(frozen=True)
class Segment:
    """A transcript or label segment on the laptop timeline (seconds)."""

    start_s: float
    end_s: float
    speaker: str
    text: str = ""


@dataclass(frozen=True)
class Take:
    """One recorded take and where its files are."""

    take_id: str
    speaker: str
    role: str
    split: str
    accent: str
    consent_release: bool
    scenario: str
    interferers: tuple[str, ...]
    laptop_path: Path
    closetalk_path: Path
    transcript_path: Path | None
    labels_path: Path | None
    meta: Mapping[str, Any] = field(default_factory=dict)


def load_speakers(root: str | Path) -> dict[str, dict[str, Any]]:
    """``speakers.json`` with roles checked."""
    speakers = json.loads((Path(root) / "speakers.json").read_text("utf-8"))
    for sid, info in speakers.items():
        if info.get("role") not in ROLES:
            raise ValueError(f"speaker {sid}: role must be one of {ROLES}")
    return speakers


def load_takes(root: str | Path, *, released_only: bool = False) -> list[Take]:
    """All takes under ``root/takes`` (author takes are dev, volunteers' are test).

    ``released_only`` keeps only takes whose speaker consented to release.
    """
    root = Path(root)
    speakers = load_speakers(root)
    takes = []
    for meta_path in sorted((root / "takes").glob("*/meta.json")):
        d = meta_path.parent
        meta = json.loads(meta_path.read_text("utf-8"))
        sid = str(meta["speaker_id"])
        if sid not in speakers:
            raise ValueError(f"take {d.name}: unknown speaker {sid}")
        info = speakers[sid]
        scenario = str(meta.get("scenario", ""))
        if scenario not in SCENARIOS:
            raise ValueError(f"take {d.name}: scenario must be one of {SCENARIOS}")
        consent = bool(info.get("consent_release", False))
        if released_only and not consent:
            continue
        transcript = d / "transcript.tsv"
        labels = d / "labels.txt"
        takes.append(
            Take(
                take_id=d.name,
                speaker=speaker_key(EARMARK_REAL, sid),
                role=info["role"],
                split="dev" if info["role"] == "author" else "test",
                accent=str(info.get("accent", "")),
                consent_release=consent,
                scenario=scenario,
                interferers=tuple(meta.get("interferers", ())),
                laptop_path=d / "laptop.wav",
                closetalk_path=d / "closetalk.wav",
                transcript_path=transcript if transcript.exists() else None,
                labels_path=labels if labels.exists() else None,
                meta=meta,
            )
        )
    return takes


def read_audio_16k(path: str | Path, *, channel: int | None = 0) -> np.ndarray:
    """Read a file as mono float32 at the contract rate."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample_to_contract(to_mono(audio, channel if audio.shape[1] > 1 else None), sr)


# --------------------------------------------------------------------------- clap alignment


def _highpass(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.diff(x, prepend=x[:1]) if x.size else x


def _envelope(x: np.ndarray, hop: int) -> np.ndarray:
    hp = _highpass(x)
    frames = hp.size // hop
    return np.sqrt((hp[: frames * hop].reshape(frames, hop) ** 2).sum(axis=1))


def find_clap(x: np.ndarray, sample_rate: int = C.SAMPLE_RATE, *, search_s: float = 10.0) -> int:
    """Onset sample of the loudest broadband transient in the first ``search_s`` seconds.

    The first difference emphasises the clap's attack; the onset is the first sample,
    within 10 ms before the loudest 1 ms frame, whose magnitude reaches a fifth of that
    frame's peak.
    """
    seg = np.asarray(x, dtype=np.float64)[: int(search_s * sample_rate)]
    hop = max(1, int(sample_rate * _ENVELOPE_HOP_S))
    env = _envelope(seg, hop)
    if env.size == 0:
        raise ValueError("signal too short to find a clap")
    k = int(np.argmax(env))
    hp = np.abs(_highpass(seg))
    peak_pos = k * hop + int(np.argmax(hp[k * hop : (k + 1) * hop]))
    lo = max(0, peak_pos - int(0.01 * sample_rate))
    above = np.flatnonzero(hp[lo : peak_pos + 1] >= 0.2 * hp[peak_pos])
    return lo + int(above[0]) if above.size else peak_pos


def align_by_clap(
    reference: np.ndarray,
    other: np.ndarray,
    sample_rate: int = C.SAMPLE_RATE,
    *,
    search_s: float = 10.0,
    max_lag_s: float = 5.0,
    refine_s: float = 0.05,
) -> int:
    """Lag ``L`` (samples) such that ``other[n + L]`` lines up with ``reference[n]``.

    A coarse lag comes from cross-correlating 1 ms high-passed energy envelopes over the
    first ``search_s`` seconds (within ``+-max_lag_s``). It is refined to the sample by
    cross-correlating the high-passed waveforms in a ``+-refine_s`` window around the
    reference clap.
    """
    hop = max(1, int(sample_rate * _ENVELOPE_HOP_S))
    n_ref = int(search_s * sample_rate)
    ref = np.asarray(reference, dtype=np.float64)[:n_ref]
    oth = np.asarray(other, dtype=np.float64)[: n_ref + int(max_lag_s * sample_rate)]
    er = _envelope(ref, hop)
    eo = _envelope(oth, hop)
    er = er - er.mean()
    eo = eo - eo.mean()
    max_k = int(max_lag_s / _ENVELOPE_HOP_S)
    size = 1 << (er.size + eo.size).bit_length()
    corr = np.fft.irfft(np.conj(np.fft.rfft(er, size)) * np.fft.rfft(eo, size), size)
    lags = np.concatenate([np.arange(0, max_k + 1), np.arange(-max_k, 0)])
    values = corr[lags % size]
    coarse = int(lags[int(np.argmax(values))]) * hop

    clap = find_clap(ref, sample_rate, search_s=search_s)
    w = int(refine_s * sample_rate)
    a, b = max(0, clap - w), clap + w
    hp_ref = _highpass(ref)[a:b]
    hp_oth = _highpass(np.asarray(other, dtype=np.float64))
    best, best_val = coarse, -np.inf
    for lag in range(coarse - 2 * hop, coarse + 2 * hop + 1):
        lo, hi = a + lag, a + lag + hp_ref.size
        if lo < 0 or hi > hp_oth.size:
            continue
        seg = hp_oth[lo:hi]
        denom = np.linalg.norm(seg) * np.linalg.norm(hp_ref)
        val = float(np.dot(seg, hp_ref) / denom) if denom > 0 else -np.inf
        if val > best_val:
            best, best_val = lag, val
    return best


def estimate_drift(lag_start: int, lag_end: int, span_samples: int) -> float:
    """Clock ratio between two recordings from lags measured at two claps ``span`` apart."""
    if span_samples <= 0:
        raise ValueError("span must be positive")
    return 1.0 + (lag_end - lag_start) / span_samples


def apply_lag(x: np.ndarray, lag: int, length: int) -> np.ndarray:
    """``y[n] = x[n + lag]`` for ``n < length``, zero outside ``x``."""
    x = np.asarray(x)
    y = np.zeros(length, dtype=x.dtype)
    lo = max(0, -lag)
    hi = min(length, x.size - lag)
    if hi > lo:
        y[lo:hi] = x[lo + lag : hi + lag]
    return y


# --------------------------------------------------------------------------- labels and text


def closetalk_labels(
    closetalk: np.ndarray,
    *,
    threshold_db: float = -35.0,
    peak_percentile: float = 99.5,
    exclude: Sequence[tuple[int, int]] = (),
    min_active_frames: int = 3,
    hangover_frames: int = C.VAD_HANGOVER_FRAMES,
) -> np.ndarray:
    """Target frame labels from an aligned close-talk track (bool ``[F]``).

    The contract rule with two changes for real recordings: the reference is a high
    percentile of frame energy rather than the maximum (a clap or a bump must not set
    it), and the default threshold is -35 dB because the close-talk phone also picks up
    the interferers faintly. ``exclude`` sample ranges (the claps) are labelled inactive
    and ignored for the reference; runs shorter than ``min_active_frames`` are dropped.
    """
    energy = frame_energy_np(closetalk)
    frames = energy.size
    usable = np.ones(frames, dtype=bool)
    for a, b in exclude:
        t0 = max(0, (int(a) - C.WINDOW_LENGTH) // C.HOP_LENGTH + 1)
        t1 = min(frames, -(-int(b) // C.HOP_LENGTH))
        usable[t0:t1] = False
    if not usable.any():
        return np.zeros(frames, dtype=bool)
    ref = float(np.percentile(energy[usable], peak_percentile))
    active = (energy > ref * 10.0 ** (threshold_db / 10.0)) & usable
    if min_active_frames > 1:
        edges = np.diff(np.concatenate([[0], active.astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            if e - s < min_active_frames:
                active[s:e] = False
    return apply_hangover_np(active, hangover_frames)


def load_transcript(path: str | Path) -> list[Segment]:
    """Hand-corrected transcript: TSV (start, end, speaker, text; optional header) or JSON list."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        items = json.loads(path.read_text("utf-8"))
        return [Segment(float(i["start_s"]), float(i["end_s"]), str(i.get("speaker", "")), str(i.get("text", ""))) for i in items]
    out = []
    for line in path.read_text("utf-8").splitlines():
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 2:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        out.append(Segment(start, end, parts[2].strip() if len(parts) > 2 else "", parts[3].strip() if len(parts) > 3 else ""))
    return sorted(out, key=lambda s: s.start_s)


def load_audacity_labels(path: str | Path) -> list[Segment]:
    """An Audacity label track (``start<TAB>end<TAB>label``; spectral lines are skipped)."""
    out = []
    for line in Path(path).read_text("utf-8").splitlines():
        if not line.strip() or line.startswith("\\"):
            continue
        parts = line.split("\t")
        try:
            start, end = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue
        out.append(Segment(start, end, parts[2].strip() if len(parts) > 2 else ""))
    return out


def labels_from_segments(
    segments: Iterable[Segment], n_frames: int, *, speaker: str | None = None, hangover_frames: int = C.VAD_HANGOVER_FRAMES
) -> np.ndarray:
    """Frame labels from segments (all, or one speaker's), with the contract hangover."""
    spans = [(s.start_s, s.end_s) for s in segments if speaker is None or s.speaker == speaker]
    return apply_hangover_np(intervals_to_frames(spans, n_frames), hangover_frames)


@dataclass(frozen=True)
class PreparedTake:
    """A take ready for scoring: laptop audio, target labels and transcript."""

    take: Take
    laptop: np.ndarray
    vad: np.ndarray
    lag: int
    label_source: str
    transcript: tuple[Segment, ...]


def prepare_take(take: Take, *, threshold_db: float = -35.0, clap_guard_s: float = 0.5) -> PreparedTake:
    """Read, clap-align and label one take (hand labels win over close-talk labels)."""
    laptop = read_audio_16k(take.laptop_path)
    closetalk = read_audio_16k(take.closetalk_path)
    lag = align_by_clap(laptop, closetalk)
    aligned = apply_lag(closetalk, lag, laptop.size)
    frames = num_frames(laptop.size)
    if take.labels_path is not None:
        vad = labels_from_segments(load_audacity_labels(take.labels_path), frames)
        source = "hand"
    else:
        clap = find_clap(laptop)
        guard = int(clap_guard_s * C.SAMPLE_RATE)
        vad = closetalk_labels(aligned, threshold_db=threshold_db, exclude=[(max(0, clap - guard), clap + guard)])
        source = "closetalk"
    transcript = tuple(load_transcript(take.transcript_path)) if take.transcript_path else ()
    return PreparedTake(take, laptop, vad, lag, source, transcript)
