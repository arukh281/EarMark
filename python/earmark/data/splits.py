"""Speaker namespaces, split-disjointness checks and held-out speaker lists.

Every speaker in every manifest is stored as a namespaced key ``"<namespace>:<id>"``.
LibriSpeech, LibriTTS(-R) and LibriCSS keep LibriSpeech's speaker numbering, so they share
the ``libri`` namespace: that is what lets one check catch a LibriCSS or test.clean
speaker leaking into training. VoiceBank-DEMAND's test speakers are VCTK speakers.

Every prepared training dataset carries the list of held-out speakers it was checked
against (``heldout_speakers.json``, :func:`save_speaker_list`), so the trainer can repeat
the check offline (:meth:`earmark.data.mixer.HeldOut.with_speakers`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "EARMARK_REAL",
    "HELDOUT_SPEAKERS_FILE",
    "KOKORO",
    "LIBRI",
    "SVARAH",
    "SplitLeakError",
    "VB_TEST_SPEAKERS",
    "VB_TEST_SPEAKER_IDS",
    "VCTK",
    "check_speaker_disjoint",
    "load_speaker_list",
    "save_speaker_list",
    "speaker_key",
    "split_speaker_key",
]

LIBRI = "libri"
VCTK = "vctk"
KOKORO = "kokoro"
SVARAH = "svarah"
EARMARK_REAL = "earmark_real"

#: VoiceBank-DEMAND test speakers (raw VCTK ids); excluded from all training data.
VB_TEST_SPEAKER_IDS: tuple[str, ...] = ("p232", "p257")
#: The same speakers as namespaced keys.
VB_TEST_SPEAKERS: tuple[str, ...] = tuple(f"{VCTK}:{s}" for s in VB_TEST_SPEAKER_IDS)
#: File name of the held-out speaker list stored with each prepared training dataset.
HELDOUT_SPEAKERS_FILE = "heldout_speakers.json"


class SplitLeakError(ValueError):
    """Raised when training data overlaps a held-out speaker, voice or noise set."""


def speaker_key(namespace: str, speaker_id: str | int) -> str:
    """Namespaced speaker key, for example ``speaker_key("libri", 3081) == "libri:3081"``."""
    ns = namespace.strip().lower()
    sid = str(speaker_id).strip()
    if not ns or ":" in ns:
        raise ValueError(f"invalid namespace {namespace!r}")
    if not sid:
        raise ValueError("empty speaker id")
    if ns == VCTK:
        sid = sid.lower()
        if sid.isdigit():
            sid = f"p{sid}"
    return f"{ns}:{sid}"


def split_speaker_key(key: str) -> tuple[str, str]:
    """Split ``"ns:id"`` into ``("ns", "id")``."""
    ns, sep, sid = key.partition(":")
    if not sep or not ns or not sid:
        raise ValueError(f"not a namespaced speaker key: {key!r}")
    return ns, sid


def check_speaker_disjoint(
    train: Iterable[str], held_out: Mapping[str, Iterable[str]]
) -> None:
    """Raise :class:`SplitLeakError` if any training speaker appears in a held-out set.

    ``held_out`` maps a set name (``"libritts_r_test"``, ``"vb_test"``, ...) to its
    namespaced speaker keys. The message lists every overlap per set.
    """
    train_set = set(train)
    leaks: dict[str, list[str]] = {}
    for name, speakers in held_out.items():
        overlap = sorted(train_set.intersection(speakers))
        if overlap:
            leaks[name] = overlap
    if leaks:
        detail = "; ".join(
            f"{name}: {', '.join(v[:10])}{' ...' if len(v) > 10 else ''} ({len(v)})"
            for name, v in leaks.items()
        )
        raise SplitLeakError(f"training speakers overlap held-out sets: {detail}")


def save_speaker_list(
    path: str | Path,
    speakers: Iterable[str],
    *,
    name: str,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write a sorted, validated speaker list as JSON (``name``, ``count``, ``speakers``, ``meta``)."""
    keys = sorted({str(s) for s in speakers})
    for key in keys:
        split_speaker_key(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "count": len(keys), "speakers": keys, "meta": dict(meta or {})}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def load_speaker_list(path: str | Path) -> frozenset[str]:
    """Speakers written by :func:`save_speaker_list` (a bare JSON list is accepted too)."""
    data = json.loads(Path(path).read_text())
    items = data["speakers"] if isinstance(data, Mapping) else data
    keys = frozenset(str(s) for s in items)
    for key in keys:
        split_speaker_key(key)
    return keys
