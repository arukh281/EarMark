"""LibriCSS: session layout, timings, enrolment lists and frame-level region labels.

LibriCSS (``for_release.zip``, 6,407,297,637 bytes, Google Drive id
``1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l`` as used by the official ``dataprep.sh``) holds
LibriSpeech test-clean utterances replayed over loudspeakers in a meeting room::

    for_release/<condition>/overlap_ratio_<r>_sil<a>_<b>_session<k>_actual<o>/
        record/raw_recording.wav        7-channel 16 kHz recording (channel 0 is used)
        clean/each_spk.wav, mix.wav     source signals
        transcription/meeting_info.txt  header line, then tab-separated
                                        start_time, end_time, speaker, utterance_id, text

Conditions are ``0L``, ``0S``, ``OV10`` to ``OV40``; sessions 0-9, each made of six
mini-sessions (one per condition). Session 0 is dev and sessions 1-9 are test.

Enrolment lists: ``libricss_speaker_info.jsonl`` holds one JSON object per mini-session
with ``dataid`` (the mini-session directory name) and ``used_uttid``/``unused_uttid``
(utterance ids per speaker). When it is not available, :func:`derive_unused_enrolment`
rebuilds the unused lists from LibriSpeech test-clean.

Suite R treats each speaker of each mini-session as the target in turn, with every other
speaker as an interferer (:func:`target_cases`, :func:`frame_labels`).
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from earmark import constants as C
from earmark.data.labels import apply_hangover_np, intervals_to_frames
from earmark.data.shards import resample_to_contract
from earmark.data.splits import LIBRI, speaker_key

__all__ = [
    "CONDITIONS",
    "DEV_SESSIONS",
    "LIBRICSS_GDRIVE_ID",
    "LIBRICSS_SPEAKER_INFO_URL",
    "LIBRICSS_ZIP_BYTES",
    "LIBRISPEECH_TEST_CLEAN_BYTES",
    "LIBRISPEECH_TEST_CLEAN_URL",
    "REGION_INTERFERER",
    "REGION_OVERLAP",
    "REGION_SILENCE",
    "REGION_TARGET",
    "SESSIONS_FILE",
    "TEST_SESSIONS",
    "UTTERANCES_FILE",
    "MiniSession",
    "Utterance",
    "ZipMiniSession",
    "derive_unused_enrolment",
    "export_channel0",
    "extract_librispeech_utts",
    "find_mini_sessions",
    "frame_labels",
    "librispeech_flac_path",
    "load_exported_sessions",
    "load_mini_session",
    "needed_enrolment_utts",
    "parse_meeting_info",
    "parse_session_name",
    "parse_speaker_info_jsonl",
    "split_for_session",
    "suite_r_cases",
    "target_cases",
    "zip_mini_sessions",
]

LIBRICSS_GDRIVE_ID = "1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l"
LIBRICSS_ZIP_BYTES = 6_407_297_637
#: Per-mini-session enrolment lists in the LibriCSS repository (one JSON object per line,
#: ``{"dataid": <mini-session>, "unused_uttid": {speaker: [ids]}}``; checked 2026-09-11).
LIBRICSS_SPEAKER_INFO_URL = (
    "https://raw.githubusercontent.com/chenzhuo1011/libri_css/master/"
    "speaker_enrollment/libricss_speaker_info.jsonl"
)
LIBRISPEECH_TEST_CLEAN_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_TEST_CLEAN_BYTES = 346_663_984
CONDITIONS: tuple[str, ...] = ("0L", "0S", "OV10", "OV20", "OV30", "OV40")
DEV_SESSIONS: frozenset[int] = frozenset({0})
TEST_SESSIONS: frozenset[int] = frozenset(range(1, 10))

REGION_SILENCE = 0
REGION_TARGET = 1
REGION_INTERFERER = 2
REGION_OVERLAP = 3

_SESSION = re.compile(r"session(\d+)")
_ACTUAL = re.compile(r"actual(\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class Utterance:
    """One replayed LibriSpeech utterance on the meeting timeline."""

    start_s: float
    end_s: float
    speaker: str  # namespaced, e.g. "libri:1089"
    utt_id: str  # LibriSpeech id, e.g. "1089-134686-0000"
    text: str


@dataclass(frozen=True)
class MiniSession:
    """One 10-minute LibriCSS mini-session."""

    name: str
    condition: str
    session: int
    split: str
    utterances: tuple[Utterance, ...]
    actual_overlap: float | None = None
    path: Path | None = None
    audio: Path | None = None

    @property
    def speakers(self) -> tuple[str, ...]:
        """Speakers in order of first appearance."""
        seen: dict[str, None] = {}
        for u in sorted(self.utterances, key=lambda u: u.start_s):
            seen.setdefault(u.speaker, None)
        return tuple(seen)

    @property
    def duration_s(self) -> float:
        return max((u.end_s for u in self.utterances), default=0.0)

    @property
    def recording(self) -> Path | None:
        """The exported single-channel FLAC if set, else the release's 7-channel WAV."""
        if self.audio is not None:
            return self.audio
        return None if self.path is None else self.path / "record" / "raw_recording.wav"


def parse_session_name(name: str) -> tuple[int, float | None]:
    """Session index and actual overlap ratio from a mini-session directory name."""
    m = _SESSION.search(name)
    if not m:
        raise ValueError(f"no session index in {name!r}")
    a = _ACTUAL.search(name)
    return int(m.group(1)), (float(a.group(1)) if a else None)


def split_for_session(session: int) -> str:
    """``"dev"`` for session 0, ``"test"`` for sessions 1-9."""
    if session in DEV_SESSIONS:
        return "dev"
    if session in TEST_SESSIONS:
        return "test"
    raise ValueError(f"LibriCSS has sessions 0-9, not {session}")


def parse_meeting_info(text: str) -> list[Utterance]:
    """Parse ``meeting_info.txt``: a header, then start, end, speaker, utterance id, text."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 4:
            parts = line.split(None, 4)
        try:
            start, end = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue  # header
        text_field = parts[4].strip() if len(parts) > 4 else ""
        out.append(Utterance(start, end, speaker_key(LIBRI, parts[2]), parts[3].strip(), text_field))
    return sorted(out, key=lambda u: (u.start_s, u.utt_id))


def load_mini_session(path: str | Path, condition: str | None = None) -> MiniSession:
    """Load one mini-session directory (the one holding ``transcription/``)."""
    path = Path(path)
    session, actual = parse_session_name(path.name)
    info = path / "transcription" / "meeting_info.txt"
    return MiniSession(
        name=path.name,
        condition=condition or path.parent.name,
        session=session,
        split=split_for_session(session),
        utterances=tuple(parse_meeting_info(info.read_text("utf-8", "replace"))),
        actual_overlap=actual,
        path=path,
    )


def find_mini_sessions(root: str | Path) -> list[MiniSession]:
    """Every mini-session under ``for_release`` (sorted by session, then condition)."""
    root = Path(root)
    found = []
    for info in root.glob("*/overlap*/transcription/meeting_info.txt"):
        session_dir = info.parent.parent
        found.append(load_mini_session(session_dir, session_dir.parent.name))
    return sorted(found, key=lambda s: (s.session, CONDITIONS.index(s.condition) if s.condition in CONDITIONS else 99, s.name))


def _ids_by_speaker(value: Any) -> dict[str, list[str]]:
    """Normalise a used/unused field: dict speaker -> ids, or a (nested) list of ids."""
    out: dict[str, list[str]] = defaultdict(list)
    if isinstance(value, Mapping):
        for spk, ids in value.items():
            items = [ids] if isinstance(ids, str) else list(ids)
            out[speaker_key(LIBRI, spk)].extend(str(i) for i in items)
        return out
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            out[speaker_key(LIBRI, item.split("-")[0])].append(item)
        elif isinstance(item, Iterable):
            stack.extend(reversed(list(item)))
    return out


def parse_speaker_info_jsonl(text: str, *, field: str = "unused_uttid") -> dict[str, dict[str, tuple[str, ...]]]:
    """``{mini-session name: {speaker: utterance ids}}`` from ``libricss_speaker_info.jsonl``."""
    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        name = obj.get("dataid") or obj.get("session") or obj.get("name")
        if name is None or field not in obj:
            raise ValueError(f"speaker info line lacks 'dataid' or {field!r}: keys {sorted(obj)}")
        out[str(name)] = {k: tuple(sorted(set(v))) for k, v in _ids_by_speaker(obj[field]).items()}
    return out


def derive_unused_enrolment(
    sessions: Sequence[MiniSession], librispeech_utts: Iterable[str]
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Unused enrolment utterances per mini-session and speaker, from LibriSpeech ids.

    Prefers utterances that no LibriCSS mini-session replays; if a speaker has none
    left, falls back to utterances not replayed in that mini-session.
    """
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for utt in librispeech_utts:
        by_speaker[speaker_key(LIBRI, utt.split("-")[0])].append(utt)
    used_anywhere = {u.utt_id for s in sessions for u in s.utterances}
    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for s in sessions:
        used_here = {u.utt_id for u in s.utterances}
        entry = {}
        for spk in s.speakers:
            cands = sorted(u for u in by_speaker.get(spk, []) if u not in used_anywhere)
            if not cands:
                cands = sorted(u for u in by_speaker.get(spk, []) if u not in used_here)
            entry[spk] = tuple(cands)
        out[s.name] = entry
    return out


def librispeech_flac_path(root: str | Path, utt_id: str) -> Path:
    """``<root>/<speaker>/<chapter>/<utt_id>.flac`` for a LibriSpeech id."""
    speaker, chapter, _ = utt_id.split("-")
    return Path(root) / speaker / chapter / f"{utt_id}.flac"


def target_cases(session: MiniSession) -> tuple[str, ...]:
    """The speakers that take a turn as the target (all of them)."""
    return session.speakers


def frame_labels(
    session: MiniSession,
    target: str,
    n_frames: int,
    *,
    offset_s: float = 0.0,
    hangover_frames: int = C.VAD_HANGOVER_FRAMES,
    sample_rate: int = C.SAMPLE_RATE,
) -> dict[str, np.ndarray]:
    """Contract-frame labels for one target speaker from the released timings.

    Returns ``target`` and ``interferer`` activity (bool ``[F]``, with the contract
    hangover) and ``region`` (int8 ``[F]``: 0 silence, 1 target only, 2 interferer only,
    3 overlap), computed from the raw timings before hangover.
    """
    if target not in session.speakers:
        raise KeyError(f"{target} does not speak in {session.name}")
    tgt = [(u.start_s, u.end_s) for u in session.utterances if u.speaker == target]
    oth = [(u.start_s, u.end_s) for u in session.utterances if u.speaker != target]
    t = intervals_to_frames(tgt, n_frames, sample_rate=sample_rate, offset_s=offset_s)
    i = intervals_to_frames(oth, n_frames, sample_rate=sample_rate, offset_s=offset_s)
    region = np.zeros(n_frames, dtype=np.int8)
    region[t & ~i] = REGION_TARGET
    region[i & ~t] = REGION_INTERFERER
    region[t & i] = REGION_OVERLAP
    return {
        "target": apply_hangover_np(t, hangover_frames),
        "interferer": apply_hangover_np(i, hangover_frames),
        "region": region,
    }


def needed_enrolment_utts(
    speaker_info: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    sessions: Iterable[MiniSession] | None = None,
    per_speaker: int | None = None,
) -> dict[str, tuple[str, ...]]:
    """Each speaker's enrolment utterance ids, pooled over mini-sessions (sorted).

    ``speaker_info`` is :func:`parse_speaker_info_jsonl` output. ``sessions`` restricts it
    to those mini-sessions and ``per_speaker`` keeps each speaker's first N ids.
    """
    wanted = None if sessions is None else {s.name for s in sessions}
    pooled: dict[str, set[str]] = defaultdict(set)
    for name, entry in speaker_info.items():
        if wanted is not None and name not in wanted:
            continue
        for spk, ids in entry.items():
            pooled[spk].update(str(i) for i in ids)
    out: dict[str, tuple[str, ...]] = {}
    for spk in sorted(pooled):
        ids = sorted(pooled[spk])
        out[spk] = tuple(ids[:per_speaker] if per_speaker is not None else ids)
    return out


def suite_r_cases(
    sessions: Iterable[MiniSession], enrolment: Mapping[str, Mapping[str, Sequence[str]]]
) -> list[dict[str, Any]]:
    """One Suite R case per (mini-session, target speaker), with its enrolment utterances.

    ``enrolment`` maps a mini-session name to ``{speaker: utterance ids}``. A speaker with
    no listed utterances cannot be enrolled and is skipped.
    """
    cases: list[dict[str, Any]] = []
    for s in sessions:
        entry = enrolment.get(s.name, {})
        for spk in target_cases(s):
            ids = [str(i) for i in entry.get(spk, ())]
            if not ids:
                continue
            cases.append(
                {
                    "session": s.name,
                    "condition": s.condition,
                    "split": s.split,
                    "target": spk,
                    "enrol_utts": ids,
                    "n_target_utts": sum(u.speaker == spk for u in s.utterances),
                    "n_interferer_utts": sum(u.speaker != spk for u in s.utterances),
                }
            )
    return cases


# --------------------------------------------------------------------------- Colab export

SESSIONS_FILE = "sessions.parquet"
UTTERANCES_FILE = "utterances.parquet"
_ZIP_INFO = re.compile(r"(?:^|/)([^/]+)/(overlap_ratio_[^/]+)/transcription/meeting_info\.txt$")


@dataclass(frozen=True)
class ZipMiniSession:
    """A mini-session found inside ``for_release.zip``."""

    condition: str
    name: str
    info_member: str
    recording_member: str


def _condition_rank(condition: str) -> int:
    return CONDITIONS.index(condition) if condition in CONDITIONS else len(CONDITIONS)


def zip_mini_sessions(names: Iterable[str]) -> list[ZipMiniSession]:
    """Mini-sessions in a ``for_release.zip`` listing that have timings and a recording."""
    listing = list(names)
    present = set(listing)
    found = []
    for name in listing:
        m = _ZIP_INFO.search(name)
        if not m:
            continue
        prefix = name[: -len("transcription/meeting_info.txt")]
        recording = f"{prefix}record/raw_recording.wav"
        if recording in present:
            found.append(ZipMiniSession(m.group(1), m.group(2), name, recording))
    return sorted(
        found, key=lambda z: (parse_session_name(z.name)[0], _condition_rank(z.condition), z.name)
    )


def export_channel0(
    zip_path: str | Path,
    out_dir: str | Path,
    *,
    channel: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[MiniSession]:
    """Write one channel of every LibriCSS recording as 16 kHz FLAC, plus manifests.

    Members are read in place from the zip (nothing is extracted). Writes
    ``audio/<condition>/<name>.flac``, ``sessions.parquet`` (one row per mini-session)
    and ``utterances.parquet`` (the released timings) under ``out_dir`` and returns the
    sessions with ``audio`` set.
    """
    out = Path(out_dir)
    sessions: list[MiniSession] = []
    lengths: list[int] = []
    with zipfile.ZipFile(zip_path) as zf:
        found = zip_mini_sessions(zf.namelist())
        if not found:
            raise FileNotFoundError(f"no LibriCSS mini-sessions in {zip_path}")
        for k, z in enumerate(found):
            utts = parse_meeting_info(zf.read(z.info_member).decode("utf-8", "replace"))
            with zf.open(z.recording_member) as fh:
                audio, sr = sf.read(fh, dtype="float32", always_2d=True)
            x = resample_to_contract(audio[:, channel], sr)
            rel = Path("audio") / z.condition / f"{z.name}.flac"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out / rel), x, C.SAMPLE_RATE, format="FLAC", subtype="PCM_16")
            session, actual = parse_session_name(z.name)
            sessions.append(
                MiniSession(
                    name=z.name,
                    condition=z.condition,
                    session=session,
                    split=split_for_session(session),
                    utterances=tuple(utts),
                    actual_overlap=actual,
                    path=None,
                    audio=out / rel,
                )
            )
            lengths.append(int(x.size))
            if progress is not None:
                progress(k + 1, len(found), z.name)
    _write_session_tables(out, sessions, lengths)
    return sessions


def _write_session_tables(out: Path, sessions: Sequence[MiniSession], lengths: Sequence[int]) -> None:
    table = pa.table(
        {
            "name": [s.name for s in sessions],
            "condition": [s.condition for s in sessions],
            "session": pa.array([s.session for s in sessions], pa.int32()),
            "split": [s.split for s in sessions],
            "actual_overlap": pa.array([s.actual_overlap for s in sessions], pa.float64()),
            "audio": [str(s.audio.relative_to(out)) if s.audio is not None else None for s in sessions],
            "num_samples": pa.array(list(lengths), pa.int64()),
            "speakers": [list(s.speakers) for s in sessions],
            "n_utterances": pa.array([len(s.utterances) for s in sessions], pa.int32()),
        }
    )
    pq.write_table(table, out / SESSIONS_FILE)
    utt: dict[str, list[Any]] = {k: [] for k in ("name", "start_s", "end_s", "speaker", "utt_id", "text")}
    for s in sessions:
        for u in s.utterances:
            utt["name"].append(s.name)
            utt["start_s"].append(u.start_s)
            utt["end_s"].append(u.end_s)
            utt["speaker"].append(u.speaker)
            utt["utt_id"].append(u.utt_id)
            utt["text"].append(u.text)
    pq.write_table(
        pa.table(
            {
                "name": pa.array(utt["name"], pa.string()),
                "start_s": pa.array(utt["start_s"], pa.float64()),
                "end_s": pa.array(utt["end_s"], pa.float64()),
                "speaker": pa.array(utt["speaker"], pa.string()),
                "utt_id": pa.array(utt["utt_id"], pa.string()),
                "text": pa.array(utt["text"], pa.string()),
            }
        ),
        out / UTTERANCES_FILE,
    )


def load_exported_sessions(root: str | Path) -> list[MiniSession]:
    """Sessions written by :func:`export_channel0` (``audio`` points at the FLACs)."""
    root = Path(root)
    utterances: dict[str, list[Utterance]] = defaultdict(list)
    for u in pq.read_table(root / UTTERANCES_FILE).to_pylist():
        utterances[u["name"]].append(
            Utterance(float(u["start_s"]), float(u["end_s"]), u["speaker"], u["utt_id"], u["text"] or "")
        )
    sessions = []
    for s in pq.read_table(root / SESSIONS_FILE).to_pylist():
        utts = tuple(sorted(utterances[s["name"]], key=lambda u: (u.start_s, u.utt_id)))
        sessions.append(
            MiniSession(
                name=s["name"],
                condition=s["condition"],
                session=int(s["session"]),
                split=s["split"],
                utterances=utts,
                actual_overlap=s["actual_overlap"],
                path=None,
                audio=root / s["audio"] if s["audio"] else None,
            )
        )
    return sessions


def extract_librispeech_utts(
    tar_path: str | Path, wanted: Iterable[str], out_dir: str | Path
) -> dict[str, Path]:
    """Stream a LibriSpeech tarball and copy only the ``wanted`` FLACs.

    Files land at ``<out_dir>/<speaker>/<chapter>/<utt_id>.flac``
    (:func:`librispeech_flac_path`); the tarball is read once, front to back, and never
    extracted. Returns the ids found.
    """
    want = {str(w) for w in wanted}
    found: dict[str, Path] = {}
    with tarfile.open(tar_path, mode="r|*") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".flac"):
                continue
            utt_id = member.name.rsplit("/", 1)[-1][: -len(".flac")]
            if utt_id not in want:
                continue
            dst = librispeech_flac_path(out_dir, utt_id)
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, dst.open("wb") as fh:
                shutil.copyfileobj(src, fh)
            found[utt_id] = dst
    return found
