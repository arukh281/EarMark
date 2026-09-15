"""Exporters for the corpora that do not come from Hugging Face parquet.

Each ``export_*`` function turns one downloaded archive into prepared int16 shards
(:mod:`.shards`). ``notebooks/kaggle_prep_vctk_musan_noise.py`` only downloads and calls
them.

* VCTK 0.92: mic1 FLAC members are read in place from the zip (no extraction).
  VoiceBank-DEMAND's test speakers (p232, p257) are excluded, silence is trimmed and
  each speaker is capped. VCTK has no chapter or session metadata, so blocks of 100
  prompt numbers stand in for sessions when pools are assigned.
* MUSAN music: one excerpt per track, split into ``train`` and ``heldout`` by artist.
* RIRS_NOISES (OpenSLR 28): simulated RIRs for training, real RIRs held out for
  evaluation only, and point-source and isotropic noises for training, read in place
  from the zip.
* DEMAND: channel 1 of each environment's 16 kHz zip, cut into segments; held-out
  environments go to a separate evaluation set.
* ESC-50 (evaluation only): the 5 s clips of permitted categories, as Earmark-Synth noise.

Every noise and music item passes :func:`.noise_filter.mentions_excluded` on its
metadata before it is written.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
import zlib
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from earmark import constants as C
from earmark.data.noise_filter import esc50_keep, mentions_excluded
from earmark.data.shards import (
    DEFAULT_MAX_SHARD_BYTES,
    ShardWriter,
    resample_to_contract,
    to_mono,
    trim_silence,
)
from earmark.data.splits import VB_TEST_SPEAKER_IDS, VCTK, speaker_key

__all__ = [
    "DEMAND_URL_TEMPLATE",
    "ESC50_URL",
    "KeepFn",
    "MUSAN_URL",
    "RIRS_NOISES_URL",
    "VCTK_BLOCK",
    "VCTK_ZIP_URL",
    "MusicTrack",
    "RirsMember",
    "VctkMember",
    "artist_split",
    "classify_rirs_member",
    "export_demand",
    "export_esc50",
    "export_musan_music",
    "export_rirs_noises",
    "export_vctk",
    "parse_esc50_meta",
    "parse_musan_music_annotations",
    "parse_rirs_noise_list",
    "parse_vctk_speaker_info",
    "stable_hash",
    "vctk_group",
    "vctk_members",
]

#: VCTK 0.92 zip (10.9 GB). The server answers HEAD with a small HTML page and ignores
#: Range, but a plain GET streams the zip, so download it whole (``curl -L -o``) and read
#: members in place (checked 2026-09-11).
VCTK_ZIP_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
RIRS_NOISES_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"
DEMAND_URL_TEMPLATE = "https://zenodo.org/records/1227121/files/{env}_16k.zip?download=1"
#: ESC-50 repository archive: evaluation only (CC BY-NC 3.0; the ESC-10 subset is CC BY).
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

#: VCTK prompt numbers per pseudo-session block.
VCTK_BLOCK = 100

#: ``keep(audio_16k, meta) -> bool``: a content check applied to each noise piece before
#: it is written (for example :class:`.noise_filter.ContentFilter`).
KeepFn = Callable[[np.ndarray, Mapping[str, Any]], bool]


def stable_hash(*parts: object) -> int:
    """A process-independent 32-bit hash (CRC-32 of the joined parts)."""
    return zlib.crc32("\x1f".join(str(p) for p in parts).encode("utf-8"))


def _rng(seed: int, *parts: object) -> np.random.Generator:
    return np.random.default_rng([int(seed), stable_hash(*parts)])


# --------------------------------------------------------------------------- VCTK

_VCTK_FLAC = re.compile(r"(?:^|/)wav48_silence_trimmed/([ps]\d+)/(\1_(\d+))_(mic[12])\.flac$")


@dataclass(frozen=True)
class VctkMember:
    """One VCTK FLAC member of the zip."""

    speaker: str
    utt: str
    number: int
    mic: str
    name: str


def vctk_members(
    names: Iterable[str], *, mic: str = "mic1", exclude: Iterable[str] = VB_TEST_SPEAKER_IDS
) -> list[VctkMember]:
    """FLAC members for one microphone, excluding ``exclude`` speakers, sorted."""
    blocked = {e.lower() for e in exclude}
    out = []
    for name in names:
        m = _VCTK_FLAC.search(name)
        if m and m.group(4) == mic and m.group(1).lower() not in blocked:
            out.append(VctkMember(m.group(1), m.group(2), int(m.group(3)), m.group(4), name))
    return sorted(out, key=lambda v: (v.speaker, v.number))


def vctk_group(number: int, block: int = VCTK_BLOCK) -> str:
    """Pseudo-session of a VCTK prompt number (``block00``, ``block01``, ...)."""
    return f"block{number // block:02d}"


def parse_vctk_speaker_info(text: str) -> dict[str, dict[str, str]]:
    """Parse ``speaker-info.txt`` into ``{"p225": {"age", "gender", "accent", "region"}}``."""
    info: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0][-1:].isdigit() or parts[0].upper() == "ID":
            continue
        sid = parts[0].lower()
        if sid.isdigit():
            sid = f"p{sid}"
        info[sid] = {
            "age": parts[1],
            "gender": parts[2],
            "accent": parts[3],
            "region": " ".join(parts[4:]),
        }
    return info


def export_vctk(
    zip_path: str | Path,
    out_dir: str | Path,
    *,
    prefix: str = "vctk",
    speakers: Iterable[str] | None = None,
    exclude: Iterable[str] = VB_TEST_SPEAKER_IDS,
    mic: str = "mic1",
    cap_seconds: float = 900.0,
    trim_db: float = C.VAD_THRESHOLD_DB,
    margin_s: float = 0.1,
    min_seconds: float = 0.5,
    seed: int = 0,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> dict[str, float]:
    """Read VCTK FLACs in place from the zip and write capped, trimmed 16 kHz shards.

    Each speaker's utterances are visited in a seeded random order until the speaker
    reaches ``cap_seconds``, so the cap samples every pseudo-session block. Pass
    ``speakers`` to export a subset (one call per worker). Call
    :func:`.shards.finalize_dataset` and :func:`.shards.add_pools` afterwards.
    """
    wanted = {s.lower() for s in speakers} if speakers is not None else None
    stats = {"speakers": 0.0, "rows": 0.0, "seconds": 0.0, "skipped_short": 0.0}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        info_name = next((n for n in names if n.endswith("speaker-info.txt")), None)
        info = parse_vctk_speaker_info(zf.read(info_name).decode("utf-8", "replace")) if info_name else {}
        texts = {n.rsplit("/", 1)[-1][:-4]: n for n in names if "/txt/" in f"/{n}" and n.endswith(".txt")}
        by_speaker: dict[str, list[VctkMember]] = defaultdict(list)
        for member in vctk_members(names, mic=mic, exclude=exclude):
            if wanted is None or member.speaker.lower() in wanted:
                by_speaker[member.speaker].append(member)
        with ShardWriter(out_dir, prefix=prefix, max_shard_bytes=max_shard_bytes) as writer:
            for spk in sorted(by_speaker):
                members = by_speaker[spk]
                order = _rng(seed, "vctk", spk).permutation(len(members))
                total = 0.0
                for k in order:
                    if total >= cap_seconds:
                        break
                    member = members[int(k)]
                    audio, sr = sf.read(io.BytesIO(zf.read(member.name)), dtype="float32")
                    y = resample_to_contract(to_mono(audio), sr)
                    start, end = trim_silence(y, threshold_db=trim_db, margin_s=margin_s)
                    if end - start < min_seconds * C.SAMPLE_RATE:
                        stats["skipped_short"] += 1
                        continue
                    y = y[start:end]
                    text_name = texts.get(member.utt)
                    meta = info.get(spk.lower(), {})
                    writer.add(
                        y,
                        C.SAMPLE_RATE,
                        utt_id=f"{VCTK}:{member.utt}_{mic}",
                        speaker=speaker_key(VCTK, spk),
                        group=vctk_group(member.number),
                        text=zf.read(text_name).decode("utf-8", "replace").strip() if text_name else None,
                        accent=meta.get("accent"),
                        gender=meta.get("gender"),
                        region=meta.get("region"),
                        source="vctk",
                        mic=mic,
                    )
                    total += len(y) / C.SAMPLE_RATE
                    stats["rows"] += 1
                stats["speakers"] += 1
                stats["seconds"] += total
    return stats


# --------------------------------------------------------------------------- MUSAN music


@dataclass(frozen=True)
class MusicTrack:
    """One MUSAN music track's annotation."""

    track: str
    genre: str
    vocals: str
    artist: str


def parse_musan_music_annotations(text: str) -> dict[str, MusicTrack]:
    """Parse a MUSAN music ``ANNOTATIONS`` file (``track genre vocals artist...``).

    The vocals flag (``Y``/``N``) is located by value, so either column order parses;
    the artist is the rest of the line.
    """
    tracks: dict[str, MusicTrack] = {}
    for line in text.splitlines():
        tok = line.split()
        if not tok:
            continue
        name = tok[0]
        if len(tok) >= 3 and tok[1].upper() in {"Y", "N"}:
            vocals, genre, artist = tok[1].upper(), tok[2], " ".join(tok[3:])
        elif len(tok) >= 3 and tok[2].upper() in {"Y", "N"}:
            genre, vocals, artist = tok[1], tok[2].upper(), " ".join(tok[3:])
        else:
            genre, vocals, artist = (tok[1] if len(tok) > 1 else ""), "", " ".join(tok[2:])
        tracks[name] = MusicTrack(name, genre, vocals, artist or name)
    return tracks


def artist_split(
    artists: Iterable[str], *, heldout_fraction: float = 0.1, seed: int = 0
) -> dict[str, str]:
    """Deterministic artist-disjoint split: ``{artist: "train" | "heldout"}``."""
    out = {}
    for artist in sorted(set(artists)):
        u = stable_hash(seed, "musan-artist", artist.lower()) / 2**32
        out[artist] = "heldout" if u < heldout_fraction else "train"
    return out


def export_musan_music(
    music_root: str | Path,
    out_train: str | Path,
    out_heldout: str | Path,
    *,
    excerpt_seconds: float = 80.0,
    heldout_fraction: float = 0.1,
    seed: int = 0,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> dict[str, float]:
    """Write one seeded excerpt per MUSAN music track, split by artist.

    ``music_root`` is the extracted ``musan/music`` directory (sub-directories with an
    ``ANNOTATIONS`` file and ``music-*.wav`` tracks).
    """
    root = Path(music_root)
    tracks: dict[str, tuple[Path, MusicTrack, str]] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        ann_path = sub / "ANNOTATIONS"
        ann = parse_musan_music_annotations(ann_path.read_text("utf-8", "replace")) if ann_path.exists() else {}
        for wav in sorted(sub.glob("*.wav")):
            meta = ann.get(wav.stem, MusicTrack(wav.stem, "", "", wav.stem))
            tracks[wav.stem] = (wav, meta, sub.name)
    splits = artist_split((t[1].artist for t in tracks.values()), heldout_fraction=heldout_fraction, seed=seed)
    stats = {"train": 0.0, "heldout": 0.0, "dropped": 0.0, "seconds": 0.0}
    with (
        ShardWriter(out_train, prefix="music", max_shard_bytes=max_shard_bytes) as w_train,
        ShardWriter(out_heldout, prefix="music", max_shard_bytes=max_shard_bytes) as w_held,
    ):
        for name in sorted(tracks):
            path, meta, subset = tracks[name]
            if mentions_excluded(name, meta.genre, meta.artist):
                stats["dropped"] += 1
                continue
            info = sf.info(str(path))
            frames = int(round(excerpt_seconds * info.samplerate))
            start = int(_rng(seed, "musan-excerpt", name).integers(0, max(1, info.frames - frames + 1)))
            audio, sr = sf.read(str(path), start=start, frames=frames, dtype="float32")
            split = splits[meta.artist]
            writer = w_held if split == "heldout" else w_train
            writer.add(
                to_mono(audio),
                sr,
                utt_id=f"musan:{name}",
                speaker=f"musan:{meta.artist}",
                group=name,
                split=split,
                genre=meta.genre,
                vocals=meta.vocals,
                artist=meta.artist,
                subset=subset,
                source="musan_music",
            )
            stats[split] += 1
            stats["seconds"] += len(audio) / sr
    return stats


# --------------------------------------------------------------------------- RIRS_NOISES


@dataclass(frozen=True)
class RirsMember:
    """One classified WAV member of the RIRS_NOISES zip."""

    name: str
    kind: str  # "sim_rir", "real_rir", "iso_noise" or "point_noise"
    room: str
    room_size: str


def classify_rirs_member(
    name: str,
    *,
    real_rirs: Collection[str] | None = None,
    iso_noises: Collection[str] | None = None,
) -> RirsMember | None:
    """Classify a RIRS_NOISES zip member; ``None`` for non-WAV entries.

    Under ``real_rirs_isotropic_noises`` the zip's own lists decide: pass the WAV
    basenames of its ``noise_list`` as ``iso_noises`` and of its ``rir_list`` as
    ``real_rirs``. Without the lists, a file whose name contains ``noise`` counts as a
    noise. Anything else there counts as a real RIR, so an unrecognised file is held out
    rather than trained on.
    """
    if not name.lower().endswith(".wav"):
        return None
    parts = name.split("/")
    base = parts[-1]
    stem = base[:-4]
    if "simulated_rirs" in parts:
        i = parts.index("simulated_rirs")
        if len(parts) < i + 4:
            return None
        size = parts[i + 1]
        return RirsMember(name, "sim_rir", f"{size}/{parts[i + 2]}", size)
    if "real_rirs_isotropic_noises" in parts:
        listed_rir = real_rirs is not None and base in real_rirs
        if iso_noises is not None:
            is_noise = base in iso_noises and not listed_rir
        else:
            is_noise = "noise" in stem.lower() and not listed_rir
        return RirsMember(name, "iso_noise" if is_noise else "real_rir", stem, "real")
    if "pointsource_noises" in parts:
        return RirsMember(name, "point_noise", "", "")
    return None


def parse_rirs_noise_list(text: str) -> dict[str, str]:
    """Map WAV basename -> its full ``noise_list`` line (the only metadata the zip has)."""
    out = {}
    for line in text.splitlines():
        wav = next((t for t in line.split() if t.lower().endswith(".wav")), None)
        if wav:
            out[wav.rsplit("/", 1)[-1]] = line.strip()
    return out


def _read_member(zf: zipfile.ZipFile, name: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(zf.read(name)), dtype="float32", always_2d=True)
    return resample_to_contract(audio[:, 0], sr), C.SAMPLE_RATE


def _segments(x: np.ndarray, seconds: float | None) -> list[np.ndarray]:
    if seconds is None:
        return [x]
    n = int(round(seconds * C.SAMPLE_RATE))
    if len(x) <= int(1.5 * n):
        return [x]
    return [x[i : i + n] for i in range(0, len(x) - n // 2, n)]


def _normalised_rir(h: np.ndarray, max_seconds: float) -> tuple[np.ndarray, int]:
    h = h[: int(round(max_seconds * C.SAMPLE_RATE))]
    peak = float(np.max(np.abs(h))) if h.size else 0.0
    if peak > 0:
        h = h * (0.9 / peak)
    return h.astype(np.float32), int(np.argmax(np.abs(h))) if h.size else 0


def export_rirs_noises(
    zip_path: str | Path,
    *,
    out_rir_sim: str | Path,
    out_noise_train: str | Path,
    out_rir_real: str | Path,
    rirs_per_room: int = 20,
    max_rir_seconds: float = 1.0,
    noise_segment_seconds: float | None = 10.0,
    seed: int = 0,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    keep: KeepFn | None = None,
) -> dict[str, float]:
    """Export RIRS_NOISES in place from its zip.

    * Simulated RIRs: up to ``rirs_per_room`` per room (seeded), cropped to
      ``max_rir_seconds`` and peak-normalised; ``kind="simulated"``.
    * Real RIRs (``real_rirs_isotropic_noises/rir_list`` plus anything unlisted there):
      all, same processing, ``kind="real"``, to ``out_rir_real`` (held out).
    * Point-source and isotropic noises: channel 0, cut into ``noise_segment_seconds``
      pieces, filtered on file name and ``noise_list`` metadata and then by ``keep`` (a
      content check such as :class:`.noise_filter.ContentFilter`) on each 16 kHz piece.
      The point-source names (``noise-free-sound-NNNN``) carry no class, so ``keep`` is
      the only filter that can act on them.
    """
    stats = {"sim_rir": 0.0, "real_rir": 0.0, "noise": 0.0, "dropped": 0.0, "dropped_content": 0.0}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        list_meta: dict[str, str] = {}
        real_list: dict[str, str] = {}
        iso_list: dict[str, str] = {}
        for n in names:
            leaf = n.rsplit("/", 1)[-1]
            if leaf == "noise_list":
                parsed = parse_rirs_noise_list(zf.read(n).decode("utf-8", "replace"))
                list_meta.update(parsed)
                if "real_rirs_isotropic_noises" in n:
                    iso_list.update(parsed)
            elif leaf == "rir_list" and "real_rirs_isotropic_noises" in n:
                real_list.update(parse_rirs_noise_list(zf.read(n).decode("utf-8", "replace")))
        members = [
            m
            for m in (
                classify_rirs_member(n, real_rirs=real_list or None, iso_noises=iso_list or None)
                for n in names
            )
            if m is not None
        ]
        by_room: dict[str, list[RirsMember]] = defaultdict(list)
        for m in members:
            if m.kind == "sim_rir":
                by_room[m.room].append(m)
        with (
            ShardWriter(out_rir_sim, prefix="rir", max_shard_bytes=max_shard_bytes) as w_sim,
            ShardWriter(out_rir_real, prefix="rir", max_shard_bytes=max_shard_bytes) as w_real,
            ShardWriter(out_noise_train, prefix="noise", max_shard_bytes=max_shard_bytes) as w_noise,
        ):
            for room in sorted(by_room):
                ms = sorted(by_room[room], key=lambda m: m.name)
                pick = _rng(seed, "rir-room", room).permutation(len(ms))[:rirs_per_room]
                for k in sorted(pick.tolist()):
                    m = ms[k]
                    h, sr = _read_member(zf, m.name)
                    h, peak = _normalised_rir(h, max_rir_seconds)
                    stem = m.name.rsplit("/", 1)[-1][:-4]
                    w_sim.add(
                        h, sr, utt_id=f"rirs:sim:{m.room}/{stem}", speaker=f"rirs:{m.room}",
                        group=m.room, kind="simulated", room=m.room, room_size=m.room_size,
                        peak_index=peak, source="rirs_noises",
                    )
                    stats["sim_rir"] += 1
            for m in sorted((m for m in members if m.kind != "sim_rir"), key=lambda m: m.name):
                base = m.name.rsplit("/", 1)[-1]
                if m.kind == "real_rir":
                    h, sr = _read_member(zf, m.name)
                    h, peak = _normalised_rir(h, max_rir_seconds)
                    w_real.add(
                        h, sr, utt_id=f"rirs:real:{m.room}", speaker=f"rirs:real:{m.room}",
                        group=m.room, kind="real", room=m.room, room_size="real",
                        peak_index=peak, source="rirs_noises",
                    )
                    stats["real_rir"] += 1
                    continue
                meta_line = list_meta.get(base, "")
                if mentions_excluded(m.name, meta_line):
                    stats["dropped"] += 1
                    continue
                x, sr = _read_member(zf, m.name)
                kind = "isotropic" if m.kind == "iso_noise" else "point_source"
                for j, seg in enumerate(_segments(x, noise_segment_seconds)):
                    if seg.size < C.WINDOW_LENGTH:
                        continue
                    utt_id = f"rirs:noise:{base[:-4]}:{j:03d}"
                    if keep is not None and not keep(seg, {"utt_id": utt_id, "kind": kind, "source": "rirs_noises"}):
                        stats["dropped_content"] += 1
                        continue
                    w_noise.add(
                        seg, sr, utt_id=utt_id, speaker=f"rirs:noise:{base[:-4]}",
                        group=base[:-4], kind=kind, source="rirs_noises", environment=None,
                        description=meta_line or None,
                    )
                    stats["noise"] += 1
    return stats


# --------------------------------------------------------------------------- DEMAND


def export_demand(
    zip_paths: Mapping[str, str | Path],
    *,
    out_train: str | Path,
    out_heldout: str | Path,
    held_out: Iterable[str],
    channel: str = "ch01",
    segment_seconds: float | None = 10.0,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    keep: KeepFn | None = None,
) -> dict[str, float]:
    """Write channel ``channel`` of each DEMAND environment, cut into segments.

    ``zip_paths`` maps an environment code (``"DKITCHEN"``) to its ``*_16k.zip``.
    Environments in ``held_out`` go to ``out_heldout`` (evaluation only). ``keep`` (a
    content check) is applied to every segment of both sets.
    """
    blocked = {h.upper() for h in held_out}
    stats = {"train_envs": 0.0, "heldout_envs": 0.0, "rows": 0.0, "dropped_content": 0.0}
    with (
        ShardWriter(out_train, prefix="demand", max_shard_bytes=max_shard_bytes) as w_train,
        ShardWriter(out_heldout, prefix="demand", max_shard_bytes=max_shard_bytes) as w_held,
    ):
        for env in sorted(zip_paths):
            code = env.upper()
            with zipfile.ZipFile(zip_paths[env]) as zf:
                member = next(
                    (n for n in zf.namelist() if n.rsplit("/", 1)[-1].lower() == f"{channel}.wav"),
                    None,
                )
                if member is None:
                    raise FileNotFoundError(f"{channel}.wav not found in {zip_paths[env]}")
                x, sr = _read_member(zf, member)
            held = code in blocked
            writer = w_held if held else w_train
            for j, seg in enumerate(_segments(x, segment_seconds)):
                utt_id = f"demand:{code}:{j:03d}"
                if keep is not None and not keep(seg, {"utt_id": utt_id, "environment": code, "source": "demand"}):
                    stats["dropped_content"] += 1
                    continue
                writer.add(
                    seg, sr, utt_id=utt_id, speaker=f"demand:{code}", group=code,
                    environment=code, source="demand", kind="ambient",
                    split="heldout" if held else "train",
                )
                stats["rows"] += 1
            stats["heldout_envs" if held else "train_envs"] += 1
    return stats


# --------------------------------------------------------------------------- ESC-50


def parse_esc50_meta(csv_text: str) -> list[dict[str, Any]]:
    """Rows of ESC-50's ``meta/esc50.csv`` (filename, fold, target, category, esc10, ...)."""
    rows = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        row = {k: v for k, v in row.items() if k is not None}
        row["esc10"] = str(row.get("esc10", "")).strip().lower() == "true"
        for key in ("fold", "target"):
            if key in row and str(row[key]).isdigit():
                row[key] = int(row[key])
        rows.append(row)
    return rows


def export_esc50(
    root: str | Path,
    out_dir: str | Path,
    *,
    prefix: str = "esc50",
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> dict[str, float]:
    """Write the ESC-50 clips that may serve as held-out test noise (evaluation only).

    ``root`` holds the extracted repository (``meta/esc50.csv`` and ``audio/*.wav``, 5 s at
    44.1 kHz), possibly one folder down. Excluded categories are dropped
    (:func:`.noise_filter.esc50_keep`); the rest are resampled to 16 kHz with
    ``source="esc50"`` and ``split="test"``, which the training-pool check rejects.
    """
    base = Path(root)
    meta_path = next(iter(sorted(base.glob("**/meta/esc50.csv"))), None)
    if meta_path is None:
        raise FileNotFoundError(f"meta/esc50.csv not found under {base}")
    audio_dir = meta_path.parent.parent / "audio"
    stats = {"rows": 0.0, "dropped": 0.0}
    with ShardWriter(out_dir, prefix=prefix, max_shard_bytes=max_shard_bytes) as writer:
        for row in sorted(parse_esc50_meta(meta_path.read_text("utf-8")), key=lambda r: str(r["filename"])):
            name, category = str(row["filename"]), str(row["category"])
            if not esc50_keep(category, name):
                stats["dropped"] += 1
                continue
            audio, sr = sf.read(str(audio_dir / name), dtype="float32", always_2d=True)
            writer.add(
                audio[:, 0], sr, utt_id=f"esc50:{name[:-4]}", speaker=f"esc50:{category}", group=category,
                source="esc50", split="test", category=category, fold=row.get("fold"),
                esc10=bool(row.get("esc10")), license="CC BY" if row.get("esc10") else "CC BY-NC 3.0",
            )
            stats["rows"] += 1
    return stats
