"""16 kHz int16 audio shards with a parquet manifest.

A prepared dataset is one directory::

    <root>/
      manifest.parquet        one row per utterance (or noise clip, RIR, music excerpt)
      dataset_info.json       totals, sample rate, contract hash
      <prefix>-00000.i16      raw little-endian int16 samples, utterances back to back
      <prefix>-00001.i16
      ...

Shards are flat files, so training memory-maps them (``numpy.memmap``) and slices an
utterance with no decoding. Audio is resampled to the contract rate with soxr and stored
as ``round(x * 32768)`` clipped to int16; readers divide by 32768.

Manifest columns always present:

========================  ==========================================================
``utt_id``                unique id, namespaced (``libri:3081_166546_000101_000001``)
``speaker``               namespaced speaker (or source) key, see :mod:`.splits`
``group``                 chapter, session or other recording unit within the speaker
``shard``                 shard file name
``offset``                first sample within the shard
``num_samples``           length in samples at 16 kHz
``peak_energy``           peak contract-frame energy of the stored audio (VAD reference)
``clipped``               samples clipped when converting to int16
========================  ==========================================================

Any other keyword passed to :meth:`ShardWriter.add` becomes a column (``text``, ``pool``,
``split``, ``voice_id``, ``environment``, ``kind``, ``room``...).

Enrolment and target pools: :func:`assign_pools` splits each speaker's audio by group
(chapter or session) into ``enrol``, ``target`` and ``spare`` utterances, and
:func:`check_pools_disjoint` enforces that no group feeds both enrolment and targets.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soxr

from earmark import constants as C
from earmark.data.labels import frame_energy_np

__all__ = [
    "DEFAULT_MAX_SHARD_BYTES",
    "INFO_NAME",
    "INT16_SCALE",
    "MANIFEST_NAME",
    "POOLS",
    "POOL_ENROL",
    "POOL_SPARE",
    "POOL_TARGET",
    "REQUIRED_COLUMNS",
    "SHARD_SUFFIX",
    "PoolLeakError",
    "ShardWriter",
    "ShardedCorpus",
    "add_pools",
    "assign_pools",
    "check_pools_disjoint",
    "dataset_info",
    "finalize_dataset",
    "from_int16",
    "peak_frame_energy",
    "resample_to_contract",
    "to_float32",
    "to_int16",
    "to_mono",
    "trim_silence",
]

SHARD_SUFFIX = ".i16"
MANIFEST_NAME = "manifest.parquet"
INFO_NAME = "dataset_info.json"
INT16_SCALE = 32768.0
DEFAULT_MAX_SHARD_BYTES = 512 * 2**20
REQUIRED_COLUMNS: tuple[str, ...] = (
    "utt_id",
    "speaker",
    "group",
    "shard",
    "offset",
    "num_samples",
    "peak_energy",
    "clipped",
)

POOL_ENROL = "enrol"
POOL_TARGET = "target"
POOL_SPARE = "spare"
POOLS: tuple[str, ...] = (POOL_ENROL, POOL_TARGET, POOL_SPARE)


class PoolLeakError(ValueError):
    """Raised when one chapter or session feeds both the enrolment and the target pool."""


# --------------------------------------------------------------------------- sample helpers


def to_mono(audio: np.ndarray, channel: int | None = None) -> np.ndarray:
    """Return mono audio. 2-D input is ``[frames, channels]`` (soundfile's layout).

    ``channel`` picks one channel; ``None`` averages them.
    """
    a = np.asarray(audio)
    if a.ndim == 1:
        return a
    if a.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {a.shape}")
    if channel is not None:
        return a[:, channel]
    return to_float32(a).mean(axis=1, dtype=np.float64).astype(np.float32)


def to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert integer PCM (scaled to [-1, 1)) or float audio to float32."""
    a = np.asarray(audio)
    if np.issubdtype(a.dtype, np.integer):
        scale = float(2 ** (8 * a.dtype.itemsize - 1))
        return (a.astype(np.float64) / scale).astype(np.float32)
    return a.astype(np.float32, copy=False)


def resample_to_contract(
    audio: np.ndarray, sample_rate: int, *, quality: str = "HQ"
) -> np.ndarray:
    """Resample mono audio to ``SAMPLE_RATE`` with soxr; returns float32."""
    x = to_float32(audio)
    if x.ndim != 1:
        raise ValueError("resample_to_contract expects mono audio; call to_mono first")
    if int(sample_rate) == C.SAMPLE_RATE:
        return x
    y = soxr.resample(x, int(sample_rate), C.SAMPLE_RATE, quality=quality)
    return np.asarray(y, dtype=np.float32)


def to_int16(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Quantise float audio to little-endian int16 as ``round(x * 32768)``.

    Returns the samples and how many were clipped to the int16 range.
    """
    scaled = np.round(np.asarray(x, dtype=np.float64) * INT16_SCALE)
    clipped = int(np.count_nonzero((scaled > 32767.0) | (scaled < -32768.0)))
    return np.clip(scaled, -32768.0, 32767.0).astype("<i2"), clipped


def from_int16(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_int16`: int16 samples to float32 in [-1, 1)."""
    return np.asarray(x).astype(np.float32) / np.float32(INT16_SCALE)


def peak_frame_energy(x: np.ndarray) -> float:
    """Peak contract-frame energy of ``x``; shorter-than-a-window clips use plain energy."""
    energy = frame_energy_np(x)
    if energy.size:
        return float(energy.max())
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def trim_silence(
    x: np.ndarray, *, threshold_db: float = C.VAD_THRESHOLD_DB, margin_s: float = 0.1
) -> tuple[int, int]:
    """Sample range ``[start, end)`` that keeps frames above ``threshold_db`` re the peak.

    A margin of ``margin_s`` is kept on both sides. An all-silent clip returns ``(0, 0)``.
    """
    energy = frame_energy_np(x)
    n = len(x)
    if energy.size == 0:
        return 0, n
    peak = energy.max()
    if peak <= 0:
        return 0, 0
    active = energy > peak * 10.0 ** (threshold_db / 10.0)
    first = int(np.argmax(active))
    last = int(len(active) - 1 - np.argmax(active[::-1]))
    margin = int(round(margin_s * C.SAMPLE_RATE))
    start = max(0, first * C.HOP_LENGTH - margin)
    end = min(n, last * C.HOP_LENGTH + C.WINDOW_LENGTH + margin)
    return start, end


# --------------------------------------------------------------------------- writer


class ShardWriter:
    """Append utterances to int16 shards and record them in a partial manifest.

    Several writers (for example one per worker process) may share ``out_dir`` as long as
    their ``prefix`` differs; shard files are created exclusively, so a clash fails loudly
    instead of overwriting. :meth:`close` writes ``manifest-<prefix>.parquet``;
    :func:`finalize_dataset` merges all partial manifests into ``manifest.parquet``.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        prefix: str = "shard",
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
        resample_quality: str = "HQ",
    ) -> None:
        if not prefix or any(c in prefix for c in "/\\") or prefix.startswith("."):
            raise ValueError(f"invalid shard prefix {prefix!r}")
        if max_shard_bytes < 2:
            raise ValueError("max_shard_bytes must hold at least one sample")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_shard_bytes = int(max_shard_bytes)
        self.resample_quality = resample_quality
        self._required: dict[str, list[Any]] = {c: [] for c in REQUIRED_COLUMNS}
        self._meta: dict[str, list[Any]] = {}
        self._fh: Any = None
        self._shard_index = -1
        self._shard_name = ""
        self._shard_samples = 0
        self._closed = False
        self.num_rows = 0
        self.total_samples = 0

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _rotate(self) -> None:
        if self._fh is not None:
            self._fh.close()
        self._shard_index += 1
        self._shard_name = f"{self.prefix}-{self._shard_index:05d}{SHARD_SUFFIX}"
        self._fh = open(self.out_dir / self._shard_name, "xb")  # noqa: SIM115 - long-lived
        self._shard_samples = 0

    def add(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        utt_id: str,
        speaker: str,
        group: str,
        **meta: Any,
    ) -> int:
        """Resample, quantise and append one utterance; returns its length in samples.

        ``audio`` is mono (float in [-1, 1] or integer PCM) at ``sample_rate``. Extra keyword
        arguments become manifest columns; rows that do not set a column get null.
        """
        if self._closed:
            raise RuntimeError("ShardWriter is closed")
        for name in meta:
            if name in REQUIRED_COLUMNS:
                raise ValueError(f"{name!r} is a reserved manifest column")
        x = resample_to_contract(to_mono(audio), sample_rate, quality=self.resample_quality)
        pcm, clipped = to_int16(x)
        n = int(pcm.size)
        if n == 0:
            raise ValueError(f"utterance {utt_id!r} is empty")
        if self._fh is None or (
            self._shard_samples > 0 and 2 * (self._shard_samples + n) > self.max_shard_bytes
        ):
            self._rotate()
        self._fh.write(pcm.tobytes())
        row = {
            "utt_id": str(utt_id),
            "speaker": str(speaker),
            "group": str(group),
            "shard": self._shard_name,
            "offset": self._shard_samples,
            "num_samples": n,
            "peak_energy": peak_frame_energy(pcm.astype(np.float32) / np.float32(INT16_SCALE)),
            "clipped": clipped,
        }
        for key, value in row.items():
            self._required[key].append(value)
        for key in meta:
            if key not in self._meta:
                self._meta[key] = [None] * self.num_rows
        for key, column in self._meta.items():
            column.append(meta.get(key))
        self._shard_samples += n
        self.num_rows += 1
        self.total_samples += n
        return n

    def close(self) -> Path | None:
        """Flush the last shard and write the partial manifest (``None`` if nothing added)."""
        if self._closed:
            return self.partial_manifest_path if self.num_rows else None
        self._closed = True
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self.num_rows == 0:
            return None
        columns: dict[str, Any] = {
            "utt_id": pa.array(self._required["utt_id"], pa.string()),
            "speaker": pa.array(self._required["speaker"], pa.string()),
            "group": pa.array(self._required["group"], pa.string()),
            "shard": pa.array(self._required["shard"], pa.string()),
            "offset": pa.array(self._required["offset"], pa.int64()),
            "num_samples": pa.array(self._required["num_samples"], pa.int64()),
            "peak_energy": pa.array(self._required["peak_energy"], pa.float64()),
            "clipped": pa.array(self._required["clipped"], pa.int32()),
        }
        for key, values in self._meta.items():
            columns[key] = pa.array(values)
        path = self.partial_manifest_path
        pq.write_table(pa.table(columns), path)
        return path

    @property
    def partial_manifest_path(self) -> Path:
        """Where :meth:`close` writes this writer's partial manifest."""
        return self.out_dir / f"manifest-{self.prefix}.parquet"


def _concat_tables(tables: Sequence[pa.Table]) -> pa.Table:
    try:
        return pa.concat_tables(tables, promote_options="default")
    except TypeError:  # pyarrow < 14
        return pa.concat_tables(tables, promote=True)


def finalize_dataset(
    out_dir: str | Path, *, name: str, info: Mapping[str, Any] | None = None
) -> Path:
    """Merge partial manifests into ``manifest.parquet`` and write ``dataset_info.json``.

    Rows are sorted by (speaker, group, utt_id). Duplicate ``utt_id`` values and missing
    shard files are errors. Calling it again after more writers finish merges their rows
    into the existing manifest.
    """
    root = Path(out_dir)
    partials = sorted(root.glob("manifest-*.parquet"))
    tables = [pq.read_table(p) for p in partials]
    manifest = root / MANIFEST_NAME
    if manifest.exists():
        tables.insert(0, pq.read_table(manifest))
    if not tables:
        raise FileNotFoundError(f"no manifests to finalise in {root}")
    table = _concat_tables(tables)
    dupes = sorted(k for k, v in Counter(table.column("utt_id").to_pylist()).items() if v > 1)
    if dupes:
        raise ValueError(f"duplicate utt_id values: {dupes[:10]}")
    for shard in set(table.column("shard").to_pylist()):
        if not (root / shard).is_file():
            raise FileNotFoundError(f"manifest references missing shard {shard}")
    table = table.sort_by([("speaker", "ascending"), ("group", "ascending"), ("utt_id", "ascending")])
    tmp = root / f".{MANIFEST_NAME}.tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, manifest)
    for p in partials:
        p.unlink()
    _write_info(root, table, name=name, extra=info)
    return manifest


def dataset_info(root: str | Path) -> dict[str, Any]:
    """Read ``dataset_info.json`` of a prepared dataset."""
    return json.loads((Path(root) / INFO_NAME).read_text())


def _write_info(root: Path, table: pa.Table, *, name: str, extra: Mapping[str, Any] | None) -> None:
    shards = sorted(set(table.column("shard").to_pylist()))
    samples = int(pc.sum(table.column("num_samples")).as_py() or 0)
    info: dict[str, Any] = {
        "name": name,
        "format": "earmark-int16-shards",
        "sample_rate": C.SAMPLE_RATE,
        "dtype": "int16",
        "scale": INT16_SCALE,
        "contract_version": C.CONTRACT_VERSION,
        "contract_hash": C.CONTRACT_HASH,
        "num_rows": table.num_rows,
        "num_speakers": len(set(table.column("speaker").to_pylist())),
        "total_seconds": samples / C.SAMPLE_RATE,
        "total_bytes": int(sum((root / s).stat().st_size for s in shards)),
        "shards": shards,
        "columns": table.column_names,
    }
    if extra:
        info.update(dict(extra))
    (root / INFO_NAME).write_text(json.dumps(info, indent=2, sort_keys=True, default=str) + "\n")


# --------------------------------------------------------------------------- pools


def assign_pools(
    speaker: Sequence[str] | np.ndarray,
    group: Sequence[str] | np.ndarray,
    seconds: Sequence[float] | np.ndarray,
    *,
    order: Sequence[Any] | np.ndarray | None = None,
    enrol_seconds: float = 40.0,
) -> np.ndarray:
    """Split each speaker's utterances into ``enrol``, ``target`` and ``spare`` pools.

    The unit is the group (a chapter or a recording session). Per speaker:

    * One group only: every utterance is ``target``. The speaker gets no enrolment, so it
      serves only as an interferer (the embedding step skips it).
    * Otherwise the enrolment group is the shortest group holding at least
      ``enrol_seconds``. Its utterances, in ``order``, become ``enrol`` until they reach
      ``enrol_seconds``; the rest of that group becomes ``spare`` (usable as interferer
      audio, never as a target). Every other group is ``target``.
    * If no single group is long enough, the shortest groups are taken whole for
      enrolment until ``enrol_seconds`` is reached, always leaving the longest group for
      targets.

    The result is deterministic (ties break on the group name). ``order`` defaults to the
    input order.
    """
    spk = np.asarray(speaker, dtype=object)
    grp = np.asarray(group, dtype=object)
    sec = np.asarray(seconds, dtype=np.float64)
    n = len(spk)
    if not (len(grp) == len(sec) == n):
        raise ValueError("speaker, group and seconds must have the same length")
    key = np.arange(n) if order is None else np.asarray(order)
    if len(key) != n:
        raise ValueError("order must have the same length as speaker")
    pools = np.full(n, POOL_TARGET, dtype=object)
    if n == 0:
        return pools
    _, spk_code = np.unique(spk.astype(str), return_inverse=True)
    rank = np.argsort(np.argsort(key, kind="stable"), kind="stable")
    by_speaker = np.lexsort((rank, spk_code))
    bounds = np.flatnonzero(np.diff(spk_code[by_speaker])) + 1
    for rows in np.split(by_speaker, bounds):
        groups = grp[rows].astype(str)
        names = sorted(set(groups.tolist()))
        if len(names) < 2:
            continue
        totals = {g: float(sec[rows][groups == g].sum()) for g in names}
        by_size = sorted(names, key=lambda g: (totals[g], g))
        long_enough = [g for g in by_size if totals[g] >= enrol_seconds]
        if long_enough:
            chosen = long_enough[0]
            acc = 0.0
            for r in rows[groups == chosen]:  # rows are already in `order`
                if acc < enrol_seconds:
                    pools[r] = POOL_ENROL
                    acc += sec[r]
                else:
                    pools[r] = POOL_SPARE
        else:
            acc = 0.0
            for g in by_size[:-1]:
                pools[rows[groups == g]] = POOL_ENROL
                acc += totals[g]
                if acc >= enrol_seconds:
                    break
    return pools


def check_pools_disjoint(
    speaker: Sequence[str] | np.ndarray,
    group: Sequence[str] | np.ndarray,
    pool: Sequence[str] | np.ndarray,
) -> None:
    """Raise :class:`PoolLeakError` if any (speaker, group) has both enrol and target rows."""
    spk = np.asarray(speaker, dtype=object).astype(str)
    grp = np.asarray(group, dtype=object).astype(str)
    pl = np.asarray(pool, dtype=object).astype(str)
    unknown = set(pl.tolist()) - set(POOLS)
    if unknown:
        raise ValueError(f"unknown pool values {sorted(unknown)}")
    pairs = np.char.add(np.char.add(spk, "\x1f"), grp)
    enrol = set(pairs[pl == POOL_ENROL].tolist())
    target = set(pairs[pl == POOL_TARGET].tolist())
    both = sorted(enrol & target)
    if both:
        shown = [b.replace("\x1f", "/") for b in both[:10]]
        raise PoolLeakError(f"groups in both enrolment and target pools: {shown}")


def add_pools(root: str | Path, *, enrol_seconds: float = 40.0) -> np.ndarray:
    """Assign pools from exact durations and store them as the manifest's ``pool`` column.

    Used after writing a corpus whose pools were not known before decoding (VCTK,
    LibriTTS-R train.clean.100). Returns the pool array.
    """
    root = Path(root)
    table = pq.read_table(root / MANIFEST_NAME)
    seconds = np.asarray(table.column("num_samples").to_numpy(), dtype=np.float64) / C.SAMPLE_RATE
    pools = assign_pools(
        table.column("speaker").to_numpy(zero_copy_only=False),
        table.column("group").to_numpy(zero_copy_only=False),
        seconds,
        order=table.column("utt_id").to_numpy(zero_copy_only=False),
        enrol_seconds=enrol_seconds,
    )
    check_pools_disjoint(
        table.column("speaker").to_numpy(zero_copy_only=False),
        table.column("group").to_numpy(zero_copy_only=False),
        pools,
    )
    if "pool" in table.column_names:
        table = table.drop_columns(["pool"])
    table = table.append_column("pool", pa.array(pools.tolist(), pa.string()))
    tmp = root / f".{MANIFEST_NAME}.tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, root / MANIFEST_NAME)
    info = dataset_info(root) if (root / INFO_NAME).exists() else {}
    info.update(
        {
            "pool_enrol_seconds": enrol_seconds,
            "pool_counts": {p: int(np.count_nonzero(pools == p)) for p in POOLS},
            "columns": table.column_names,
        }
    )
    (root / INFO_NAME).write_text(json.dumps(info, indent=2, sort_keys=True, default=str) + "\n")
    return pools


# --------------------------------------------------------------------------- reader


class _ShardFiles:
    """Shard paths plus lazily opened read-only memmaps, shared between corpus views."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = [Path(p) for p in paths]
        self._maps: dict[int, np.memmap] = {}

    def get(self, index: int) -> np.memmap:
        mm = self._maps.get(index)
        if mm is None:
            mm = np.memmap(self.paths[index], dtype="<i2", mode="r")
            self._maps[index] = mm
        return mm


class ShardedCorpus:
    """Read-only view of one or more prepared datasets, backed by memmapped shards.

    Rows keep manifest order. Columns come back as NumPy arrays (strings as object
    arrays). :meth:`subset` and :meth:`concat` return new views that share the memmaps.
    """

    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        table = pq.read_table(root / MANIFEST_NAME)
        missing = [c for c in REQUIRED_COLUMNS if c not in table.column_names]
        if missing:
            raise ValueError(f"{root / MANIFEST_NAME} lacks columns {missing}")
        shard_names = table.column("shard").to_numpy(zero_copy_only=False).astype(str)
        unique, index = np.unique(shard_names, return_inverse=True)
        self._files = _ShardFiles([root / s for s in unique])
        self._shard_idx = index.astype(np.int64)
        self._columns = {
            name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names
        }
        self._offset = np.asarray(self._columns["offset"], dtype=np.int64)
        self._length = np.asarray(self._columns["num_samples"], dtype=np.int64)
        self.roots: tuple[Path, ...] = (root,)

    @classmethod
    def _view(
        cls,
        files: _ShardFiles,
        shard_idx: np.ndarray,
        columns: dict[str, np.ndarray],
        roots: tuple[Path, ...],
    ) -> ShardedCorpus:
        self = cls.__new__(cls)
        self._files = files
        self._shard_idx = shard_idx
        self._columns = columns
        self._offset = np.asarray(columns["offset"], dtype=np.int64)
        self._length = np.asarray(columns["num_samples"], dtype=np.int64)
        self.roots = roots
        return self

    def __len__(self) -> int:
        return len(self._length)

    def __repr__(self) -> str:
        roots = ", ".join(str(r) for r in self.roots)
        return f"ShardedCorpus({len(self)} rows, {self.seconds().sum() / 3600:.2f} h, {roots})"

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    def has_column(self, name: str) -> bool:
        return name in self._columns

    def column(self, name: str) -> np.ndarray:
        """One manifest column (raises ``KeyError`` if absent)."""
        return self._columns[name]

    @property
    def speaker(self) -> np.ndarray:
        return self._columns["speaker"]

    @property
    def group(self) -> np.ndarray:
        return self._columns["group"]

    @property
    def num_samples(self) -> np.ndarray:
        return self._length

    @property
    def peak_energy(self) -> np.ndarray:
        return np.asarray(self._columns["peak_energy"], dtype=np.float64)

    def seconds(self) -> np.ndarray:
        return self._length / C.SAMPLE_RATE

    def audio_int16(self, row: int, start: int = 0, length: int | None = None) -> np.ndarray:
        """Samples ``[start, start + length)`` of a row as an int16 memmap view (clamped)."""
        n = int(self._length[row])
        start = min(max(0, int(start)), n)
        stop = n if length is None else min(n, start + max(0, int(length)))
        base = int(self._offset[row])
        return self._files.get(int(self._shard_idx[row]))[base + start : base + stop]

    def audio(self, row: int, start: int = 0, length: int | None = None) -> np.ndarray:
        """Like :meth:`audio_int16` but as float32 in [-1, 1)."""
        return from_int16(self.audio_int16(row, start, length))

    def subset(self, rows: np.ndarray | Sequence[int]) -> ShardedCorpus:
        """View of the given rows (integer indices or a boolean mask)."""
        idx = np.asarray(rows)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        idx = idx.astype(np.int64)
        columns = {k: v[idx] for k, v in self._columns.items()}
        return ShardedCorpus._view(self._files, self._shard_idx[idx], columns, self.roots)

    def where(self, name: str, values: Iterable[Any]) -> ShardedCorpus:
        """Rows whose column ``name`` is in ``values``."""
        wanted = set(values)
        col = self._columns[name]
        return self.subset(np.fromiter((v in wanted for v in col), dtype=bool, count=len(col)))

    @staticmethod
    def concat(corpora: Sequence[ShardedCorpus]) -> ShardedCorpus:
        """One view over several corpora; columns missing from a corpus become null."""
        if not corpora:
            raise ValueError("nothing to concatenate")
        paths: list[Path] = []
        shard_idx: list[np.ndarray] = []
        maps: dict[int, np.memmap] = {}
        for corpus in corpora:
            base = len(paths)
            paths.extend(corpus._files.paths)
            maps.update({base + k: v for k, v in corpus._files._maps.items()})
            shard_idx.append(corpus._shard_idx + base)
        files = _ShardFiles(paths)
        files._maps = maps
        names: list[str] = []
        for corpus in corpora:
            names.extend(n for n in corpus.column_names if n not in names)
        columns: dict[str, np.ndarray] = {}
        for name in names:
            parts = []
            for corpus in corpora:
                if corpus.has_column(name):
                    parts.append(corpus.column(name))
                else:
                    parts.append(np.full(len(corpus), None, dtype=object))
            if all(p.dtype != object for p in parts):
                columns[name] = np.concatenate(parts)
            else:
                columns[name] = np.concatenate([p.astype(object) for p in parts])
        roots = tuple(r for c in corpora for r in c.roots)
        return ShardedCorpus._view(files, np.concatenate(shard_idx), columns, roots)
