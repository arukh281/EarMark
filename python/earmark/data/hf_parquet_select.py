"""Column projection and row-group fetch from Hugging Face parquet datasets.

LibriTTS-R is on the Hub as parquet (``mythicinfinity/libritts_r``). Files sit under
``data/<split>/<split>-NNNNN-of-MMMMM.parquet``; each row holds ``audio`` (a struct of
``bytes`` and ``path``, 24 kHz WAV), ``text_normalized``, ``text_original``,
``speaker_id``, ``path``, ``chapter_id`` and ``id``. Row groups hold 100 rows, 25-35 MB,
almost all of it ``audio.bytes`` (checked 2026-09-11).

Two passes keep the transfer to what is needed:

1. :func:`read_index` reads only small columns (by default ``id``, ``speaker_id``,
   ``chapter_id``, ``text_normalized``) of every row group. Parquet stores each column
   chunk contiguously, so the reader fetches the footer plus those chunks by HTTP range
   request and never touches ``audio``.
2. A selection (every row, or :func:`select_speaker_capped`) becomes a
   :class:`FetchPlan`: the row groups that contain at least one selected row.
   :func:`iter_selected_rows` fetches those row groups one at a time and yields only
   the selected rows; :func:`export_to_shards` decodes, resamples and shards them.

Files are opened with ``cache_type="none"``, so every read is exactly the byte range
pyarrow asks for. Authentication uses ``HF_TOKEN`` from the environment. Any fsspec
filesystem works (the tests use the local one).
"""

from __future__ import annotations

import io
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from earmark.data.shards import (
    DEFAULT_MAX_SHARD_BYTES,
    POOL_ENROL,
    POOL_TARGET,
    ShardWriter,
    assign_pools,
    to_mono,
)
from earmark.data.splits import LIBRI, speaker_key

__all__ = [
    "AUDIO_COLUMN",
    "DEFAULT_CHARS_PER_SECOND",
    "ExportJob",
    "FetchPlan",
    "FILE_COLUMN",
    "INDEX_COLUMNS",
    "LIBRITTS_R_HELDOUT_SPLITS",
    "LIBRITTS_R_REPO",
    "LIBRITTS_R_SPLITS",
    "ROW_COLUMN",
    "ROW_GROUP_COLUMN",
    "SELECTION_STRATEGIES",
    "SpeakerCap",
    "decode_audio",
    "estimate_seconds",
    "export_to_shards",
    "hf_filesystem",
    "iter_selected_rows",
    "libritts_r_heldout_speakers",
    "libritts_r_parquet_files",
    "list_parquet_files",
    "make_export_jobs",
    "open_parquet",
    "plan_fetch",
    "read_index",
    "read_speakers",
    "run_export_job",
    "select_all",
    "select_speaker_capped",
]

LIBRITTS_R_REPO = "mythicinfinity/libritts_r"
LIBRITTS_R_SPLITS: tuple[str, ...] = (
    "train.clean.100",
    "train.clean.360",
    "train.other.500",
    "dev.clean",
    "dev.other",
    "test.clean",
    "test.other",
)
#: LibriTTS-R splits whose speakers must never appear in training (Earmark-Synth dev/test).
LIBRITTS_R_HELDOUT_SPLITS: tuple[str, ...] = ("dev.clean", "dev.other", "test.clean", "test.other")
#: How :func:`select_speaker_capped` fills each speaker's target budget.
SELECTION_STRATEGIES: tuple[str, ...] = ("row_groups", "round_robin")
INDEX_COLUMNS: tuple[str, ...] = ("id", "speaker_id", "chapter_id", "text_normalized")
AUDIO_COLUMN = "audio"
FILE_COLUMN = "_file"
ROW_GROUP_COLUMN = "_row_group"
ROW_COLUMN = "_row"
#: Read-speech rate used to estimate durations from text before any audio is fetched.
DEFAULT_CHARS_PER_SECOND = 14.0


# --------------------------------------------------------------------------- filesystem


def hf_filesystem(token: str | None = None) -> Any:
    """An authenticated ``HfFileSystem`` (token argument, else ``HF_TOKEN``, else anonymous)."""
    from huggingface_hub import HfFileSystem

    tok = token if token is not None else (os.environ.get("HF_TOKEN") or None)
    return HfFileSystem(token=tok)


def list_parquet_files(fs: Any, pattern: str) -> list[str]:
    """Sorted parquet paths matching a glob on ``fs``; raises if none match."""
    files = sorted(fs.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no parquet files match {pattern}")
    return files


def libritts_r_parquet_files(
    fs: Any, split: str, *, repo_id: str = LIBRITTS_R_REPO, revision: str | None = None
) -> list[str]:
    """Parquet files of one LibriTTS-R split on the Hub (``revision`` pins a commit)."""
    if split not in LIBRITTS_R_SPLITS:
        raise ValueError(f"unknown LibriTTS-R split {split!r}; expected one of {LIBRITTS_R_SPLITS}")
    rev = f"@{revision}" if revision else ""
    return list_parquet_files(fs, f"datasets/{repo_id}{rev}/data/{split}/*.parquet")


@contextmanager
def open_parquet(fs: Any, path: str) -> Iterator[pq.ParquetFile]:
    """Open a parquet file on ``fs`` for exact-range reads (no read-ahead cache)."""
    try:
        handle = fs.open(path, "rb", cache_type="none")
    except TypeError:
        handle = fs.open(path, "rb")
    try:
        yield pq.ParquetFile(handle)
    finally:
        handle.close()


# --------------------------------------------------------------------------- index


def _index_file(fs: Any, path: str, columns: Sequence[str]) -> list[pa.Table]:
    tables: list[pa.Table] = []
    with open_parquet(fs, path) as pf:
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=list(columns))
            n = t.num_rows
            t = t.append_column(FILE_COLUMN, pa.array([path] * n, pa.string()))
            t = t.append_column(ROW_GROUP_COLUMN, pa.array(np.full(n, rg, np.int32)))
            t = t.append_column(ROW_COLUMN, pa.array(np.arange(n, dtype=np.int32)))
            tables.append(t)
    return tables


def read_index(
    fs: Any,
    files: Sequence[str],
    columns: Sequence[str] = INDEX_COLUMNS,
    *,
    progress: Callable[[str], None] | None = None,
    max_workers: int = 1,
) -> pa.Table:
    """Read only ``columns`` of every row group, tagging rows with their location.

    The result has the requested columns plus ``_file``, ``_row_group`` and ``_row``
    (the row's position inside its row group), in file order. Each row group costs one
    small range request per column (about 1.2 s from outside the Hub, so train.clean.360's
    roughly 1,165 row groups take about 23 minutes serially); ``max_workers`` threads read
    several files at once.
    """
    paths = list(files)
    per_file: list[list[pa.Table]] = []
    if max_workers > 1 and len(paths) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(paths))) as pool:
            futures = [pool.submit(_index_file, fs, p, columns) for p in paths]
            for path, future in zip(paths, futures, strict=True):
                per_file.append(future.result())
                if progress is not None:
                    progress(path)
    else:
        for path in paths:
            per_file.append(_index_file(fs, path, columns))
            if progress is not None:
                progress(path)
    tables = [t for group in per_file for t in group]
    if not tables:
        raise ValueError("no row groups read")
    return pa.concat_tables(tables)


def read_speakers(
    fs: Any, files: Sequence[str], *, namespace: str = LIBRI, max_workers: int = 1
) -> frozenset[str]:
    """Namespaced speakers of parquet files, reading only the ``speaker_id`` column."""
    index = read_index(fs, files, ("speaker_id",), max_workers=max_workers)
    return frozenset(str(s) for s in _speakers(index, namespace).tolist())


def libritts_r_heldout_speakers(
    fs: Any,
    *,
    splits: Sequence[str] = LIBRITTS_R_HELDOUT_SPLITS,
    revision: str | None = None,
    max_workers: int = 8,
) -> frozenset[str]:
    """Every speaker of the LibriTTS-R dev and test splits (the Earmark-Synth speakers)."""
    files = [f for s in splits for f in libritts_r_parquet_files(fs, s, revision=revision)]
    return read_speakers(fs, files, max_workers=max_workers)


def estimate_seconds(
    texts: Iterable[str | None],
    *,
    chars_per_second: float = DEFAULT_CHARS_PER_SECOND,
    minimum_seconds: float = 0.5,
) -> np.ndarray:
    """Duration estimate from transcript length, used only to plan what to fetch."""
    lengths = np.fromiter((len(t or "") for t in texts), dtype=np.float64)
    return np.maximum(lengths / chars_per_second, minimum_seconds)


def _speakers(index: pa.Table, namespace: str) -> np.ndarray:
    raw = index.column("speaker_id").to_numpy(zero_copy_only=False)
    return np.array([speaker_key(namespace, s) for s in raw], dtype=object)


def select_all(
    index: pa.Table,
    *,
    namespace: str = LIBRI,
    chars_per_second: float = DEFAULT_CHARS_PER_SECOND,
) -> pa.Table:
    """Every row, with ``speaker`` (namespaced) and ``est_seconds`` columns added.

    Pools are assigned after writing, from exact durations (:func:`.shards.add_pools`).
    """
    est = estimate_seconds(
        index.column("text_normalized").to_pylist(), chars_per_second=chars_per_second
    )
    out = index.append_column("speaker", pa.array(_speakers(index, namespace).tolist(), pa.string()))
    return out.append_column("est_seconds", pa.array(est))


def select_speaker_capped(
    index: pa.Table,
    *,
    cap_seconds: float = 180.0,
    enrol_seconds: float = 40.0,
    chars_per_second: float = DEFAULT_CHARS_PER_SECOND,
    overselect: float = 1.0,
    namespace: str = LIBRI,
    strategy: str = "row_groups",
) -> pa.Table:
    """Pick about ``cap_seconds`` of audio per speaker, split into pools by chapter.

    Pools come from :func:`.shards.assign_pools` on estimated durations: the enrolment
    chapter contributes up to ``enrol_seconds`` (``enrol`` rows), then ``target`` rows
    from the speaker's other chapters are added until the speaker's estimated total
    reaches ``cap_seconds * overselect``. Spare rows are dropped. Single-chapter speakers
    get ``target`` rows only. Adds ``speaker``, ``pool`` and ``est_seconds`` columns.

    ``strategy`` decides which target rows fill the budget:

    * ``"row_groups"`` (default) keeps the fetch small. Speakers are visited in file
      order, and each one's targets come first from row groups the plan already fetches,
      then from the row groups holding the most of that speaker's candidate audio. A
      speaker's targets therefore often come from one chapter.
    * ``"round_robin"`` spreads targets across all of the speaker's other chapters.

    Row groups are the unit of transfer and every speaker needs at least one, so a
    selection that keeps some of every speaker still touches most row groups; the
    locality order removes only the extra ones. Either way nothing but the selected rows
    is ever decoded or stored.
    """
    if strategy not in SELECTION_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {SELECTION_STRATEGIES}")
    n = index.num_rows
    speakers = _speakers(index, namespace)
    groups = index.column("chapter_id").to_numpy(zero_copy_only=False).astype(str)
    ids = index.column("id").to_numpy(zero_copy_only=False).astype(str)
    est = estimate_seconds(
        index.column("text_normalized").to_pylist(), chars_per_second=chars_per_second
    )
    pools = assign_pools(speakers, groups, est, order=ids, enrol_seconds=enrol_seconds)
    position, block = _row_positions(index)
    keep = np.zeros(n, dtype=bool)
    _, code = np.unique(speakers.astype(str), return_inverse=True)
    by_speaker = np.lexsort((ids, code))
    per_speaker = np.split(by_speaker, np.flatnonzero(np.diff(code[by_speaker])) + 1) if n else []
    use_blocks = strategy == "row_groups" and block is not None
    if use_blocks:
        per_speaker.sort(key=lambda rows: int(position[rows].min()))
    touched: set[int] = set()
    for rows in per_speaker:
        enrol = rows[pools[rows] == POOL_ENROL]
        keep[enrol] = True
        budget = max(0.0, (cap_seconds - float(est[enrol].sum())) * overselect)
        target = rows[pools[rows] == POOL_TARGET]
        if use_blocks and block is not None:
            touched.update(int(b) for b in block[enrol])
            _take_by_block(target, est, position, block, budget, keep, touched)
        else:
            _take_round_robin(target, groups, est, budget, keep)
    rows_kept = np.flatnonzero(keep)
    out = index.take(pa.array(rows_kept))
    out = out.append_column("speaker", pa.array(speakers[rows_kept].tolist(), pa.string()))
    out = out.append_column("pool", pa.array(pools[rows_kept].tolist(), pa.string()))
    return out.append_column("est_seconds", pa.array(est[rows_kept]))


def _row_positions(index: pa.Table) -> tuple[np.ndarray, np.ndarray | None]:
    """Global row position and row-group id of every index row (``None`` without locations)."""
    n = index.num_rows
    if not all(c in index.column_names for c in (FILE_COLUMN, ROW_GROUP_COLUMN, ROW_COLUMN)):
        return np.arange(n, dtype=np.int64), None
    files = index.column(FILE_COLUMN).to_numpy(zero_copy_only=False).astype(str)
    rg = index.column(ROW_GROUP_COLUMN).to_numpy().astype(np.int64)
    row = index.column(ROW_COLUMN).to_numpy().astype(np.int64)
    _, file_code = np.unique(files, return_inverse=True)
    rg_span = int(rg.max()) + 1 if n else 1
    row_span = int(row.max()) + 1 if n else 1
    block = file_code.astype(np.int64) * rg_span + rg
    return block * row_span + row, block


def _take_round_robin(
    target: np.ndarray, groups: np.ndarray, est: np.ndarray, budget: float, keep: np.ndarray
) -> None:
    by_group: dict[str, list[int]] = defaultdict(list)
    for r in target:  # already in id order
        by_group[groups[r]].append(int(r))
    queues = [by_group[g] for g in sorted(by_group)]
    acc = 0.0
    depth = 0
    while acc < budget and any(depth < len(q) for q in queues):
        for q in queues:
            if depth < len(q) and acc < budget:
                keep[q[depth]] = True
                acc += float(est[q[depth]])
        depth += 1


def _take_by_block(
    target: np.ndarray,
    est: np.ndarray,
    position: np.ndarray,
    block: np.ndarray,
    budget: float,
    keep: np.ndarray,
    touched: set[int],
) -> None:
    by_block: dict[int, list[int]] = defaultdict(list)
    for r in target[np.argsort(position[target], kind="stable")]:
        by_block[int(block[r])].append(int(r))
    totals = {b: float(est[rows].sum()) for b, rows in by_block.items()}
    acc = 0.0
    for b in sorted(by_block, key=lambda b: (b not in touched, -totals[b], b)):
        if acc >= budget:
            break
        touched.add(b)
        for r in by_block[b]:
            if acc >= budget:
                break
            keep[r] = True
            acc += float(est[r])


# --------------------------------------------------------------------------- fetch plan


@dataclass(frozen=True)
class FetchPlan:
    """Which row groups to fetch, which rows to keep in each, and what that costs.

    ``group_bytes`` and ``file_bytes`` count the compressed size of ``columns`` only:
    the planned row groups versus every row group of the files involved.
    """

    rows: Mapping[tuple[str, int], tuple[int, ...]]
    group_bytes: Mapping[tuple[str, int], int]
    file_bytes: Mapping[str, int]
    columns: tuple[str, ...]

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(sorted({f for f, _ in self.rows}))

    @property
    def n_rows(self) -> int:
        return sum(len(v) for v in self.rows.values())

    @property
    def n_row_groups(self) -> int:
        return len(self.rows)

    @property
    def fetch_bytes(self) -> int:
        return int(sum(self.group_bytes.values()))

    @property
    def total_bytes(self) -> int:
        return int(sum(self.file_bytes.values()))

    def row_groups(self, path: str) -> tuple[int, ...]:
        return tuple(sorted(rg for f, rg in self.rows if f == path))

    def for_file(self, path: str) -> FetchPlan:
        """The part of the plan that touches one file."""
        return FetchPlan(
            rows={k: v for k, v in self.rows.items() if k[0] == path},
            group_bytes={k: v for k, v in self.group_bytes.items() if k[0] == path},
            file_bytes={path: self.file_bytes.get(path, 0)},
            columns=self.columns,
        )

    def summary(self) -> str:
        frac = self.fetch_bytes / self.total_bytes if self.total_bytes else 0.0
        return (
            f"{self.n_rows} rows in {self.n_row_groups} row groups of {len(self.files)} files; "
            f"fetch {self.fetch_bytes / 1e9:.2f} GB of {self.total_bytes / 1e9:.2f} GB ({frac:.0%})"
        )


def _top_level(path_in_schema: str) -> str:
    return path_in_schema.split(".", 1)[0]


def plan_fetch(
    fs: Any, selection: pa.Table, columns: Sequence[str] = (AUDIO_COLUMN, "id")
) -> FetchPlan:
    """Plan the row-group fetch for ``selection`` (output of :func:`read_index` or a filter).

    Reads only parquet footers to size the plan.
    """
    wanted = set(columns)
    files = selection.column(FILE_COLUMN).to_numpy(zero_copy_only=False).astype(str)
    rgs = selection.column(ROW_GROUP_COLUMN).to_numpy()
    rws = selection.column(ROW_COLUMN).to_numpy()
    rows: dict[tuple[str, int], list[int]] = defaultdict(list)
    for f, rg, r in zip(files, rgs, rws, strict=True):
        rows[(str(f), int(rg))].append(int(r))
    group_bytes: dict[tuple[str, int], int] = {}
    file_bytes: dict[str, int] = {}
    for path in sorted(set(files.tolist())):
        with open_parquet(fs, path) as pf:
            md = pf.metadata
            total = 0
            for rg in range(md.num_row_groups):
                meta = md.row_group(rg)
                size = sum(
                    meta.column(j).total_compressed_size
                    for j in range(meta.num_columns)
                    if _top_level(meta.column(j).path_in_schema) in wanted
                )
                total += size
                if (path, rg) in rows:
                    group_bytes[(path, rg)] = size
            file_bytes[path] = total
    return FetchPlan(
        rows={k: tuple(sorted(set(v))) for k, v in rows.items()},
        group_bytes=group_bytes,
        file_bytes=file_bytes,
        columns=tuple(columns),
    )


def iter_selected_rows(
    fs: Any, plan: FetchPlan, columns: Sequence[str] | None = None
) -> Iterator[dict[str, Any]]:
    """Fetch the planned row groups one at a time and yield only the selected rows.

    Each yielded dict holds the requested columns plus ``_file``, ``_row_group`` and
    ``_row``. Memory stays at about one row group.
    """
    cols = list(columns or plan.columns)
    for path in plan.files:
        with open_parquet(fs, path) as pf:
            for rg in plan.row_groups(path):
                keep = plan.rows[(path, rg)]
                table = pf.read_row_group(rg, columns=cols).take(pa.array(keep))
                for pos, record in zip(keep, table.to_pylist(), strict=True):
                    record[FILE_COLUMN] = path
                    record[ROW_GROUP_COLUMN] = rg
                    record[ROW_COLUMN] = pos
                    yield record


def decode_audio(cell: Any) -> tuple[np.ndarray, int]:
    """Decode a Hugging Face ``Audio`` cell (``{"bytes", "path"}`` or raw bytes) to mono float32."""
    data = cell.get("bytes") if isinstance(cell, Mapping) else cell
    if not isinstance(data, (bytes, bytearray, memoryview)):
        path = cell.get("path") if isinstance(cell, Mapping) else None
        raise ValueError(f"audio cell holds no embedded bytes (path={path!r})")
    audio, sr = sf.read(io.BytesIO(bytes(data)), dtype="float32", always_2d=False)
    return np.asarray(to_mono(audio), dtype=np.float32), int(sr)


# --------------------------------------------------------------------------- export


class SpeakerCap:
    """Exact per-speaker budgets enforced while decoding (estimates only plan the fetch).

    An utterance is admitted while its speaker's exact total in that pool is still below
    the budget, so every pool reaches its budget and only the last utterance crosses it:
    the rule :func:`.shards.assign_pools` and :func:`select_speaker_capped` use on
    estimates. Enrolment rows get ``enrol_seconds``; target rows get ``cap_seconds``,
    minus ``enrol_seconds`` for speakers that have an enrolment pool.
    """

    def __init__(
        self,
        cap_seconds: float,
        *,
        enrol_seconds: float,
        speakers_with_enrolment: Iterable[str],
    ) -> None:
        self.cap_seconds = float(cap_seconds)
        self.enrol_seconds = float(enrol_seconds)
        self._with_enrol = set(speakers_with_enrolment)
        self.enrol: dict[str, float] = defaultdict(float)
        self.target: dict[str, float] = defaultdict(float)

    @classmethod
    def from_selection(
        cls, selection: pa.Table, cap_seconds: float, *, enrol_seconds: float
    ) -> SpeakerCap:
        speakers = selection.column("speaker").to_numpy(zero_copy_only=False)
        pools = selection.column("pool").to_numpy(zero_copy_only=False)
        with_enrol = set(speakers[pools == POOL_ENROL].tolist())
        return cls(cap_seconds, enrol_seconds=enrol_seconds, speakers_with_enrolment=with_enrol)

    def budget(self, speaker: str, pool: str) -> float:
        """Seconds a speaker's ``pool`` may reach before further utterances are refused."""
        if pool == POOL_ENROL:
            return self.enrol_seconds
        reserved = self.enrol_seconds if speaker in self._with_enrol else 0.0
        return max(0.0, self.cap_seconds - reserved)

    def admit(self, speaker: str, pool: str, seconds: float) -> bool:
        """Record and admit an utterance while its speaker's pool is below budget."""
        totals = self.enrol if pool == POOL_ENROL else self.target
        if totals[speaker] >= self.budget(speaker, pool):
            return False
        totals[speaker] += float(seconds)
        return True


def export_to_shards(
    fs: Any,
    plan: FetchPlan,
    selection: pa.Table,
    writer: ShardWriter,
    *,
    namespace: str = LIBRI,
    source: str = "libritts_r",
    split: str = "",
    cap: SpeakerCap | None = None,
    text_column: str = "text_normalized",
) -> dict[str, float]:
    """Fetch, decode, resample and shard the selected rows. Returns simple counters.

    Metadata (speaker, chapter, text, pool) comes from ``selection``; only the audio and
    the id are fetched. ``utt_id`` becomes ``"<namespace>:<id>"`` and ``group`` the
    chapter id.
    """
    files = selection.column(FILE_COLUMN).to_numpy(zero_copy_only=False).astype(str)
    rgs = selection.column(ROW_GROUP_COLUMN).to_numpy()
    rws = selection.column(ROW_COLUMN).to_numpy()
    lookup = {
        (str(f), int(g), int(r)): i for i, (f, g, r) in enumerate(zip(files, rgs, rws, strict=True))
    }
    speakers = (
        selection.column("speaker").to_numpy(zero_copy_only=False)
        if "speaker" in selection.column_names
        else _speakers(selection, namespace)
    )
    chapters = selection.column("chapter_id").to_numpy(zero_copy_only=False)
    texts = (
        selection.column(text_column).to_pylist() if text_column in selection.column_names else None
    )
    pools = (
        selection.column("pool").to_numpy(zero_copy_only=False)
        if "pool" in selection.column_names
        else None
    )
    stats = {"rows": 0.0, "seconds": 0.0, "skipped_cap": 0.0}
    for record in iter_selected_rows(fs, plan, (AUDIO_COLUMN, "id")):
        i = lookup[(record[FILE_COLUMN], int(record[ROW_GROUP_COLUMN]), int(record[ROW_COLUMN]))]
        audio, sr = decode_audio(record[AUDIO_COLUMN])
        seconds = len(audio) / sr
        pool = str(pools[i]) if pools is not None else None
        if cap is not None and not cap.admit(str(speakers[i]), pool or POOL_TARGET, seconds):
            stats["skipped_cap"] += 1
            continue
        meta: dict[str, Any] = {"chapter_id": str(chapters[i]), "source": source, "split": split}
        if texts is not None:
            meta["text"] = texts[i]
        if pool is not None:
            meta["pool"] = pool
        writer.add(
            audio,
            sr,
            utt_id=f"{namespace}:{record['id']}",
            speaker=str(speakers[i]),
            group=str(chapters[i]),
            **meta,
        )
        stats["rows"] += 1
        stats["seconds"] += seconds
    return stats


@dataclass(frozen=True)
class ExportJob:
    """One worker's share of an export (picklable, for ``ProcessPoolExecutor``)."""

    plan: FetchPlan
    selection: pa.Table
    out_dir: str
    prefix: str
    source: str = "libritts_r"
    split: str = ""
    namespace: str = LIBRI
    cap_seconds: float | None = None
    enrol_seconds: float = 40.0
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES
    fs_factory: Callable[[], Any] | None = field(default=None, compare=False)


def make_export_jobs(
    plan: FetchPlan, selection: pa.Table, out_dir: str, *, prefix: str, **kwargs: Any
) -> list[ExportJob]:
    """Split a plan into one job per parquet file (shard prefix ``<prefix>-fNNN``)."""
    files_col = selection.column(FILE_COLUMN).to_numpy(zero_copy_only=False).astype(str)
    jobs = []
    for k, path in enumerate(plan.files):
        mask = pa.array(files_col == path)
        jobs.append(
            ExportJob(
                plan=plan.for_file(path),
                selection=selection.filter(mask),
                out_dir=str(out_dir),
                prefix=f"{prefix}-f{k:03d}",
                **kwargs,
            )
        )
    return jobs


def run_export_job(job: ExportJob) -> dict[str, float]:
    """Run one :class:`ExportJob` in the current process (creates its own filesystem).

    Each job enforces the per-speaker cap on its own rows, so a speaker whose rows span
    two files can exceed the cap by at most one file's share.
    """
    fs = (job.fs_factory or hf_filesystem)()
    cap = (
        SpeakerCap.from_selection(job.selection, job.cap_seconds, enrol_seconds=job.enrol_seconds)
        if job.cap_seconds is not None
        else None
    )
    with ShardWriter(job.out_dir, prefix=job.prefix, max_shard_bytes=job.max_shard_bytes) as writer:
        return export_to_shards(
            fs,
            job.plan,
            job.selection,
            writer,
            namespace=job.namespace,
            source=job.source,
            split=job.split,
            cap=cap,
        )
