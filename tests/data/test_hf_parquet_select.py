"""Column projection, row-group planning and export on LibriTTS-R-shaped parquet files.

The files sit on the local filesystem behind fsspec, the same interface ``HfFileSystem``
provides on Kaggle. A byte-counting wrapper shows that the index pass never reads audio.
"""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from earmark import constants as C
from earmark.data import hf_parquet_select as H
from earmark.data.shards import (
    POOL_ENROL,
    POOL_TARGET,
    ShardedCorpus,
    ShardWriter,
    check_pools_disjoint,
    finalize_dataset,
)

AUDIO_SR = 24000
ROWS_PER_GROUP = 4
ROWS_PER_FILE = 20
#: speaker -> chapters; every chapter holds four utterances and fills exactly one row group.
LAYOUT = {"11": 3, "22": 3, "33": 1}
SCHEMA = pa.schema(
    [
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("text_normalized", pa.string()),
        ("text_original", pa.string()),
        ("speaker_id", pa.string()),
        ("path", pa.string()),
        ("chapter_id", pa.string()),
        ("id", pa.string()),
    ]
)


def _wav_bytes(seconds: float, f0: float) -> bytes:
    t = np.arange(int(round(seconds * AUDIO_SR))) / AUDIO_SR
    buf = io.BytesIO()
    sf.write(buf, (0.2 * np.sin(2 * np.pi * f0 * t)).astype(np.float32), AUDIO_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _records() -> list[dict[str, Any]]:
    records = []
    for spk, chapters in LAYOUT.items():
        for ch in range(chapters):
            chapter = f"{spk}{ch:03d}"
            for u in range(4):
                seconds = 2.0 + 0.5 * ((u + ch) % 3)  # text length makes the estimate exact
                uid = f"{spk}_{chapter}_{u:06d}_000000"
                text = "x" * int(round(seconds * H.DEFAULT_CHARS_PER_SECOND))
                records.append(
                    {
                        "audio": {"bytes": _wav_bytes(seconds, 150.0 + int(spk)), "path": f"{uid}.wav"},
                        "text_normalized": text,
                        "text_original": text,
                        "speaker_id": spk,
                        "path": f"train/{spk}/{chapter}/{uid}.wav",
                        "chapter_id": chapter,
                        "id": uid,
                    }
                )
    return records


def _local_fs() -> Any:
    return fsspec.filesystem("file")


@pytest.fixture(scope="module")
def parquet_files(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    root = tmp_path_factory.mktemp("libritts_like") / "data" / "train.clean.360"
    root.mkdir(parents=True)
    records = _records()
    n_files = -(-len(records) // ROWS_PER_FILE)
    paths = []
    for k in range(n_files):
        chunk = records[k * ROWS_PER_FILE : (k + 1) * ROWS_PER_FILE]
        path = root / f"train.clean.360-{k:05d}-of-{n_files:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(chunk, schema=SCHEMA), path, row_group_size=ROWS_PER_GROUP)
        paths.append(str(path))
    return paths


@pytest.fixture(scope="module")
def index(parquet_files: list[str]) -> pa.Table:
    return H.read_index(_local_fs(), parquet_files)


class _CountingFile(io.RawIOBase):
    def __init__(self, fh: Any, counter: list[int]) -> None:
        super().__init__()
        self._fh = fh
        self._counter = counter

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, pos: int, whence: int = 0) -> int:
        return self._fh.seek(pos, whence)

    def tell(self) -> int:
        return self._fh.tell()

    def readinto(self, buffer: Any) -> int:
        data = self._fh.read(len(buffer))
        buffer[: len(data)] = data
        self._counter[0] += len(data)
        return len(data)

    def close(self) -> None:
        self._fh.close()
        super().close()


class CountingFS:
    """A local filesystem that counts the bytes read through it."""

    def __init__(self) -> None:
        self.fs = _local_fs()
        self.counter = [0]

    def glob(self, pattern: str) -> list[str]:
        return self.fs.glob(pattern)

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> io.RawIOBase:
        return _CountingFile(self.fs.open(path, mode), self.counter)


def _by_speaker(selection: pa.Table) -> dict[str, dict[str, list[tuple[str, float]]]]:
    out: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for spk, pool, chapter, est in zip(
        selection.column("speaker").to_pylist(),
        selection.column("pool").to_pylist(),
        selection.column("chapter_id").to_pylist(),
        selection.column("est_seconds").to_pylist(),
        strict=True,
    ):
        out[spk][pool].append((chapter, est))
    return out


def test_index_pass_reads_only_the_small_columns(parquet_files: list[str]) -> None:
    fs = CountingFS()
    index = H.read_index(fs, parquet_files)
    total = sum(Path(p).stat().st_size for p in parquet_files)
    assert index.num_rows == 28
    assert fs.counter[0] < 0.1 * total
    assert set(index.column_names) == set(H.INDEX_COLUMNS) | {H.FILE_COLUMN, H.ROW_GROUP_COLUMN, H.ROW_COLUMN}
    assert index.column(H.ROW_GROUP_COLUMN).to_pylist()[:8] == [0] * 4 + [1] * 4
    assert index.column(H.ROW_COLUMN).to_pylist()[:5] == [0, 1, 2, 3, 0]


def test_threaded_index_matches_serial(parquet_files: list[str], index: pa.Table) -> None:
    done: list[str] = []
    threaded = H.read_index(_local_fs(), parquet_files, max_workers=4, progress=done.append)
    assert threaded.equals(index)
    assert done == parquet_files
    assert H.read_speakers(_local_fs(), parquet_files, max_workers=2) == {"libri:11", "libri:22", "libri:33"}


def test_speaker_capped_selection_respects_budgets_and_chapters(index: pa.Table) -> None:
    selection = H.select_speaker_capped(index, cap_seconds=10.0, enrol_seconds=4.0)
    check_pools_disjoint(
        selection.column("speaker").to_pylist(),
        selection.column("chapter_id").to_pylist(),
        selection.column("pool").to_pylist(),
    )
    rows = _by_speaker(selection)
    assert set(rows) == {"libri:11", "libri:22", "libri:33"}
    for spk in ("libri:11", "libri:22"):
        enrol, target = rows[spk][POOL_ENROL], rows[spk][POOL_TARGET]
        assert sum(e for _, e in enrol) == pytest.approx(4.5) and len({c for c, _ in enrol}) == 1
        assert sum(e for _, e in target) == pytest.approx(7.5)
        assert {c for c, _ in target} == {f"{spk[-2:]}002"}  # one chapter: the fewest row groups
    assert set(rows["libri:33"]) == {POOL_TARGET} and len(rows["libri:33"][POOL_TARGET]) == 4
    spread = _by_speaker(
        H.select_speaker_capped(index, cap_seconds=10.0, enrol_seconds=4.0, strategy="round_robin")
    )
    for spk in ("libri:11", "libri:22"):
        assert {c for c, _ in spread[spk][POOL_TARGET]} == {f"{spk[-2:]}001", f"{spk[-2:]}002"}
    with pytest.raises(ValueError, match="strategy"):
        H.select_speaker_capped(index, strategy="nope")
    everything = H.select_all(index)
    assert everything.num_rows == 28 and {"speaker", "est_seconds"} <= set(everything.column_names)


def test_row_group_strategy_fetches_fewer_row_groups(index: pa.Table) -> None:
    fs = _local_fs()
    local = H.plan_fetch(fs, H.select_speaker_capped(index, cap_seconds=10.0, enrol_seconds=4.0))
    spread = H.plan_fetch(
        fs, H.select_speaker_capped(index, cap_seconds=10.0, enrol_seconds=4.0, strategy="round_robin")
    )
    assert (local.n_row_groups, spread.n_row_groups) == (5, 7)
    assert local.n_rows == 14 and spread.n_rows == 12
    assert local.fetch_bytes < spread.fetch_bytes <= spread.total_bytes
    assert local.summary().startswith("14 rows in 5 row groups of 2 files")
    first = local.files[0]
    assert local.for_file(first).files == (first,)


def test_fetch_decode_and_export(index: pa.Table, tmp_path: Path) -> None:
    fs = _local_fs()
    selection = H.select_speaker_capped(index, cap_seconds=10.0, enrol_seconds=4.0)
    plan = H.plan_fetch(fs, selection)
    rows = list(H.iter_selected_rows(fs, plan))
    assert sorted(r["id"] for r in rows) == sorted(selection.column("id").to_pylist())
    audio, sr = H.decode_audio(rows[0][H.AUDIO_COLUMN])
    assert sr == AUDIO_SR and audio.dtype == np.float32 and audio.ndim == 1
    cap = H.SpeakerCap.from_selection(selection, 10.0, enrol_seconds=4.0)
    with ShardWriter(tmp_path / "out", prefix="c360") as writer:
        stats = H.export_to_shards(fs, plan, selection, writer, split="train.clean.360", cap=cap)
    finalize_dataset(tmp_path / "out", name="clean360cap_test")
    corpus = ShardedCorpus(tmp_path / "out")
    assert stats["rows"] == 14 == len(corpus) and stats["skipped_cap"] == 0
    assert set(corpus.speaker.astype(str)) == {"libri:11", "libri:22", "libri:33"}
    assert set(corpus.column("split")) == {"train.clean.360"} and set(corpus.column("source")) == {"libritts_r"}
    check_pools_disjoint(corpus.speaker, corpus.group, corpus.column("pool"))
    est = dict(zip(selection.column("id").to_pylist(), selection.column("est_seconds").to_pylist(), strict=True))
    for utt, n in zip(corpus.column("utt_id").astype(str), corpus.num_samples, strict=True):
        assert utt.startswith("libri:")
        assert abs(int(n) - round(est[utt.split(":", 1)[1]] * C.SAMPLE_RATE)) <= 2


def test_speaker_cap_refuses_utterances_past_the_budget(index: pa.Table, tmp_path: Path) -> None:
    fs = _local_fs()
    selection = H.select_speaker_capped(index, cap_seconds=10.0, enrol_seconds=4.0)
    tight = H.SpeakerCap.from_selection(selection, 6.0, enrol_seconds=4.0)
    assert tight.budget("libri:11", POOL_TARGET) == 2.0 and tight.budget("libri:33", POOL_TARGET) == 6.0
    with ShardWriter(tmp_path / "tight", prefix="t") as writer:
        stats = H.export_to_shards(fs, H.plan_fetch(fs, selection), selection, writer, cap=tight)
    assert stats["rows"] == 9 and stats["skipped_cap"] == 5
    assert tight.enrol["libri:11"] == pytest.approx(4.5)  # the pool reaches its budget once
    assert tight.target["libri:11"] == pytest.approx(3.0)
    assert tight.target["libri:33"] == pytest.approx(7.5)


def test_export_jobs_split_by_file(index: pa.Table, tmp_path: Path) -> None:
    fs = _local_fs()
    selection = H.select_speaker_capped(index, cap_seconds=10.0, enrol_seconds=4.0)
    plan = H.plan_fetch(fs, selection)
    jobs = H.make_export_jobs(
        plan, selection, str(tmp_path / "jobs"), prefix="c360", split="train.clean.360",
        cap_seconds=10.0, enrol_seconds=4.0, fs_factory=_local_fs,
    )
    assert [j.prefix for j in jobs] == ["c360-f000", "c360-f001"]
    assert sum(H.run_export_job(job)["rows"] for job in jobs) == 14
    finalize_dataset(tmp_path / "jobs", name="jobs")
    assert len(ShardedCorpus(tmp_path / "jobs")) == 14


def test_errors_are_explicit(tmp_path: Path) -> None:
    fs = _local_fs()
    with pytest.raises(ValueError, match="unknown LibriTTS-R split"):
        H.libritts_r_parquet_files(fs, "train.dirty")
    with pytest.raises(FileNotFoundError):
        H.list_parquet_files(fs, str(tmp_path / "none" / "*.parquet"))
    with pytest.raises(ValueError, match="no embedded bytes"):
        H.decode_audio({"bytes": None, "path": "x.wav"})
    audio, sr = H.decode_audio(_wav_bytes(0.5, 200.0))
    assert sr == AUDIO_SR and audio.size == AUDIO_SR // 2
    assert H.LIBRITTS_R_HELDOUT_SPLITS == ("dev.clean", "dev.other", "test.clean", "test.other")
