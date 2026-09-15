"""int16 shards, manifests, resampling and the enrolment/target pool split."""

from __future__ import annotations

import json
import math

import numpy as np
import pyarrow.parquet as pq
import pytest

from earmark import constants as C
from earmark.data.shards import (
    MANIFEST_NAME,
    POOL_ENROL,
    POOL_SPARE,
    POOL_TARGET,
    PoolLeakError,
    ShardedCorpus,
    ShardWriter,
    assign_pools,
    check_pools_disjoint,
    dataset_info,
    finalize_dataset,
    from_int16,
    peak_frame_energy,
    resample_to_contract,
    to_int16,
    to_mono,
    trim_silence,
)

from .conftest import Corpora

SR = C.SAMPLE_RATE


def test_int16_quantisation_and_clipping() -> None:
    x = np.array([0.0, 0.5, -0.5, 1.2, -1.5, 32767 / 32768], dtype=np.float32)
    pcm, clipped = to_int16(x)
    assert pcm.dtype == np.dtype("<i2") and clipped == 2
    np.testing.assert_array_equal(pcm, [0, 16384, -16384, 32767, -32768, 32767])
    np.testing.assert_allclose(from_int16(pcm)[:3], x[:3])


def test_to_mono_scales_integer_channels() -> None:
    stereo = np.array([[16384, -16384], [32767, 32767]], dtype=np.int16)
    np.testing.assert_allclose(to_mono(stereo), [0.0, 32767 / 32768], atol=1e-6)
    np.testing.assert_array_equal(to_mono(stereo, 0), [16384, 32767])


def test_resampling_keeps_frequency_and_level() -> None:
    sr_in = 24000
    t = np.arange(2 * sr_in) / sr_in
    y = resample_to_contract(0.5 * np.sin(2 * np.pi * 1000.0 * t), sr_in)
    assert y.dtype == np.float32 and abs(y.size - 2 * SR) <= 1
    core = y[2000:-2000]
    assert abs(20 * math.log10(np.sqrt(np.mean(core**2)) / (0.5 / math.sqrt(2)))) < 0.05
    spec = np.abs(np.fft.rfft(core * np.hanning(core.size)))
    assert abs(np.argmax(spec) * SR / core.size - 1000.0) < 2.0


def test_trim_silence_keeps_speech_with_margin() -> None:
    x = np.zeros(3 * SR, dtype=np.float32)
    x[SR : 2 * SR] = 0.3 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
    start, end = trim_silence(x, margin_s=0.1)
    assert SR - int(0.1 * SR) - C.WINDOW_LENGTH <= start <= SR - int(0.1 * SR) + C.HOP_LENGTH
    assert 2 * SR + int(0.1 * SR) - C.HOP_LENGTH <= end <= 2 * SR + int(0.1 * SR) + C.WINDOW_LENGTH
    assert trim_silence(np.zeros(SR)) == (0, 0)


def test_writer_and_reader_round_trip(tmp_path, rng: np.random.Generator) -> None:
    root = tmp_path / "ds"
    items = []
    with ShardWriter(root, prefix="a", max_shard_bytes=40_000) as w:
        for i in range(5):
            sr = 24000 if i % 2 else SR
            x = (rng.standard_normal(int(0.6 * sr)) * 0.1).astype(np.float32)
            meta = {"text": f"utt {i}"} if i != 3 else {"pool": POOL_TARGET}
            w.add(x, sr, utt_id=f"u{i}", speaker=f"s{i % 2}", group=f"g{i}", **meta)
            items.append((f"u{i}", to_int16(resample_to_contract(x, sr))[0]))
    assert w.partial_manifest_path.exists()
    finalize_dataset(root, name="unit", info={"source": "synthetic"})
    assert not list(root.glob("manifest-*.parquet"))
    corpus = ShardedCorpus(root)
    assert len(corpus) == 5
    assert len(set(corpus.column("shard"))) > 1  # rotation happened
    ids = corpus.column("utt_id").astype(str).tolist()
    assert ids == sorted(ids, key=lambda u: (f"s{int(u[1]) % 2}", f"g{u[1]}", u))
    by_id = dict(items)
    for row, uid in enumerate(ids):
        np.testing.assert_array_equal(corpus.audio_int16(row), by_id[uid])
        stored = corpus.audio(row)
        assert math.isclose(corpus.peak_energy[row], peak_frame_energy(stored), rel_tol=1e-6)
        np.testing.assert_array_equal(corpus.audio_int16(row, 100, 50), by_id[uid][100:150])
    texts = dict(zip(ids, corpus.column("text"), strict=True))
    assert texts["u3"] is None and texts["u0"] == "utt 0"
    info = dataset_info(root)
    assert info["num_rows"] == 5 and info["sample_rate"] == SR and info["contract_hash"] == C.CONTRACT_HASH
    assert info["source"] == "synthetic"
    assert info["total_bytes"] == 2 * sum(int(v.size) for v in by_id.values())


def test_prefix_clash_and_duplicates_fail(tmp_path) -> None:
    with ShardWriter(tmp_path, prefix="p") as w:
        w.add(np.ones(400, np.float32) * 0.1, SR, utt_id="x", speaker="s", group="g")
    with pytest.raises(FileExistsError), ShardWriter(tmp_path, prefix="p") as w2:
        w2.add(np.ones(400, np.float32) * 0.1, SR, utt_id="y", speaker="s", group="g")
    with ShardWriter(tmp_path, prefix="q") as w3:
        w3.add(np.ones(400, np.float32) * 0.1, SR, utt_id="x", speaker="s", group="g")
    with pytest.raises(ValueError, match="duplicate"):
        finalize_dataset(tmp_path, name="dupes")
    with pytest.raises(ValueError, match="reserved"), ShardWriter(tmp_path / "r", prefix="r") as w4:
        w4.add(np.ones(400, np.float32), SR, utt_id="z", speaker="s", group="g", offset=3)


def _random_manifest(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    speakers, groups, seconds, ids = [], [], [], []
    for s in range(int(rng.integers(1, 12))):
        for g in range(int(rng.integers(1, 6))):
            for u in range(int(rng.integers(1, 9))):
                speakers.append(f"spk{s}")
                groups.append(f"ch{g}")
                seconds.append(float(rng.uniform(0.5, 20.0)))
                ids.append(f"spk{s}_ch{g}_{u:03d}")
    return np.array(speakers), np.array(groups), np.array(seconds), np.array(ids)


@pytest.mark.parametrize("seed", range(25))
def test_pools_are_disjoint_by_group(seed: int) -> None:
    """Property: no chapter/session ever feeds both enrolment and targets."""
    rng = np.random.default_rng(seed)
    spk, grp, sec, ids = _random_manifest(rng)
    enrol_s = float(rng.uniform(5.0, 60.0))
    perm = rng.permutation(len(spk))
    pools = assign_pools(spk[perm], grp[perm], sec[perm], order=ids[perm], enrol_seconds=enrol_s)
    check_pools_disjoint(spk[perm], grp[perm], pools)
    # Invariant to input order (the order key decides, not the row order).
    again = assign_pools(spk, grp, sec, order=ids, enrol_seconds=enrol_s)
    np.testing.assert_array_equal(pools, again[perm])
    for s in np.unique(spk):
        mine = spk[perm] == s
        g, p, t = grp[perm][mine], pools[mine], sec[perm][mine]
        enrol_groups = set(g[p == POOL_ENROL])
        target_groups = set(g[p == POOL_TARGET])
        assert not enrol_groups & target_groups
        assert set(g[p == POOL_SPARE]) <= enrol_groups
        if len(set(g)) < 2:
            assert (p == POOL_TARGET).all()
            continue
        assert enrol_groups and target_groups
        totals = {x: t[g == x].sum() for x in set(g)}
        if any(v >= enrol_s for v in totals.values()):
            assert len(enrol_groups) == 1
            assert enrol_s <= t[p == POOL_ENROL].sum() < enrol_s + t.max()
        else:
            for x in enrol_groups:  # whole groups when none is long enough
                assert (p[g == x] == POOL_ENROL).all()


def test_pool_leak_is_detected() -> None:
    with pytest.raises(PoolLeakError):
        check_pools_disjoint(["a", "a"], ["c1", "c1"], [POOL_ENROL, POOL_TARGET])
    with pytest.raises(ValueError, match="unknown pool"):
        check_pools_disjoint(["a"], ["c1"], ["train"])


def test_add_pools_on_prepared_corpus(corpora: Corpora) -> None:
    speech = corpora.speech
    pools = speech.column("pool").astype(str)
    check_pools_disjoint(speech.speaker, speech.group, pools)
    single = speech.speaker.astype(str) == "libri:199"
    assert (pools[single] == POOL_TARGET).all()
    vctk = speech.speaker.astype(str) == "vctk:p301"
    assert {POOL_ENROL, POOL_TARGET} <= set(pools[vctk])
    table = pq.read_table(corpora.root / "speech" / MANIFEST_NAME)
    assert "pool" in table.column_names
    info = json.loads((corpora.root / "speech" / "dataset_info.json").read_text())
    assert info["pool_enrol_seconds"] == 2.0


def test_subset_where_and_concat(corpora: Corpora) -> None:
    speech = corpora.speech
    one = speech.where("speaker", {"libri:101"})
    assert set(one.speaker.astype(str)) == {"libri:101"}
    np.testing.assert_array_equal(one.audio_int16(0), speech.audio_int16(int(np.flatnonzero(speech.speaker == "libri:101")[0])))
    both = ShardedCorpus.concat([one, corpora.noise])
    assert len(both) == len(one) + len(corpora.noise)
    np.testing.assert_array_equal(both.audio_int16(len(one)), corpora.noise.audio_int16(0))
    assert both.column("environment")[0] is None  # missing columns become null
