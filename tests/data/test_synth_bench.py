"""Earmark-Synth: seeded sparse design, disjointness, manifest round trip and rendering."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from earmark import constants as C
from earmark.data.embeddings import opus_available
from earmark.data.labels import active_power
from earmark.data.mixer import INTERFERER_AGENT
from earmark.data.shards import POOL_ENROL, POOL_TARGET, ShardedCorpus
from earmark.data.splits import SplitLeakError
from earmark.data.synth_bench import (
    COND_AGENT,
    COND_CLEAN,
    COND_ESC50,
    COND_INTERFERER_ONLY,
    COND_LONG_ABSENCE,
    DEFAULT_FRACTIONS,
    HYPOTHESIS_CONDITIONS,
    BenchPools,
    SuiteDesign,
    allocate_conditions,
    check_manifest_disjoint,
    design_suite,
    load_bench_pools,
    read_manifest,
    render_mixture,
    summarize,
    write_manifest,
)

from .conftest import Corpora, write_corpus

SR = C.SAMPLE_RATE
# A stand-in training split: non-empty (an empty set would make the leak guard vacuous) and
# disjoint from every benchmark fixture speaker, so design_suite's guard runs without firing.
TRAIN_SPEAKERS = {"libri:5000"}


@pytest.fixture(scope="module")
def bench(corpora: Corpora, tmp_path_factory: pytest.TempPathFactory) -> BenchPools:
    rng = np.random.default_rng(3)
    esc_row = {
        "audio": (rng.standard_normal(5 * SR) * 0.05).astype(np.float32),
        "utt_id": "esc50:1-17367-A-10",
        "speaker": "esc50:rain",
        "group": "rain",
        "source": "esc50",
        "category": "rain",
    }
    esc = write_corpus(tmp_path_factory.mktemp("esc") / "esc", [esc_row], name="esc")
    return BenchPools(
        targets=corpora.speech,
        interferers=corpora.speech,
        noise=ShardedCorpus.concat([corpora.noise, esc]),
        rirs=corpora.rirs,
        agent=corpora.agent,
        music=corpora.music,
    )


def design(n: int) -> SuiteDesign:
    return SuiteDesign(n_mixtures=n, seconds=4.0)


def db(x: float | torch.Tensor) -> float:
    return 10.0 * math.log10(float(x))


def test_allocation_and_hypothesis_share() -> None:
    labels = allocate_conditions(80, DEFAULT_FRACTIONS)
    assert len(labels) == 80
    assert sum(labels.count(c) for c in HYPOTHESIS_CONDITIONS) == 32  # 40 %
    assert allocate_conditions(7, {"grid": 1.0}) == ["grid"] * 7
    with pytest.raises(ValueError):
        SuiteDesign(fractions={"nope": 1.0})
    with pytest.raises(ValueError):
        SuiteDesign(n_mixtures=0)


def test_design_is_seeded(bench: BenchPools) -> None:
    a = design_suite(bench, split="dev", design=design(80), seed=0, train_speakers=TRAIN_SPEAKERS)
    b = design_suite(bench, split="dev", design=design(80), seed=0, train_speakers=TRAIN_SPEAKERS)
    assert [s.to_json() for s in a] == [s.to_json() for s in b]
    c = design_suite(bench, split="dev", design=design(80), seed=1, train_speakers=TRAIN_SPEAKERS)
    assert [s.to_json() for s in a] != [s.to_json() for s in c]
    assert Counter(s.condition for s in a) == Counter(allocate_conditions(80, DEFAULT_FRACTIONS))
    info = summarize(a)
    assert info["n"] == 80 and abs(info["hypothesis_fraction"] - 0.4) < 1e-9
    assert len({s.mixture_id for s in a}) == 80 and all(s.mixture_id.startswith("dev-") for s in a)


def test_enrolment_comes_from_another_chapter_and_speakers_stay_disjoint(bench: BenchPools) -> None:
    specs = design_suite(bench, split="test", design=design(60), seed=2, train_speakers=TRAIN_SPEAKERS)
    check_manifest_disjoint(specs, {"libri:5000"})
    speech = bench.targets
    ids = speech.column("utt_id").astype(str)
    group = dict(zip(ids, speech.group.astype(str), strict=True))
    pool = dict(zip(ids, speech.column("pool").astype(str), strict=True))
    speaker = dict(zip(ids, speech.speaker.astype(str), strict=True))
    for s in specs:
        assert s.enrol_utts
        assert all(pool[u] == POOL_ENROL and speaker[u] == s.target_speaker for u in s.enrol_utts)
        enrol_groups = {group[u] for u in s.enrol_utts}
        for seg in s.target_segments:
            assert pool[seg.utt_id] == POOL_TARGET and group[seg.utt_id] not in enrol_groups
        if s.interferer and s.interferer_kind != INTERFERER_AGENT:
            assert s.interferer_speaker != s.target_speaker
        assert s.target_present == (s.condition != COND_INTERFERER_ONLY)
    with pytest.raises(SplitLeakError):
        check_manifest_disjoint(specs, {specs[0].target_speaker})
    with pytest.raises(SplitLeakError):
        design_suite(bench, split="test", design=design(10), seed=2, train_speakers={"libri:101"})


def test_manifest_round_trip(bench: BenchPools, tmp_path: Path) -> None:
    specs = design_suite(bench, split="dev", design=design(24), seed=4, train_speakers=TRAIN_SPEAKERS)
    path = write_manifest(specs, tmp_path / "dev" / "manifest.parquet", train_speakers=TRAIN_SPEAKERS)
    assert [s.to_json() for s in read_manifest(path)] == [s.to_json() for s in specs]
    table = pq.read_table(path)
    assert set(table.column("contract_hash").to_pylist()) == {C.CONTRACT_HASH}
    assert table.column("condition").to_pylist() == [s.condition for s in specs]


def test_rendered_mixtures_match_their_specs(bench: BenchPools) -> None:
    specs = design_suite(bench, split="dev", design=design(40), seed=5, train_speakers=TRAIN_SPEAKERS)
    n_snr = n_sir = 0
    seen: set[str] = set()
    for spec in specs:
        if spec.codec is not None:
            continue
        out = render_mixture(spec, bench, apply_codec=False)
        assert out["mixture"].shape == (int(spec.seconds * SR),) and out["mixture"].dtype == np.float32
        assert float(np.abs(out["mixture"]).max()) <= 0.99 + 1e-6
        mix, itf, noise = (torch.from_numpy(out[k].astype(np.float64)) for k in ("mixture", "interferer", "noise"))
        target = mix - itf - noise
        seen.add(spec.condition)
        if not spec.target_present:
            assert not out["vad"].any() and not out["reference"].any()
            continue
        assert out["vad"].any()
        if spec.snr_db is not None:
            assert abs(db(active_power(target) / noise.square().mean()) - spec.snr_db) < 0.1
            n_snr += 1
        if spec.sir_db is not None and spec.interferer:
            assert abs(db(active_power(target) / active_power(itf)) - spec.sir_db) < 0.1
            n_sir += 1
        if spec.condition == COND_CLEAN:
            assert not noise.any() and not itf.any()
            np.testing.assert_allclose(out["reference"], out["mixture"], atol=1e-6)
        if spec.condition == COND_LONG_ABSENCE:
            assert np.flatnonzero(out["vad"]).max() < 3.2 * C.FRAME_RATE_HZ
    assert n_snr >= 5 and n_sir >= 5
    assert {COND_CLEAN, COND_INTERFERER_ONLY, COND_AGENT, COND_ESC50} <= seen


def test_load_bench_pools_keeps_only_held_out_voices_and_music(corpora: Corpora, tmp_path: Path) -> None:
    rng = np.random.default_rng(4)

    def one(name: str, **meta: str) -> ShardedCorpus:
        row = {
            "audio": (rng.standard_normal(SR) * 0.05).astype(np.float32),
            "utt_id": f"{name}:1",
            "speaker": f"{name}:1",
            "group": "g",
            **meta,
        }
        return write_corpus(tmp_path / name, [row], name=name)

    test_voice = one("kokoro_test", voice_id="af_x", split="test")
    held_music = one("music_held", split="heldout")
    common = {
        "targets": corpora.speech.roots[0],
        "interferers": [corpora.speech.roots[0]],
        "noise": corpora.noise.roots[0],
        "agent": [corpora.agent.roots[0], test_voice.roots[0]],
    }
    pools = load_bench_pools(**common, rirs=corpora.rirs.roots[0], music=[corpora.music.roots[0], held_music.roots[0]])
    assert pools.agent is not None and set(pools.agent.column("split")) == {"test"} and len(pools.agent) == 1
    assert pools.music is not None and set(pools.music.column("split")) == {"heldout"} and len(pools.music) == 1
    assert pools.rirs is not None and len(pools.rirs) == len(corpora.rirs)
    everything = load_bench_pools(**common, agent_splits=None)
    assert everything.agent is not None and len(everything.agent) == len(corpora.agent) + 1
    with pytest.raises(ValueError):
        load_bench_pools(targets=corpora.speech.roots[0], interferers=[], noise=corpora.noise.roots[0])


@pytest.mark.skipif(not opus_available(), reason="ffmpeg with libopus is not available")
def test_opus_condition_is_coded(bench: BenchPools) -> None:
    suite = design_suite(bench, split="dev", design=design(40), seed=5, train_speakers=TRAIN_SPEAKERS)
    specs = [s for s in suite if s.codec == "opus16k"]
    assert specs
    raw = render_mixture(specs[0], bench, apply_codec=False)["mixture"]
    coded = render_mixture(specs[0], bench, apply_codec=True)["mixture"]
    assert coded.shape == raw.shape and not np.allclose(coded, raw)
