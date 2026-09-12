"""PyTorch-stream dev runner on a tiny generated Earmark-Synth set (random weights, CPU only).

Nothing is downloaded: speakers are seeded harmonic complexes written through the real shard
writer, the manifest comes from :func:`earmark.data.synth_bench.design_suite`, and the model
is a randomly initialised S-GRU. A delay-line stand-in model pins the latency alignment.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from earmark import constants as C
from earmark.data import synth_bench
from earmark.data.embeddings import CodecUnavailable, StubSpeakerEncoder
from earmark.data.labels import num_frames
from earmark.data.shards import ShardedCorpus, ShardWriter, add_pools, finalize_dataset
from earmark.data.synth_bench import (
    COND_CLEAN,
    COND_GRID,
    COND_INTERFERER_ONLY,
    COND_LONG_ABSENCE,
    COND_OPUS,
    COND_SIR0,
    BenchPools,
    MixtureSpec,
    SuiteDesign,
    design_suite,
    render_mixture,
    write_manifest,
)
from earmark.eval import dev_runner as dr
from earmark.eval import metrics as M
from earmark.eval.bargein import binarize, pool_counts, score_bargein
from earmark.eval.runs_log import DEFAULT_RUNS_PATH, read_runs, verify_log
from earmark.eval.stream_score import EvalItem, ItemAudio, score_items
from earmark.model import OUTPUT_DELAY_SAMPLES, EarmarkNet

SR = C.SAMPLE_RATE
HOP = C.HOP_LENGTH
# A stand-in training split: non-empty (an empty set would make the leak guard vacuous) and
# disjoint from the libri:90x bench voices, so design_suite's guard runs without firing.
TRAIN_SPEAKERS = {"libri:5000"}
SECONDS = 3.2
N_MIXTURES = 6
FRACTIONS = {
    COND_GRID: 0.3,
    COND_CLEAN: 0.15,
    COND_INTERFERER_ONLY: 0.2,
    COND_SIR0: 0.2,
    COND_LONG_ABSENCE: 0.15,
}


# --------------------------------------------------------------------------------------------
# Tiny synthetic Earmark-Synth set


def _voice(rng: np.random.Generator, f0: float, seconds: float, peak_hz: float) -> np.ndarray:
    """A voiced, syllable-modulated harmonic complex (float32, peak 0.1-0.3)."""
    t = np.arange(int(seconds * SR)) / SR
    freq = f0 * (1.0 + 0.03 * np.sin(2 * np.pi * rng.uniform(2.0, 4.0) * t))
    phase = 2 * np.pi * np.cumsum(freq) / SR
    x = np.zeros_like(t)
    for k in range(1, int(6000 // f0)):
        x += np.exp(-0.5 * ((k * f0 - peak_hz) / peak_hz) ** 2) / k * np.sin(k * phase)
    syllables = np.sin(2 * np.pi * rng.uniform(3.0, 5.0) * t + rng.uniform(0, 2 * np.pi))
    x = x * np.clip(syllables, 0.0, None) ** 0.7 * np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.05)
    return (x / (np.abs(x).max() + 1e-12) * rng.uniform(0.1, 0.3)).astype(np.float32)


def _write(
    root: Path, rows: list[dict[str, Any]], name: str, *, pools: float | None = None
) -> ShardedCorpus:
    with ShardWriter(root, prefix="t") as writer:
        for row in rows:
            fields = dict(row)
            writer.add(fields.pop("audio"), SR, **fields)
    finalize_dataset(root, name=name)
    if pools is not None:
        add_pools(root, enrol_seconds=pools)
    return ShardedCorpus(root)


@dataclasses.dataclass
class Bench:
    root: Path
    pools: BenchPools
    manifest: Path
    specs: list[MixtureSpec]


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> Bench:
    """Four two-chapter speakers, one noise clip and one RIR; a 6-mixture dev manifest."""
    root = tmp_path_factory.mktemp("earmark_synth")
    rng = np.random.default_rng(11)
    speech = []
    for s, f0 in enumerate((117.0, 143.0, 171.0, 203.0)):
        for chapter in range(2):
            for u in range(3):
                speech.append(
                    {
                        "audio": _voice(rng, f0, rng.uniform(1.0, 1.8), 600.0 + 200.0 * s),
                        "utt_id": f"libri:{900 + s}_{chapter}_{u}",
                        "speaker": f"libri:{900 + s}",
                        "group": str(chapter),
                    }
                )
    _write(root / "speech", speech, "speech_dev", pools=2.0)
    noise = {
        "audio": (rng.standard_normal(3 * SR) * 0.05).astype(np.float32),
        "utt_id": "demand:DKITCHEN:000",
        "speaker": "demand:DKITCHEN",
        "group": "DKITCHEN",
        "source": "demand",
    }
    _write(root / "noise", [noise], "noise_dev")
    n = np.arange(3000)
    h = rng.standard_normal(n.size) * np.exp(-6.9 * n / (0.3 * SR)) * 0.2
    h[:12], h[12] = 0.0, 1.0
    rir = {
        "audio": (h / np.abs(h).max() * 0.9).astype(np.float32),
        "utt_id": "rirs:sim:room/0",
        "speaker": "rirs:room",
        "group": "room",
    }
    _write(root / "rirs", [rir], "rirs_dev")
    dev = root / "dev"
    dev.mkdir()
    pools_file = dev / dr.POOLS_FILE_NAME
    folders = {
        "targets": "../speech",
        "interferers": "../speech",
        "noise": "../noise",
        "rirs": "../rirs",
    }
    pools_file.write_text(yaml.safe_dump(folders))
    pools = dr.load_pools_file(pools_file)
    design = SuiteDesign(n_mixtures=N_MIXTURES, seconds=SECONDS, fractions=FRACTIONS)
    specs = design_suite(pools, split="dev", design=design, seed=0, train_speakers=TRAIN_SPEAKERS)
    manifest = write_manifest(specs, dev / "manifest.parquet", train_speakers=TRAIN_SPEAKERS)
    return Bench(root, pools, manifest, specs)


@pytest.fixture(scope="module")
def net() -> EarmarkNet:
    return dr.load_model("S-GRU", seed=0)[0]


@pytest.fixture(scope="module")
def embedder(bench: Bench) -> dr.EnrolmentEmbedder:
    return dr.EnrolmentEmbedder(bench.pools, StubSpeakerEncoder())


@pytest.fixture(scope="module")
def personal_run(bench: Bench, net: EarmarkNet, embedder: dr.EnrolmentEmbedder) -> dr.DevRun:
    return dr.run_dev(
        net, bench.specs, bench.pools, mode="personal", embed=embedder, batch=4, chunk_hops=7
    )


class DelayLine:
    """Stand-in model: the output is the input delayed by ``1 + extra_hops`` hops; VAD is flat."""

    def __init__(self, extra_hops: int = 0, vad: float = 0.5) -> None:
        self.delay = (1 + extra_hops) * HOP
        self.vad = vad

    def condition(self, emb: torch.Tensor | None = None, *, batch: int | None = None) -> None:
        return None

    def init_state(
        self,
        batch: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(batch, self.delay, device=device, dtype=dtype)

    def step(
        self, frame: torch.Tensor, emb: object, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        buf = torch.cat([state, frame], dim=-1)
        return buf[:, :HOP], torch.full((frame.shape[0],), self.vad), buf[:, HOP:]


@dataclasses.dataclass
class _Opaque:
    """Not a tensor type, so a weights_only load refuses a checkpoint holding it."""

    value: int = 1


def _max_list_len(obj: Any) -> int:
    if isinstance(obj, dict):
        return max((_max_list_len(v) for v in obj.values()), default=0)
    if isinstance(obj, list):
        return max([len(obj), *(_max_list_len(v) for v in obj)])
    return 0


# --------------------------------------------------------------------------------------------
# VAD AUC


def test_roc_auc_is_the_pairwise_probability() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.3
    s = np.round(rng.random(500) + 0.5 * y, 1)  # coarse rounding: many ties
    diff = s[y][:, None] - s[~y][None, :]
    pairwise = float(np.mean((diff > 0) + 0.5 * (diff == 0)))
    assert dr.roc_auc(s, y) == pytest.approx(pairwise, abs=1e-12)
    assert dr.roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert dr.roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert dr.roc_auc(np.full(10, 0.3), np.arange(10) % 2 == 0) == 0.5
    assert math.isnan(dr.roc_auc([0.1, 0.2], [1, 1]))
    with pytest.raises(ValueError):
        dr.roc_auc([0.1, np.nan], [0, 1])


def test_vad_auc_summary_pools_frames_and_bootstraps_speakers() -> None:
    rng = np.random.default_rng(1)
    scores, labels, clusters = [], [], []
    for i in range(12):
        y = rng.random(300) < 0.4
        scores.append(np.clip(0.5 * y + rng.normal(0.25, 0.2, 300), 0.0, 1.0))
        labels.append(y)
        clusters.append(f"spk{i % 4}")
    out = dr.vad_auc_summary(scores, labels, clusters, n_resamples=300, seed=0)
    exact = dr.roc_auc(np.concatenate(scores), np.concatenate(labels))
    assert out["auc"] == exact and out["n_units"] == 4 and out["unit"] == "cluster"
    assert out["n_frames"] == 3600
    low, high = out["ci95"]
    assert low <= exact <= high and high - low < 0.1
    coarse = dr.vad_auc_summary(scores, labels, clusters, n_resamples=300, seed=0, max_bins=64)
    assert coarse["ci95"][0] == pytest.approx(low, abs=0.01)
    silent = dr.vad_auc_summary([np.zeros(5)], [np.zeros(5, dtype=bool)], ["a"])
    assert math.isnan(silent["auc"]) and silent["n_active_frames"] == 0


# --------------------------------------------------------------------------------------------
# Streaming, alignment and scoring


def test_delay_line_output_is_scored_sample_aligned(bench: Bench) -> None:
    run = dr.run_dev(DelayLine(), bench.specs, bench.pools, mode="denoise", batch=4, chunk_hops=7)
    assert len(run.items) == N_MIXTURES and not run.skipped and not run.embeddings
    checked = 0
    for it in run.items:
        r = render_mixture(it.spec, bench.pools)
        assert it.vad.shape == it.labels.shape == (num_frames(r["mixture"].size),)
        assert np.all(it.vad == 0.5)
        np.testing.assert_array_equal(it.labels, r["vad"])
        if it.spec.condition == COND_CLEAN:  # output == reference
            assert math.isnan(it.row["si_sdri"]) and it.row["si_sdr"] > 60.0
        elif it.spec.target_present:  # output == mixture, sample-aligned
            assert it.row["si_sdri"] == pytest.approx(0.0, abs=1e-9)
            expected = M.si_sdr(r["reference"], r["mixture"])
            assert it.row["si_sdr"] == pytest.approx(expected, abs=1e-6)
            checked += 1
        else:
            assert math.isnan(it.row["si_sdr"]) and it.row["vad_active_frames"] == 0
    assert checked >= 3
    # One hop too late must show up as a large SI-SDR loss, so the check above is not vacuous.
    late = dr.run_dev(DelayLine(extra_hops=1), bench.specs, bench.pools, mode="denoise", batch=4)
    shifted = [it.row["si_sdri"] for it in late.items if math.isfinite(it.row["si_sdri"])]
    assert shifted and float(np.mean(shifted)) < -1.0


def test_streaming_scores_match_the_offline_forward(
    bench: Bench, net: EarmarkNet, embedder: dr.EnrolmentEmbedder, personal_run: dr.DevRun
) -> None:
    assert personal_run.mode == "personal" and len(personal_run.items) == N_MIXTURES
    assert set(personal_run.embeddings) == {s.mixture_id for s in bench.specs}
    for it in personal_run.items:
        r = render_mixture(it.spec, bench.pools)
        n = r["mixture"].size
        x = torch.zeros(1, (-(-n // HOP) + 1) * HOP)  # plus the flush hop
        x[0, :n] = torch.from_numpy(r["mixture"])
        with torch.no_grad():
            out = net(x, torch.from_numpy(embedder(it.spec))[None])
        est = out.wav[0, OUTPUT_DELAY_SAMPLES : OUTPUT_DELAY_SAMPLES + n].numpy()
        np.testing.assert_allclose(it.vad, out.vad[0, 1 : 1 + num_frames(n)].numpy(), atol=1e-5)
        if math.isfinite(it.row["si_sdr"]):
            assert it.row["si_sdr"] == pytest.approx(M.si_sdr(r["reference"], est), abs=1e-3)
        if math.isfinite(it.row["si_sdri"]):
            expected = M.si_sdr_improvement(r["reference"], est, r["mixture"])
            assert it.row["si_sdri"] == pytest.approx(expected, abs=1e-3)
        if math.isfinite(it.row["interferer_suppression_db"]):
            region = M.interferer_region(r["reference"], r["interferer"])
            expected = M.interferer_suppression_db(r["mixture"], est, region)
            assert it.row["interferer_suppression_db"] == pytest.approx(expected, abs=1e-3)


def test_batching_padding_and_chunking_do_not_change_scores(
    bench: Bench, net: EarmarkNet, embedder: dr.EnrolmentEmbedder
) -> None:
    design = SuiteDesign(n_mixtures=3, seconds=3.005, fractions=FRACTIONS)  # not whole hops
    short = design_suite(bench.pools, split="dev", design=design, seed=1, train_speakers=TRAIN_SPEAKERS)
    renamed = (dataclasses.replace(s, mixture_id=f"dev-short-{i}") for i, s in enumerate(short))
    specs = [*bench.specs[:3], *renamed]
    assert any(round(s.seconds * SR) % HOP for s in specs)
    one = dr.run_dev(net, specs, bench.pools, embed=embedder, batch=1, chunk_hops=1000)
    many = dr.run_dev(net, specs, bench.pools, embed=embedder, batch=4, chunk_hops=3)
    for a, b in zip(one.items, many.items, strict=True):
        assert a.spec.mixture_id == b.spec.mixture_id
        np.testing.assert_allclose(a.vad, b.vad, atol=1e-5)
        for key in ("si_sdr", "si_sdri", "interferer_suppression_db"):
            assert math.isfinite(a.row[key]) == math.isfinite(b.row[key])
            if math.isfinite(a.row[key]):
                assert a.row[key] == pytest.approx(b.row[key], abs=1e-3)


def test_matched_threshold_and_pooled_bargeins(bench: Bench, personal_run: dr.DevRun) -> None:
    items = personal_run.items
    thr = dr.choose_threshold(items, target_recall=0.95)
    assert thr["source"] == "matched-on-this-run" and thr["achieved_recall"] >= 0.95
    summary = dr.score_run(personal_run, n_resamples=100)
    got = summary["bargein"]
    assert got["threshold"]["threshold"] == thr["threshold"]
    expected = pool_counts(
        score_bargein(binarize(it.vad, thr["threshold"]), it.labels) for it in items
    )
    assert got["false_barge_ins"] == expected.false_barge_ins
    assert (got["hits"], got["reference_onsets"]) == (expected.hits, expected.reference_onsets)
    assert got["silent_minutes"] == pytest.approx(expected.silent_minutes)
    assert got["frame_recall"] == pytest.approx(expected.frame_recall)
    assert got["frame_recall"] >= 0.95
    pooled_auc = dr.roc_auc(
        np.concatenate([it.vad for it in items]), np.concatenate([it.labels for it in items])
    )
    assert summary["vad"]["auc"] == pooled_auc
    finite = [it.row["si_sdri"] for it in items if math.isfinite(it.row["si_sdri"])]
    assert summary["metrics"]["si_sdri"]["n"] == len(finite)
    assert summary["metrics"]["si_sdri"]["mean"] == pytest.approx(float(np.mean(finite)))
    assert set(summary["by_condition"]) == {s.condition for s in bench.specs}
    assert summary["do_no_harm"]["n"] == sum(s.condition == COND_CLEAN for s in bench.specs)

    fixed = dr.score_run(personal_run, vad_threshold=0.5, n_resamples=100)
    assert fixed["bargein"]["threshold"] == {"threshold": 0.5, "source": "fixed", "level": "fixed"}
    with pytest.raises(ValueError):
        dr.choose_threshold(items, fixed=1.5)
    with pytest.raises(ValueError, match="n_resamples"):
        dr.summarize_run(personal_run, threshold=None, n_resamples=0)

    absent = [s for s in bench.specs if not s.target_present]
    absent_run = dr.run_dev(DelayLine(), absent, bench.pools, mode="denoise")
    empty = dr.score_run(absent_run, n_resamples=50)
    assert empty["bargein"] is None and "no barge-in scores" in empty["bargein_note"]
    assert empty["acceptance"]["passed"] is False


def test_codec_mixtures_are_skipped_never_scored_uncoded(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    design = SuiteDesign(n_mixtures=2, seconds=SECONDS, fractions={COND_OPUS: 1.0})
    specs = design_suite(bench.pools, split="dev", design=design, seed=3, train_speakers=TRAIN_SPEAKERS)
    assert all(s.codec == "opus16k" for s in specs)

    def unavailable(x: np.ndarray, **kwargs: Any) -> np.ndarray:
        raise CodecUnavailable("no libopus in this test")

    monkeypatch.setattr(synth_bench, "opus_roundtrip", unavailable)
    run = dr.run_dev(DelayLine(), specs, bench.pools, mode="denoise")
    assert not run.items and [s["item_id"] for s in run.skipped] == [s.mixture_id for s in specs]
    assert all("unavailable" in s["reason"] for s in run.skipped)
    identity = lambda x, **kwargs: np.asarray(x, dtype=np.float32)  # noqa: E731
    monkeypatch.setattr(synth_bench, "opus_roundtrip", identity)
    run = dr.run_dev(DelayLine(), specs, bench.pools, mode="denoise")
    assert len(run.items) == 2 and not run.skipped


def test_enhancer_adapter_matches_the_runner(
    bench: Bench, net: EarmarkNet, embedder: dr.EnrolmentEmbedder, personal_run: dr.DevRun
) -> None:
    system = dr.EarmarkStreamSystem(net, mode="personal", chunk_hops=11)
    items = []
    for s in bench.specs:
        r = render_mixture(s, bench.pools)
        audio = ItemAudio(
            mixture=r["mixture"],
            reference=r["reference"] if s.target_present else None,
            embedding=embedder(s),
            reference_vad=r["vad"],
        )
        item = EvalItem(item_id=s.mixture_id, cluster=s.target_speaker, load=lambda a=audio: a)
        items.append(item)
    rows = score_items(items, system, metrics=("si_sdri",), vad_threshold=0.5)
    by_id = {it.spec.mixture_id: it for it in personal_run.items}
    compared = 0
    for row in rows:
        ref = by_id[row["item_id"]]
        if math.isfinite(ref.row["si_sdri"]):
            assert row["si_sdri"] == pytest.approx(ref.row["si_sdri"], abs=1e-3)
            compared += 1
        assert "bargein_false_barge_ins" in row
    assert compared >= 3
    first = bench.specs[0]
    out = system.enhance(render_mixture(first, bench.pools)["mixture"], embedder(first))
    n = int(round(SECONDS * SR))
    assert out.audio.shape == (n,) and out.vad is not None and out.vad.shape == (num_frames(n),)
    np.testing.assert_allclose(out.vad, by_id[first.mixture_id].vad, atol=1e-5)
    with pytest.raises(ValueError, match="embedding"):
        system.enhance(np.zeros(1600, dtype=np.float32))
    with pytest.raises(ValueError):
        dr.EarmarkStreamSystem(net, mode="gate")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# Loading and validation


def test_checkpoint_loading(net: EarmarkNet, tmp_path: Path) -> None:
    state = net.state_dict()
    plain = tmp_path / "plain.pt"
    torch.save(state, plain)
    wrapped = tmp_path / "wrapped.pt"
    prefixed = {f"module.{k}": v for k, v in state.items()}
    torch.save({"model": prefixed, "step": 7, "config": "s_gru"}, wrapped)
    for path in (plain, wrapped):
        loaded, info = dr.load_model("S-GRU", path, seed=123)
        assert info["random_weights"] is False
        assert info["checkpoint_sha256"] == dr.file_sha256(path)
        for key, value in loaded.state_dict().items():
            torch.testing.assert_close(value, state[key])
    opaque = tmp_path / "opaque.pt"
    torch.save({"model": state, "extra": _Opaque()}, opaque)
    with pytest.raises(ValueError, match="trust-checkpoint"):
        dr.load_model("S-GRU", opaque)
    trusted, _ = dr.load_model("S-GRU", opaque, trust_checkpoint=True)
    torch.testing.assert_close(trusted.vad_head.weight, state["vad_head.weight"])
    with pytest.raises(ValueError, match="does not fit"):
        dr.load_model("M", plain)
    mislabelled = tmp_path / "m.pt"
    torch.save({"model": state, "config": "M"}, mislabelled)
    with pytest.raises(ValueError, match="saved for config"):
        dr.load_model("S-GRU", mislabelled)
    with pytest.raises(ValueError, match="state_dict"):
        dr.extract_state_dict({"step": 3})
    a, info_a = dr.load_model("S-GRU", seed=5)
    b, _ = dr.load_model("S-GRU", seed=5)
    torch.testing.assert_close(a.vad_head.weight, b.vad_head.weight)
    assert info_a["random_weights"] is True and info_a["init_seed"] == 5


def test_manifest_and_pools_are_validated(bench: Bench, tmp_path: Path) -> None:
    table = pq.read_table(bench.manifest)
    column = table.schema.get_field_index("contract_hash")
    stale = table.set_column(column, "contract_hash", pa.array(["0" * 16] * table.num_rows))
    pq.write_table(stale, tmp_path / "stale.parquet")
    with pytest.raises(ValueError, match="contract"):
        dr.load_manifest(tmp_path / "stale.parquet")
    pq.write_table(table.drop_columns(["contract_hash"]), tmp_path / "plain.parquet")
    with pytest.raises(ValueError, match="no contract_hash"):
        dr.load_manifest(tmp_path / "plain.parquet")
    loaded = dr.load_manifest(bench.manifest)
    assert [s.to_json() for s in loaded] == [s.to_json() for s in bench.specs]
    with pytest.raises(ValueError, match="duplicate"):
        dr.run_dev(DelayLine(), [bench.specs[0], bench.specs[0]], bench.pools, mode="denoise")
    with pytest.raises(ValueError, match="embedding source"):
        dr.run_dev(DelayLine(), bench.specs, bench.pools, mode="personal")

    dr.check_pools_cover(bench.specs, bench.pools)
    noise = bench.pools.noise
    with pytest.raises(ValueError, match="not in the pools"):
        dr.check_pools_cover(bench.specs, BenchPools(targets=noise, interferers=noise, noise=noise))

    n_grid = sum(s.condition == COND_GRID for s in bench.specs)
    assert n_grid >= 1
    grid = dr.select_specs(bench.specs, conditions=[COND_GRID])
    assert [s.condition for s in grid] == [COND_GRID] * n_grid
    assert len(dr.select_specs(bench.specs, limit=2)) == 2
    with pytest.raises(ValueError, match="unknown conditions"):
        dr.select_specs(bench.specs, conditions=["nope"])
    with pytest.raises(ValueError, match="no mixtures"):
        dr.select_specs(bench.specs, conditions=["agent_tts"])
    bad = tmp_path / "pools.yaml"
    bad.write_text("targets: x\nsurprise: y\n")
    with pytest.raises(ValueError, match="unknown keys"):
        dr.load_pools_file(bad)


def test_precomputed_embeddings_round_trip(
    bench: Bench, embedder: dr.EnrolmentEmbedder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vectors = {s.mixture_id: embedder(s) for s in bench.specs}
    assert all(abs(float(np.linalg.norm(v)) - 1.0) < 1e-5 for v in vectors.values())
    assert embedder.clip(bench.specs[0]).size > SR
    path = dr.save_enrolment_embeddings(
        tmp_path / "emb.npz", bench.specs, vectors, encoder=embedder.name
    )
    table = dr.load_enrolment_embeddings(path)
    assert len(table) == len(bench.specs) and table.name == f"precomputed:{embedder.name}"
    table.check(bench.specs)
    for s in bench.specs:
        np.testing.assert_array_equal(table(s), vectors[s.mixture_id])
    spec = bench.specs[0]
    changed = dataclasses.replace(spec, enrol_utts=(*spec.enrol_utts, "libri:900_9_9"))
    with pytest.raises(ValueError, match="different enrolment"):
        table(changed)
    with pytest.raises(ValueError, match="different enrolment"):
        table.check([changed])
    with pytest.raises(KeyError):
        table(dataclasses.replace(spec, mixture_id="dev-unknown"))

    partial = dr.save_enrolment_embeddings(
        tmp_path / "partial.npz", bench.specs[:2], vectors, encoder=embedder.name
    )
    with pytest.raises(ValueError, match="no embedding for"):
        dr.load_enrolment_embeddings(partial).check(bench.specs)
    out = tmp_path / "never.json"
    argv = [
        "--config", "S-GRU", "--manifest", str(bench.manifest), "--embeddings", str(partial),
        "--split", "smoke", "--out", str(out), "--no-log",
    ]  # fmt: skip
    with pytest.raises(SystemExit) as exc:
        dr.main(argv)
    assert exc.value.code == 2 and "no embedding for" in capsys.readouterr().err
    assert not out.exists()


# --------------------------------------------------------------------------------------------
# CLI


def test_cli_end_to_end_writes_numbers_only_and_logs_pytorch_stream(
    bench: Bench, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    res = tmp_path / "res"
    out, rows, log, emb = res / "dev.json", res / "rows.jsonl", res / "runs.jsonl", res / "emb.npz"
    argv = [
        "--config", "S-GRU", "--manifest", str(bench.manifest), "--encoder", "stub",
        "--split", "smoke", "--batch", "4", "--chunk-hops", "9", "--resamples", "100",
        "--out", str(out), "--rows-out", str(rows), "--runs-log", str(log),
        "--embeddings-out", str(emb), "--notes", "unit test",
    ]  # fmt: skip
    assert dr.main(argv) == 0
    summary = json.loads(out.read_text())
    assert summary["inference_path"] == summary["system"]["inference_path"] == "pytorch-stream"
    assert summary["system"]["name"] == "earmark-S-GRU-personal"
    assert summary["system"]["params"] > 0 and summary["system"]["mmac_per_s"] > 0
    assert summary["model"]["random_weights"] is True and summary["model"]["checkpoint"] is None
    assert summary["model"]["state_bytes_per_stream"] > 0
    assert summary["n_items"] + summary["n_skipped"] == N_MIXTURES
    assert summary["manifest"]["sha256"] == dr.file_sha256(bench.manifest)
    assert summary["pools_file"] == str(bench.manifest.parent / dr.POOLS_FILE_NAME)
    assert 0.0 <= summary["vad"]["auc"] <= 1.0
    assert summary["metrics"]["si_sdri"]["n"] >= 1
    assert summary["metrics"]["si_sdri"]["unit"] == "cluster"
    assert summary["bargein"]["threshold"]["source"] == "matched-on-this-run"
    assert set(summary["acceptance"]) == {"si_sdri_improves", "vad_auc", "passed"}
    # Numbers only: nothing audio-sized in any output, and no audio files next to them.
    assert _max_list_len(summary) < 1000
    lines = rows.read_text().splitlines()
    assert len(lines) == summary["n_items"]
    assert all(_max_list_len(json.loads(x)) < 1000 for x in lines)
    assert sorted(p.suffix for p in res.iterdir()) == [".json", ".jsonl", ".jsonl", ".npz"]
    assert verify_log(log) == 1
    (record,) = read_runs(log)
    assert record["inference_path"] == "pytorch-stream" and record["split"] == "smoke"
    assert record["system"] == "earmark-S-GRU-personal" and record["suite"] == dr.SUITE
    assert record["config"]["results_json"] == str(out) and record["notes"] == "unit test"
    assert record["metrics"]["vad_auc"] == pytest.approx(summary["vad"]["auc"])
    assert "dev_runner:" in capsys.readouterr().err

    # Precomputed embeddings reproduce the run; --check exits by the acceptance block.
    again_path = tmp_path / "again.json"
    rc = dr.main([
        "--config", "S-GRU", "--manifest", str(bench.manifest), "--embeddings", str(emb),
        "--split", "smoke", "--batch", "3", "--resamples", "100", "--out", str(again_path),
        "--no-log", "--check",
    ])  # fmt: skip
    again = json.loads(again_path.read_text())
    assert rc == (0 if again["acceptance"]["passed"] else 1)
    assert again["embeddings"].startswith("precomputed:")
    si_sdri = summary["metrics"]["si_sdri"]["mean"]
    assert again["metrics"]["si_sdri"]["mean"] == pytest.approx(si_sdri, abs=1e-3)
    assert again["vad"]["auc"] == pytest.approx(summary["vad"]["auc"], abs=1e-4)
    assert verify_log(log) == 1


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--encoder", "stub"], "random weights"),
        (["--encoder", "stub", "--split", "test"], "--vad-threshold"),
        (["--checkpoint", "CKPT", "--encoder", "stub"], "stub encoder"),
        (["--checkpoint", "CKPT", "--mode", "denoise", "--split", "test", "--vad-threshold", "0.5"],
         "manifest"),
        (["--encoder", "stub", "--split", "smoke", "--vad-threshold", "1.5"], "probability"),
        (["--encoder", "stub", "--split", "smoke", "--resamples", "0"], "--resamples"),
        (["--split", "smoke", "--mode", "denoise", "--embeddings-out", "EMB"], "personal mode"),
    ],
)
def test_cli_refuses_runs_that_would_be_mislabelled(
    bench: Bench,
    net: EarmarkNet,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    ckpt = tmp_path / "s_gru.pt"
    torch.save(net.state_dict(), ckpt)
    out, log = tmp_path / "x.json", tmp_path / "runs.jsonl"
    placeholders = {"CKPT": str(ckpt), "EMB": str(tmp_path / "emb.npz")}
    argv = [
        "--config", "S-GRU", "--manifest", str(bench.manifest), "--out", str(out),
        "--runs-log", str(log), *(placeholders.get(a, a) for a in extra),
    ]  # fmt: skip
    with pytest.raises(SystemExit) as exc:
        dr.main(argv)
    assert exc.value.code == 2 and message in capsys.readouterr().err
    assert not out.exists() and not log.exists()


def test_module_entry_point_prints_help(repo_root: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(repo_root / "python")}
    proc = subprocess.run(
        [sys.executable, "-m", "earmark.eval.dev_runner", "--help"],
        capture_output=True, text=True, env=env, cwd=repo_root, timeout=120, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert "--manifest" in proc.stdout and "--config" in proc.stdout


def test_default_results_path_is_under_results_dev() -> None:
    now = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
    manifest = Path("data/earmark_synth/dev/manifest.parquet")
    path = dr.default_out_path("earmark-M-personal", manifest, now=now)
    assert path.parent == dr.DEFAULT_RESULTS_DIR == DEFAULT_RUNS_PATH.parent / "dev"
    assert path.name == "earmark-M-personal_dev_20260911T120000Z.json"


@pytest.mark.parametrize(
    ("ci_low", "auc", "passed"),
    [
        (0.2, 0.93, True),
        (-0.1, 0.95, False),
        (0.5, 0.89, False),
        (None, 0.95, False),
        (0.5, None, False),
    ],
)
def test_acceptance_needs_si_sdri_ci_above_zero_and_auc_of_0_9(
    ci_low: float | None, auc: float | None, passed: bool
) -> None:
    summary = {"metrics": {"si_sdri": {"mean": 1.0, "ci95": [ci_low, 2.0]}}, "vad": {"auc": auc}}
    assert dr.acceptance(summary)["passed"] is passed
    assert dr.acceptance({})["passed"] is False
