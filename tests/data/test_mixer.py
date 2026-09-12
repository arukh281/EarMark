"""The training mixer: levels, VAD labels, NULL and target-absent examples, determinism, leaks.

Everything runs on the synthetic corpora of ``conftest.py``. Levels are checked on the
returned components: SNR is the target's active power over the noise's mean power and SIR
the target's active power over the interferer's, both as mixed.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from earmark import constants as C
from earmark.data.agent_voice import split_voices
from earmark.data.embeddings import StubSpeakerEncoder, compute_speaker_embeddings
from earmark.data.labels import active_power, active_power_np, num_frames, vad_labels
from earmark.data.mixer import (
    INTERFERER_AGENT,
    INTERFERER_HUMAN,
    INTERFERER_NONE,
    INTERFERER_TV,
    HeldOut,
    Mixer,
    MixerConfig,
    MixerPools,
    fft_convolve,
    level_gain,
    load_training_pools,
    loudspeaker_response,
    rir_windows,
    segment_peak_frames,
    soft_clip,
    synthetic_rir,
    validate_training_pools,
)
from earmark.data.noise_filter import VB_HELDOUT_ENVIRONMENTS
from earmark.data.shards import POOL_ENROL, POOL_TARGET, ShardedCorpus
from earmark.data.splits import VB_TEST_SPEAKERS, SplitLeakError

from .conftest import Corpora, write_corpus

SR = C.SAMPLE_RATE
B = 8
BATCH_KEYS = {
    "mixture", "target", "vad", "embedding", "null_embedding", "target_present",
    "interferer_kind", "loudspeaker", "reverberant", "clipped", "snr_db", "sir_db",
    "target_level_db", "speaker_index",
}  # fmt: skip
COMPONENT_KEYS = {
    "target_mix", "interferer_mix", "noise_mix", "target_direct", "interferer_reference",
    "output_gain",
}  # fmt: skip


def pools_of(c: Corpora, **overrides: Any) -> MixerPools:
    parts: dict[str, Any] = {
        "speech": c.speech,
        "noise": c.noise,
        "embeddings": c.embeddings,
        "rirs": c.rirs,
        "music": c.music,
        "agent": c.agent,
    }
    parts.update(overrides)
    return MixerPools(**parts)


def make_mixer(c: Corpora, *, seed: int = 0, components: bool = True, **config: Any) -> Mixer:
    settings: dict[str, Any] = {"batch_size": B, "example_seconds": 3.0, **config}
    return Mixer(pools_of(c), MixerConfig(**settings), seed=seed, return_components=components)


def db(x: float | torch.Tensor) -> float:
    return 10.0 * math.log10(float(x))


def assert_same_batch(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True, msg=key)


# --------------------------------------------------------------------------- batch contract


def test_batch_keys_shapes_and_dtypes(corpora: Corpora) -> None:
    mixer = Mixer(pools_of(corpora), MixerConfig(batch_size=4), seed=0)
    batch = mixer.batch(0)
    t, f = 4 * SR, num_frames(4 * SR)
    assert f == 399 and mixer.n_frames == f
    assert set(batch) == BATCH_KEYS
    for key in ("mixture", "target"):
        assert batch[key].shape == (4, t) and batch[key].dtype == torch.float32
    assert batch["vad"].shape == (4, f) and batch["vad"].dtype == torch.float32
    assert set(batch["vad"].unique().tolist()) <= {0.0, 1.0}
    assert batch["embedding"].shape == (4, C.EMBEDDING_DIM) and batch["embedding"].dtype == torch.float32
    for key in ("null_embedding", "target_present", "loudspeaker", "reverberant", "clipped"):
        assert batch[key].shape == (4,) and batch[key].dtype == torch.bool
    for key in ("interferer_kind", "speaker_index"):
        assert batch[key].shape == (4,) and batch[key].dtype == torch.int64
    for key in ("snr_db", "sir_db", "target_level_db"):
        assert batch[key].shape == (4,) and batch[key].dtype == torch.float32
    assert torch.isfinite(batch["mixture"]).all()
    assert float(batch["mixture"].abs().max()) <= 0.99 + 1e-6
    with_components = make_mixer(corpora).batch(0)
    assert set(with_components) == BATCH_KEYS | COMPONENT_KEYS


def test_drawn_values_stay_in_the_configured_ranges(corpora: Corpora) -> None:
    cfg = MixerConfig()
    batch = make_mixer(corpora, seed=3, components=False).batch(0)
    assert ((batch["snr_db"] >= cfg.snr_db[0]) & (batch["snr_db"] <= cfg.snr_db[1])).all()
    level = batch["target_level_db"]
    assert ((level >= cfg.target_level_db[0]) & (level <= cfg.target_level_db[1])).all()
    has = batch["interferer_kind"] > 0
    sir = batch["sir_db"]
    assert torch.isnan(sir[~has]).all()
    assert ((sir[has] >= cfg.sir_db[0]) & (sir[has] <= cfg.sir_db[1])).all()


# --------------------------------------------------------------------------- determinism


def test_fixed_seed_gives_fixed_batch(corpora: Corpora) -> None:
    first = make_mixer(corpora, seed=11).batch(3)
    assert_same_batch(first, make_mixer(corpora, seed=11).batch(3))
    assert not torch.equal(first["mixture"], make_mixer(corpora, seed=12).batch(3)["mixture"])
    assert not torch.equal(first["mixture"], make_mixer(corpora, seed=11).batch(4)["mixture"])


def test_resume_from_state_dict(corpora: Corpora) -> None:
    a = make_mixer(corpora, seed=5)
    a.sample()
    a.sample()
    state = a.state_dict()
    assert state["next_index"] == 2 and state["seed"] == 5
    b = make_mixer(corpora, seed=5)
    b.load_state_dict(state)
    assert_same_batch(a.sample(), b.sample())
    with pytest.raises(ValueError):
        make_mixer(corpora, seed=6).load_state_dict(state)
    with pytest.raises(ValueError):
        make_mixer(corpora, seed=5, snr_db=(0.0, 1.0)).load_state_dict(state)


def test_prefetching_iterator_yields_the_same_batches(corpora: Corpora) -> None:
    a = make_mixer(corpora, seed=9)
    stream = a.iterate(prefetch=2)
    got = [next(stream) for _ in range(3)]
    stream.close()
    assert a.next_index == 3
    reference = make_mixer(corpora, seed=9)
    for k, batch in enumerate(got):
        assert_same_batch(batch, reference.batch(k))


# --------------------------------------------------------------------------- levels


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_snr_and_sir_within_a_tenth_of_a_db(corpora: Corpora, seed: int) -> None:
    """Property: the drawn SNR and SIR hold to 0.1 dB in the mixture the model sees.

    In an unclipped example the mixture must equal the sum of the returned components, and
    the levels are measured on those components with the float64 NumPy active power
    (``active_power_np``), not the torch function the mixer itself uses. A clipped example
    must differ from its component sum; its ``snr_db`` and ``sir_db`` are pre-clipping
    values (see the mixer docstring), so only its components are checked.
    """
    batch = make_mixer(corpora, seed=seed).batch(0)
    for i in range(B):
        parts = [batch[key][i] for key in ("target_mix", "interferer_mix", "noise_mix")]
        summed = parts[0] + parts[1] + parts[2]
        if bool(batch["clipped"][i]):
            assert float((batch["mixture"][i] - summed).abs().max()) > 1e-4
        else:
            torch.testing.assert_close(batch["mixture"][i], summed, atol=1e-6, rtol=0)
        target, interferer, noise = (p.double().numpy() for p in parts)
        gain_db = 20.0 * math.log10(float(batch["output_gain"][i]))
        level = float(batch["target_level_db"][i]) + gain_db
        snr = float(batch["snr_db"][i])
        sir = float(batch["sir_db"][i])
        p_noise = float(np.mean(noise * noise))
        has_interferer = int(batch["interferer_kind"][i]) != INTERFERER_NONE
        if bool(batch["target_present"][i]):
            p_target = float(active_power_np(target))
            assert abs(db(p_target) - level) < 0.1
            assert abs(db(p_target / p_noise) - snr) < 0.1
            if has_interferer:
                assert abs(db(p_target / float(active_power_np(interferer))) - sir) < 0.1
        else:
            assert float(np.abs(target).max()) == 0.0
            assert abs(db(p_noise) - (level - snr)) < 0.1
            if has_interferer:
                assert abs(db(float(active_power_np(interferer))) - (level - sir)) < 0.1


def test_only_clipping_separates_the_mixture_from_its_components(corpora: Corpora) -> None:
    """Both branches of the property above, forced: every example clipped, then none."""
    clipped = make_mixer(corpora, seed=6, p_clip=1.0).batch(0)
    unclipped = make_mixer(corpora, seed=6, p_clip=0.0).batch(0)
    assert bool(clipped["clipped"].all()) and not bool(unclipped["clipped"].any())
    peak_limit = MixerConfig().max_peak + 1e-6
    for batch, clip in ((clipped, True), (unclipped, False)):
        summed = batch["target_mix"] + batch["interferer_mix"] + batch["noise_mix"]
        err = (batch["mixture"] - summed).abs().amax(-1)
        if clip:
            assert bool((err > 1e-4).all())
        else:
            assert float(err.max()) <= 1e-6
        assert bool((batch["mixture"].abs().amax(-1) <= peak_limit).all())


def test_target_absent_examples(corpora: Corpora) -> None:
    batch = make_mixer(
        corpora, seed=4, p_target_absent=1.0, p_null=0.0, p_interferer=1.0
    ).batch(0)
    assert not batch["target_present"].any()
    assert float(batch["target"].abs().max()) == 0.0
    assert float(batch["vad"].sum()) == 0.0
    assert (batch["interferer_kind"] > 0).all()
    assert (batch["mixture"].abs().amax(-1) > 0).all()
    assert not batch["reverberant"].any()
    for i in range(B):
        gain_db = 20.0 * math.log10(float(batch["output_gain"][i]))
        level = float(batch["target_level_db"][i]) + gain_db
        p_noise = batch["noise_mix"][i].double().square().mean()
        assert abs(db(p_noise) - (level - float(batch["snr_db"][i]))) < 0.1
        p_i = active_power(batch["interferer_mix"][i].double())
        assert abs(db(p_i) - (level - float(batch["sir_db"][i]))) < 0.1


# --------------------------------------------------------------------------- NULL embedding


def test_null_embedding_examples_train_all_speech_denoised(corpora: Corpora) -> None:
    common: dict[str, Any] = {"p_target_absent": 0.0, "p_interferer": 1.0}
    plain = make_mixer(corpora, seed=21, p_null=0.0, **common).batch(0)
    null = make_mixer(corpora, seed=21, p_null=1.0, **common).batch(0)
    assert null["null_embedding"].all() and not plain["null_embedding"].any()
    assert float(null["embedding"].abs().max()) == 0.0
    torch.testing.assert_close(plain["embedding"].norm(dim=-1), torch.ones(B), atol=1e-4, rtol=0)
    # The NULL flag changes nothing but the reference, the labels and the embedding.
    torch.testing.assert_close(null["mixture"], plain["mixture"], rtol=0, atol=0)
    torch.testing.assert_close(
        null["target"], plain["target"] + null["interferer_reference"], rtol=0, atol=1e-6
    )
    assert (null["vad"] >= plain["vad"]).all()
    assert float(null["vad"].sum()) > float(plain["vad"].sum())
    # The reference keeps interferer speech but never the noise.
    residual = null["target"] - null["target_direct"]
    assert float((residual * null["noise_mix"]).sum().abs()) < 0.5 * float(null["noise_mix"].square().sum())


def test_null_fraction_and_interferer_mix_follow_the_plan(corpora: Corpora) -> None:
    mixer = make_mixer(corpora, seed=17, components=False)
    records = [r for k in range(50) for r in mixer.describe(k)]
    n = len(records)
    assert n == 50 * B
    null = sum(r["null_embedding"] for r in records) / n
    absent = sum(not r["target_present"] for r in records) / n
    kinds = Counter(r["interferer_kind"] for r in records)
    with_interferer = n - kinds[INTERFERER_NONE]
    assert abs(null - 0.2) < 0.06
    assert abs(absent - 0.15) < 0.06
    assert abs(with_interferer / n - 0.6) < 0.08
    assert abs(kinds[INTERFERER_HUMAN] / with_interferer - 1 / 2) < 0.1
    assert abs(kinds[INTERFERER_AGENT] / with_interferer - 1 / 3) < 0.1
    assert abs(kinds[INTERFERER_TV] / with_interferer - 1 / 6) < 0.08
    others = [r for r in records if r["interferer_kind"] in (INTERFERER_HUMAN, INTERFERER_TV)]
    assert abs(sum(r["loudspeaker"] for r in others) / len(others) - 0.5) < 0.12
    assert all(r["loudspeaker"] for r in records if r["interferer_kind"] == INTERFERER_AGENT)
    assert abs(sum(r["reverberant"] for r in records) / n - 0.5) < 0.08


# --------------------------------------------------------------------------- labels


def test_vad_labels_follow_the_contract_rule_on_dry_targets(corpora: Corpora) -> None:
    """Property: labels sit between the -40 dB rule with the threshold moved +-1 dB.

    Each example holds one whole dry utterance, so the utterance peak is inside it; the
    tolerance only absorbs where the frames fall relative to the stored utterance.
    """
    batch = make_mixer(
        corpora, seed=7, p_reverb=0.0, p_pause=0.0, p_null=0.0, p_target_absent=0.0
    ).batch(0)
    for i in range(B):
        direct = batch["target_direct"][i].double()
        labels = batch["vad"][i].bool()
        strict = vad_labels(direct, threshold_db=C.VAD_THRESHOLD_DB + 1.0)
        loose = vad_labels(direct, threshold_db=C.VAD_THRESHOLD_DB - 1.0)
        assert strict.any()
        assert (strict <= labels).all() and (labels <= loose).all()


def test_vad_labels_track_the_direct_path_on_reverberant_targets(corpora: Corpora) -> None:
    batch = make_mixer(
        corpora, seed=8, p_reverb=1.0, p_pause=0.0, p_null=0.0, p_target_absent=0.0
    ).batch(0)
    assert batch["reverberant"].all()
    for i in range(B):
        direct = batch["target_direct"][i].double()
        labels = batch["vad"][i].bool()
        strict = vad_labels(direct, threshold_db=C.VAD_THRESHOLD_DB + 6.0)
        loose = vad_labels(direct, threshold_db=C.VAD_THRESHOLD_DB - 6.0)
        assert (strict <= labels).all() and (labels <= loose).all()
        # The reverberant tail is louder than the direct path's -40 dB point but unlabelled.
        assert float(batch["target_mix"][i].square().sum()) > float(direct.square().sum())


@pytest.fixture(scope="module")
def bursts(corpora: Corpora, tmp_path_factory: pytest.TempPathFactory) -> Corpora:
    """Speech made of steady tone bursts with abrupt edges.

    A steady burst has the same peak frame energy wherever the frames fall, and its
    activity stops within one frame, so the contract rule gives exact labels to compare.
    """
    rows = []
    for s in range(3):
        for ch in range(2):
            for u in range(2):
                t = np.arange(int((0.8 + 0.2 * u + 0.1 * ch) * SR)) / SR
                pad = np.zeros(int(0.1 * SR))
                burst = 0.2 * np.sin(2 * np.pi * (220.0 + 110.0 * s) * t)
                rows.append(
                    {
                        "audio": np.concatenate([pad, burst, pad]).astype(np.float32),
                        "utt_id": f"libri:{900 + s}_{ch}_{u}",
                        "speaker": f"libri:{900 + s}",
                        "group": str(ch),
                    }
                )
    root = tmp_path_factory.mktemp("bursts")
    speech = write_corpus(root / "speech", rows, name="bursts", pools=0.5)
    emb = compute_speaker_embeddings(speech, StubSpeakerEncoder(), per_speaker=2, seed=0, clip_seconds=(1.0, 1.5))
    return Corpora(root, speech, corpora.noise, corpora.rirs, corpora.music, corpora.agent, emb)


def test_vad_labels_and_hangover_are_exact_on_steady_bursts(bursts: Corpora) -> None:
    config = MixerConfig(
        batch_size=B, example_seconds=3.0, p_reverb=0.0, p_pause=0.0, p_null=0.0, p_target_absent=0.0
    )
    batch = Mixer(pools_of(bursts), config, seed=7, return_components=True).batch(0)
    for i in range(B):
        direct = batch["target_direct"][i].double()
        labels = batch["vad"][i].bool()
        assert int((labels != vad_labels(direct)).sum()) <= 1
        raw = vad_labels(direct, hangover_frames=0)
        last_raw = int(torch.nonzero(raw)[-1])
        last = int(torch.nonzero(labels)[-1])
        assert abs(last - (last_raw + C.VAD_HANGOVER_FRAMES)) <= 1
        first_raw = int(torch.nonzero(raw)[0])
        assert abs(int(torch.nonzero(labels)[0]) - first_raw) <= 1  # no hangover before onsets


# --------------------------------------------------------------------------- provenance


def test_targets_never_come_from_enrolment_chapters(corpora: Corpora) -> None:
    speech = corpora.speech
    ids = speech.column("utt_id").astype(str)
    row_of = {u: i for i, u in enumerate(ids)}
    pool = speech.column("pool").astype(str)
    group = speech.group.astype(str)
    spk = speech.speaker.astype(str)
    enrol_groups = {(s, g) for s, g, p in zip(spk, group, pool, strict=True) if p == POOL_ENROL}
    mixer = make_mixer(corpora, seed=13, p_target_absent=0.0, p_interferer=1.0)
    kinds = set()
    for k in range(20):
        records = mixer.describe(k)
        batch_speakers = [corpora.embeddings.speakers[i] for i in mixer.batch(k)["speaker_index"].tolist()] if k < 2 else None
        for j, rec in enumerate(records):
            if batch_speakers is not None:
                assert batch_speakers[j] == rec["target_speaker"]
            assert rec["target_utts"]
            for utt in rec["target_utts"]:
                r = row_of[utt]
                assert pool[r] == POOL_TARGET
                assert spk[r] == rec["target_speaker"]
                assert (spk[r], group[r]) not in enrol_groups
            kinds.add(rec["interferer_kind"])
            if rec["interferer_kind"] in (INTERFERER_HUMAN, INTERFERER_TV):
                assert rec["interferer_speaker"] != rec["target_speaker"]
            if rec["interferer_kind"] == INTERFERER_AGENT:
                assert rec["interferer_speaker"].startswith("kokoro:")
    assert kinds == {INTERFERER_HUMAN, INTERFERER_AGENT, INTERFERER_TV}


# --------------------------------------------------------------------------- leak checks


def tiny(root: Path, name: str, *, utt_id: str = "x:1", speaker: str = "x:1", **meta: Any) -> ShardedCorpus:
    rng = np.random.default_rng(1)
    row = {
        "audio": (rng.standard_normal(SR) * 0.05).astype(np.float32),
        "utt_id": utt_id,
        "speaker": speaker,
        "group": "g",
        **meta,
    }
    return write_corpus(root / name, [row], name=name)


def test_default_training_pools_pass(corpora: Corpora) -> None:
    validate_training_pools(pools_of(corpora))


def test_held_out_defaults() -> None:
    held = HeldOut()
    assert set(VB_TEST_SPEAKERS) <= held.speakers
    assert held.demand_environments == VB_HELDOUT_ENVIRONMENTS
    assert held.agent_voices == {v for v, s in split_voices().items() if s == "test"}
    more = held.with_speakers(["libri:1089"])
    assert "libri:1089" in more.speakers and "libri:1089" not in held.speakers


def test_held_out_speakers_are_rejected(corpora: Corpora) -> None:
    with pytest.raises(SplitLeakError, match="held-out speakers in speech"):
        validate_training_pools(pools_of(corpora), HeldOut().with_speakers({"libri:101"}))
    with pytest.raises(SplitLeakError, match="held-out speakers"):
        Mixer(pools_of(corpora), MixerConfig(batch_size=2), held_out=HeldOut(speakers=frozenset({"vctk:p301"})))


@pytest.mark.parametrize(
    ("role", "meta", "message"),
    [
        ("noise", {"environment": "TBUS", "source": "demand", "split": "train"}, "held-out DEMAND"),
        ("noise", {"source": "esc50"}, "blocked noise sources"),
        ("noise", {"source": "rirs_noises", "description": "idling engine"}, "excluded-class"),
        ("noise", {"source": "demand", "split": "heldout"}, "non-training noise"),
        ("rirs", {"kind": "real"}, "non-simulated RIRs"),
        ("music", {"split": "heldout"}, "held-out music"),
        ("agent", {"split": "test"}, "held-out agent"),
    ],
)
def test_held_out_material_is_rejected(
    corpora: Corpora, tmp_path: Path, role: str, meta: dict[str, Any], message: str
) -> None:
    extra = ShardedCorpus.concat([getattr(corpora, role), tiny(tmp_path, role, **meta)])
    with pytest.raises(SplitLeakError, match=message):
        validate_training_pools(pools_of(corpora, **{role: extra}))


def test_test_agent_voices_are_rejected(corpora: Corpora, tmp_path: Path) -> None:
    voice = sorted(HeldOut().agent_voices)[0]
    bad = tiny(
        tmp_path, "agent", utt_id=f"kokoro:{voice}:00000", speaker=f"kokoro:{voice}",
        voice_id=voice, split="train",
    )
    extra = ShardedCorpus.concat([corpora.agent, bad])
    with pytest.raises(SplitLeakError, match="test agent voices"):
        validate_training_pools(pools_of(corpora, agent=extra))


# --------------------------------------------------------------------------- DSP helpers


def test_fft_convolve_matches_numpy(rng: np.random.Generator) -> None:
    x = rng.standard_normal(1000)
    h = rng.standard_normal(64)
    y = fft_convolve(torch.from_numpy(x), torch.from_numpy(h)).numpy()
    np.testing.assert_allclose(y, np.convolve(x, h)[:1000], atol=1e-9)
    with pytest.raises(ValueError):
        fft_convolve(torch.zeros(10), torch.zeros(10), n_fft=8)


def test_level_gain_hits_the_level_and_ignores_silence() -> None:
    power = torch.tensor([0.01, 0.0], dtype=torch.float64)
    gain = level_gain(power, torch.tensor([-20.0, -20.0], dtype=torch.float64))
    assert abs(db(power[0] * gain[0] ** 2) + 20.0) < 1e-9
    assert float(gain[1]) == 0.0


def test_soft_clip_has_unit_small_signal_gain_and_compresses_peaks() -> None:
    x = torch.linspace(-0.5, 0.5, 1001, dtype=torch.float64)[None]
    y = soft_clip(x, torch.tensor([3.0], dtype=torch.float64))
    slope = (y[0, 501] - y[0, 499]) / (x[0, 501] - x[0, 499])
    assert abs(float(slope) - 1.0) < 1e-3
    assert float(y.abs().max()) < float(x.abs().max())


def test_loudspeaker_response_band_limits() -> None:
    freqs = torch.fft.rfftfreq(1024, d=1.0 / SR).double()
    resp = loudspeaker_response(
        freqs,
        torch.tensor([300.0], dtype=torch.float64),
        torch.tensor([5000.0], dtype=torch.float64),
        torch.tensor([[1000.0, 2000.0, 3000.0]], dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
        torch.ones(1, 3, dtype=torch.float64),
    )[0]
    assert float(resp[0]) == 0.0
    for hz in (100.0, 1500.0, 7900.0):
        k = int(round(hz * 1024 / SR))
        f = k * SR / 1024
        butterworth = (1 + (300.0 / f) ** 4) ** -0.5 * (1 + (f / 5000.0) ** 8) ** -0.5
        assert abs(float(resp[k]) - butterworth) < 1e-9  # 2nd-order high-pass, 4th-order low-pass
    assert float(resp[round(1500 * 1024 / SR)]) > 0.95
    assert float(resp[round(100 * 1024 / SR)]) < 0.15 and float(resp[round(7900 * 1024 / SR)]) < 0.2


def test_synthetic_rir_and_rir_windows(rng: np.random.Generator) -> None:
    h = synthetic_rir(rng, 0.3, 4000)
    assert h[0] == 1.0 and int(np.argmax(np.abs(h))) == 0
    drr = 10.0 * math.log10(1.0 / float(np.sum(h[1:].astype(np.float64) ** 2)))
    assert -0.01 <= drr <= 10.01
    ht = torch.from_numpy(h)[None]
    direct, early = rir_windows(ht, torch.tensor([0]), direct_samples=40, early_samples=800)
    assert not direct[0, 41:].any() and not early[0, 801:].any()
    assert torch.equal(early[0, :801], ht[0, :801]) and torch.equal(direct[0, :41], ht[0, :41])


def test_segment_peak_frames_marks_only_overlapping_frames() -> None:
    out = segment_peak_frames([(2.0, 1600, 3200), (5.0, 8000, 1600)], 80)
    assert (out[9:30] == 2.0).all() and (out[49:60] == 5.0).all()
    assert np.isinf(out[:9]).all() and np.isinf(out[30:49]).all() and np.isinf(out[60:]).all()
    shifted = segment_peak_frames([(2.0, 1600, 3200)], 80, delay=160)
    assert (shifted[10:31] == 2.0).all() and np.isinf(shifted[9])


# --------------------------------------------------------------------------- loading


def test_load_training_pools_keeps_only_training_rows(corpora: Corpora, tmp_path: Path) -> None:
    voice = sorted(HeldOut().agent_voices)[0]
    test_voice = tiny(
        tmp_path, "agent_test", utt_id=f"kokoro:{voice}:00001", speaker=f"kokoro:{voice}",
        voice_id=voice, split="test",
    )
    held_music = tiny(tmp_path, "music_held", utt_id="musan:held", speaker="musan:someone", split="heldout")
    emb_path = corpora.embeddings.save(tmp_path / "emb.npz")
    common: dict[str, Any] = {
        "speech": corpora.speech.roots[0],
        "noise": [corpora.noise.roots[0]],
        "embeddings": emb_path,
        "rirs": corpora.rirs.roots[0],
        "music": [corpora.music.roots[0], held_music.roots[0]],
        "agent": [corpora.agent.roots[0], test_voice.roots[0]],
    }
    pools = load_training_pools(**common)
    assert pools.agent is not None and set(pools.agent.column("split")) == {"train"}
    assert pools.music is not None and set(pools.music.column("split")) == {"train"}
    validate_training_pools(pools)
    batch = Mixer(pools, MixerConfig(batch_size=2, example_seconds=1.0), seed=0).batch(0)
    assert batch["mixture"].shape == (2, SR)
    with pytest.raises(SplitLeakError):
        validate_training_pools(load_training_pools(**common, agent_splits=None))


# --------------------------------------------------------------------------- devices


def _accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return None


@pytest.mark.slow
@pytest.mark.skipif(_accelerator() is None, reason="no CUDA or MPS device")
def test_mixer_matches_the_cpu_on_an_accelerator(corpora: Corpora) -> None:
    """Opt-in (``-m slow``): the same seed renders the same batch on an accelerator."""
    cpu = make_mixer(corpora, seed=31).batch(0)
    config = MixerConfig(batch_size=B, example_seconds=3.0)
    accel = Mixer(pools_of(corpora), config, seed=31, device=_accelerator(), return_components=True).batch(0)
    assert accel["mixture"].device.type == _accelerator()
    for key in ("mixture", "target", "embedding", "noise_mix", "interferer_mix"):
        torch.testing.assert_close(accel[key].cpu(), cpu[key], atol=1e-4, rtol=1e-3, msg=key)
    assert float((accel["vad"].cpu() != cpu["vad"]).float().mean()) < 0.01
    for key in ("null_embedding", "target_present", "interferer_kind", "speaker_index", "clipped"):
        assert torch.equal(accel[key].cpu(), cpu[key]), key
