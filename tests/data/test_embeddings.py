"""The frozen speaker-encoder interface, the stub encoder, enrolment clips and the embedding table."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from earmark import constants as C
from earmark.data.embeddings import (
    KALDI_FBANK,
    WESPEAKER_ONNX,
    WESPEAKER_ONNX_BYTES,
    WESPEAKER_ONNX_SHA256,
    EnrolAugment,
    EnrolAugmentConfig,
    SpeakerEmbeddings,
    SpeakerEncoder,
    StubSpeakerEncoder,
    compute_speaker_embeddings,
    enrolment_clip,
    l2_normalise,
    opus_available,
    opus_roundtrip,
    random_eq,
)
from earmark.data.shards import POOL_ENROL

from .conftest import Corpora, synth_voice, write_corpus

SR = C.SAMPLE_RATE


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum() / (a.norm() * b.norm()))


def unit_table(rng: np.random.Generator, speakers: int, k: int = 8) -> np.ndarray:
    t = rng.standard_normal((speakers, k, C.EMBEDDING_DIM)).astype(np.float32)
    return t / np.linalg.norm(t, axis=-1, keepdims=True)


def test_stub_encoder_contract(rng: np.random.Generator) -> None:
    enc = StubSpeakerEncoder()
    assert isinstance(enc, SpeakerEncoder)
    assert enc.embedding_dim == C.EMBEDDING_DIM
    x = torch.from_numpy(synth_voice(rng, 150.0, 2.0))
    e = enc.embed(x)
    assert e.shape == (1, C.EMBEDDING_DIM) and e.dtype == torch.float32
    torch.testing.assert_close(e.norm(dim=-1), torch.ones(1))
    torch.testing.assert_close(StubSpeakerEncoder().embed(x[None]), e)
    assert cosine(e[0], enc.embed(0.2 * x)[0]) > 0.999
    assert torch.isfinite(enc.embed(torch.full((100,), 0.01))).all()


def test_stub_encoder_separates_voices(rng: np.random.Generator) -> None:
    enc = StubSpeakerEncoder()
    a1 = synth_voice(rng, 110.0, 2.0, peak_hz=600.0)
    a2 = synth_voice(rng, 110.0, 2.0, peak_hz=600.0)
    b = synth_voice(rng, 240.0, 2.0, peak_hz=1800.0)
    e = enc.embed(torch.from_numpy(np.stack([a1, a2, b])))
    assert cosine(e[0], e[1]) > cosine(e[0], e[2]) + 0.05


def test_l2_normalise() -> None:
    torch.testing.assert_close(l2_normalise(torch.tensor([[3.0, 4.0]])), torch.tensor([[0.6, 0.8]]))
    assert torch.equal(l2_normalise(torch.zeros(1, 3)), torch.zeros(1, 3))


def test_wespeaker_pins() -> None:
    assert WESPEAKER_ONNX.endswith(".onnx")
    assert len(WESPEAKER_ONNX_SHA256) == 64
    int(WESPEAKER_ONNX_SHA256, 16)
    assert WESPEAKER_ONNX_BYTES == 26_530_309
    assert KALDI_FBANK["num_mel_bins"] == 80 and KALDI_FBANK["dither"] == 0.0
    assert KALDI_FBANK["sample_frequency"] == float(SR)


def test_embedding_table_round_trip_and_validation(tmp_path: Path, rng: np.random.Generator) -> None:
    table = SpeakerEmbeddings(("libri:1", "libri:2", "vctk:p225"), unit_table(rng, 3), {"encoder": "x"})
    assert table.per_speaker == 8 and table.dim == C.EMBEDDING_DIM
    assert table.index()["libri:2"] == 1
    back = SpeakerEmbeddings.load(table.save(tmp_path / "sub" / "emb.npz"))
    assert back.speakers == table.speakers and back.meta == {"encoder": "x"}
    np.testing.assert_array_equal(back.table, table.table)
    with pytest.raises(ValueError, match="L2"):
        SpeakerEmbeddings(("a:1",), np.ones((1, 2, 4), np.float32))
    with pytest.raises(ValueError, match="duplicate"):
        SpeakerEmbeddings(("a:1", "a:1"), unit_table(rng, 2))
    with pytest.raises(ValueError, match="shape"):
        SpeakerEmbeddings(("a:1",), unit_table(rng, 2))
    bad = unit_table(rng, 1)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        SpeakerEmbeddings(("a:1",), bad)
    merged = SpeakerEmbeddings.merge([table, SpeakerEmbeddings(("libri:9",), unit_table(rng, 1))])
    assert merged.speakers[-1] == "libri:9" and merged.table.shape == (4, 8, C.EMBEDDING_DIM)
    with pytest.raises(ValueError):
        SpeakerEmbeddings.merge([table, table])


def test_speaker_embeddings_cover_enrolled_speakers(corpora: Corpora) -> None:
    speech, emb = corpora.speech, corpora.embeddings
    pools = speech.column("pool").astype(str)
    enrolled = {s for s, p in zip(speech.speaker.astype(str), pools, strict=True) if p == POOL_ENROL}
    assert set(emb.speakers) == enrolled
    assert "libri:199" not in emb.speakers  # one chapter only: an interferer, never enrolled
    assert emb.table.shape == (len(enrolled), 8, C.EMBEDDING_DIM)
    np.testing.assert_allclose(np.linalg.norm(emb.table, axis=-1), 1.0, atol=1e-5)
    assert not np.allclose(emb.table[0, 0], emb.table[0, 1])
    assert emb.meta["encoder"] == StubSpeakerEncoder.name
    assert emb.meta["contract_hash"] == C.CONTRACT_HASH


def test_embeddings_are_seeded_per_speaker(corpora: Corpora) -> None:
    emb = corpora.embeddings
    kwargs = {"per_speaker": 8, "clip_seconds": (1.0, 2.0), "speakers": ["libri:103"]}
    one = compute_speaker_embeddings(corpora.speech, StubSpeakerEncoder(), seed=0, **kwargs)
    assert one.speakers == ("libri:103",)
    np.testing.assert_array_equal(one.table[0], emb.table[emb.index()["libri:103"]])
    other = compute_speaker_embeddings(corpora.speech, StubSpeakerEncoder(), seed=1, **kwargs)
    assert not np.array_equal(other.table[0], one.table[0])


def test_embeddings_need_enrolment_pools(tmp_path: Path, rng: np.random.Generator) -> None:
    row = {"audio": synth_voice(rng, 150.0, 1.0), "utt_id": "libri:1_1_1", "speaker": "libri:1", "group": "1"}
    corpus = write_corpus(tmp_path / "nopool", [row], name="nopool")
    with pytest.raises(ValueError, match="pool"):
        compute_speaker_embeddings(corpus, StubSpeakerEncoder())


def test_enrolment_clip_length(corpora: Corpora, rng: np.random.Generator) -> None:
    speech = corpora.speech
    pools = speech.column("pool").astype(str)
    rows = np.flatnonzero((speech.speaker.astype(str) == "libri:101") & (pools == POOL_ENROL))
    assert rows.size
    clip = enrolment_clip(speech, rows, rng, seconds=(5.0, 10.0))
    assert clip.dtype == np.float32 and 5 * SR <= clip.size <= 10 * SR
    assert float(np.abs(clip).max()) > 0.01
    with pytest.raises(ValueError):
        enrolment_clip(speech, [], rng)


def test_enrol_augment_keeps_length_and_records_what_it_did(corpora: Corpora, rng: np.random.Generator) -> None:
    calls: list[int] = []

    def fake_opus(x: np.ndarray) -> np.ndarray:
        calls.append(x.size)
        return x * np.float32(0.9)

    config = EnrolAugmentConfig(p_noise=1.0, p_reverb=1.0, p_eq=1.0, p_opus=1.0)
    augment = EnrolAugment(config, noise=corpora.noise, rirs=corpora.rirs, opus=fake_opus)
    clip = synth_voice(rng, 150.0, 3.0)
    out, applied = augment(clip, rng)
    assert out.shape == clip.shape and out.dtype == np.float32 and np.isfinite(out).all()
    assert float(np.abs(out).max()) <= 0.99 + 1e-6
    assert {"rir", "eq", "noise", "snr_db", "opus"} <= set(applied)
    assert calls == [clip.size]
    assert config.snr_db[0] <= applied["snr_db"] <= config.snr_db[1]
    plain, nothing = EnrolAugment(EnrolAugmentConfig(p_noise=0, p_reverb=0, p_eq=0, p_opus=0))(clip, rng)
    assert nothing == {} and plain.shape == clip.shape


def test_random_eq_is_identity_at_zero_gain(rng: np.random.Generator) -> None:
    x = rng.standard_normal(4000).astype(np.float32)
    np.testing.assert_allclose(random_eq(x, rng, max_db=0.0), x, atol=1e-5)
    shaped = random_eq(x, rng, max_db=6.0)
    assert shaped.shape == x.shape and not np.allclose(shaped, x)


@pytest.mark.skipif(not opus_available(), reason="ffmpeg with libopus is not available")
def test_opus_roundtrip_keeps_length_and_level(rng: np.random.Generator) -> None:
    x = synth_voice(rng, 150.0, 2.0)
    y = opus_roundtrip(x, bitrate=16000)
    assert y.shape == x.shape and y.dtype == np.float32
    assert abs(10.0 * math.log10(float(np.mean(y**2)) / float(np.mean(x**2)))) < 3.0
