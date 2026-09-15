"""Synthetic corpora for the data tests: nothing is downloaded.

"Speakers" are harmonic complexes with a speaker-specific pitch and spectral peak under a
syllable-rate envelope, which is enough for level, label, pool and determinism checks.
Everything is written through the real :class:`ShardWriter`, so the tests exercise the
same shard and manifest format that the notebooks produce.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from earmark import constants as C
from earmark.data.embeddings import SpeakerEmbeddings, StubSpeakerEncoder, compute_speaker_embeddings
from earmark.data.agent_voice import split_voices
from earmark.data.shards import ShardedCorpus, ShardWriter, add_pools, finalize_dataset

SR = C.SAMPLE_RATE
_TRAIN_VOICES = sorted(v for v, s in split_voices().items() if s == "train")
#: One American-female and one British-male training voice (never a held-out test voice).
AGENT_TRAIN_VOICES: tuple[str, str] = (
    next(v for v in _TRAIN_VOICES if v.startswith("af_")),
    next(v for v in _TRAIN_VOICES if v.startswith("bm_")),
)


def synth_voice(rng: np.random.Generator, f0: float, seconds: float, *, peak_hz: float = 900.0) -> np.ndarray:
    """A voiced, syllable-modulated harmonic signal (float32, peak about 0.1-0.3)."""
    t = np.arange(int(seconds * SR)) / SR
    freq = f0 * (1.0 + 0.03 * np.sin(2 * np.pi * rng.uniform(2.0, 4.0) * t))
    phase = 2 * np.pi * np.cumsum(freq) / SR
    x = np.zeros_like(t)
    for k in range(1, int(7000 // f0)):
        x += np.exp(-0.5 * ((k * f0 - peak_hz) / peak_hz) ** 2) / k * np.sin(k * phase)
    rate = rng.uniform(3.0, 5.0)
    env = np.clip(np.sin(2 * np.pi * rate * t + rng.uniform(0, 2 * np.pi)), 0.0, None) ** 0.7
    edge = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.05)
    x = x * env * edge
    return (x / (np.abs(x).max() + 1e-12) * rng.uniform(0.1, 0.3)).astype(np.float32)


def write_corpus(root: Path, rows: Iterable[Mapping[str, Any]], *, name: str, pools: float | None = None) -> ShardedCorpus:
    """Write rows (``audio``, ``utt_id``, ``speaker``, ``group``, extra columns) as a dataset."""
    with ShardWriter(root, prefix="t", max_shard_bytes=400_000) as writer:
        for row in rows:
            row = dict(row)
            audio = row.pop("audio")
            sr = row.pop("sample_rate", SR)
            writer.add(audio, sr, **row)
    finalize_dataset(root, name=name)
    if pools is not None:
        add_pools(root, enrol_seconds=pools)
    return ShardedCorpus(root)


@dataclass
class Corpora:
    root: Path
    speech: ShardedCorpus
    noise: ShardedCorpus
    rirs: ShardedCorpus
    music: ShardedCorpus
    agent: ShardedCorpus
    embeddings: SpeakerEmbeddings


def _speech_rows(rng: np.random.Generator) -> list[dict[str, Any]]:
    rows = []
    for s in range(6):  # six multi-chapter speakers
        f0 = 110.0 + 25.0 * s
        for ch in range(2 + s % 2):
            for u in range(3):
                rows.append(
                    {
                        "audio": synth_voice(rng, f0, rng.uniform(1.0, 2.5), peak_hz=600 + 150 * s),
                        "utt_id": f"libri:{101 + s}_{ch}_{u}",
                        "speaker": f"libri:{101 + s}",
                        "group": f"{ch}",
                        "text": "synthetic",
                    }
                )
    for u in range(3):  # a single-chapter speaker: interferer only
        rows.append(
            {
                "audio": synth_voice(rng, 260.0, 1.5),
                "utt_id": f"libri:199_0_{u}",
                "speaker": "libri:199",
                "group": "0",
            }
        )
    for block in range(2):
        for u in range(3):
            rows.append(
                {
                    "audio": synth_voice(rng, 190.0, rng.uniform(1.0, 2.0), peak_hz=1400),
                    "utt_id": f"vctk:p301_{block * 100 + u:03d}_mic1",
                    "speaker": "vctk:p301",
                    "group": f"block{block:02d}",
                }
            )
    return rows


def make_rir(rng: np.random.Generator, delay: int, rt60: float, taps: int = 4000) -> np.ndarray:
    n = np.arange(taps)
    h = rng.standard_normal(taps) * np.exp(-6.9 * n / (rt60 * SR)) * 0.2
    h[:delay] = 0.0
    h[delay] = 1.0
    return (h / np.abs(h).max() * 0.9).astype(np.float32)


@pytest.fixture(scope="session")
def corpora(tmp_path_factory: pytest.TempPathFactory) -> Corpora:
    """Speech, noise, RIR, music and agent corpora plus stub embeddings (built once)."""
    root = tmp_path_factory.mktemp("corpora")
    rng = np.random.default_rng(7)
    speech = write_corpus(root / "speech", _speech_rows(rng), name="speech_test", pools=2.0)
    noise_rows = []
    white = rng.standard_normal(3 * SR) * 0.1
    brown = np.cumsum(rng.standard_normal(3 * SR))
    brown = brown - np.convolve(brown, np.ones(401) / 401, mode="same")
    for j, (x, env) in enumerate(((white, "DKITCHEN"), (brown / np.abs(brown).max() * 0.1, "NPARK"))):
        noise_rows.append(
            {"audio": x.astype(np.float32), "utt_id": f"demand:{env}:{j:03d}", "speaker": f"demand:{env}",
             "group": env, "environment": env, "source": "demand", "split": "train"}
        )
    noise_rows.append(
        {"audio": (rng.standard_normal(2 * SR) * 0.05).astype(np.float32), "utt_id": "rirs:noise:n1:000",
         "speaker": "rirs:noise:n1", "group": "n1", "environment": None, "source": "rirs_noises", "split": "train"}
    )
    noise = write_corpus(root / "noise", noise_rows, name="noise_test")
    rir_rows = []
    for j, (room, size) in enumerate((("smallroom/Room001", "smallroom"), ("smallroom/Room001", "smallroom"),
                                      ("mediumroom/Room002", "mediumroom"), ("mediumroom/Room002", "mediumroom"))):
        rir_rows.append(
            {"audio": make_rir(rng, 10 + 15 * j, 0.2 + 0.1 * j), "utt_id": f"rirs:sim:{room}/{j}",
             "speaker": f"rirs:{room}", "group": room, "kind": "simulated", "room": room, "room_size": size}
        )
    rirs = write_corpus(root / "rirs", rir_rows, name="rir_test")
    music_rows = []
    for j in range(2):
        t = np.arange(3 * SR) / SR
        chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0 * (j + 1), 277.0 * (j + 1), 330.0 * (j + 1)))
        music_rows.append(
            {"audio": (0.05 * chord).astype(np.float32), "utt_id": f"musan:track{j}", "speaker": f"musan:artist{j}",
             "group": f"track{j}", "split": "train", "genre": "pop", "artist": f"artist{j}"}
        )
    music = write_corpus(root / "music", music_rows, name="music_test")
    agent_rows = []
    for v, voice in enumerate(AGENT_TRAIN_VOICES):
        for u in range(2):
            agent_rows.append(
                {"audio": synth_voice(rng, 170.0 + 60 * v, 1.5, peak_hz=1100), "utt_id": f"kokoro:{voice}:{u:05d}",
                 "speaker": f"kokoro:{voice}", "group": voice, "voice_id": voice, "split": "train"}
            )
    agent = write_corpus(root / "agent", agent_rows, name="agent_test")
    embeddings = compute_speaker_embeddings(
        speech, StubSpeakerEncoder(), per_speaker=8, seed=0, clip_seconds=(1.0, 2.0)
    )
    return Corpora(root, speech, noise, rirs, music, agent, embeddings)
