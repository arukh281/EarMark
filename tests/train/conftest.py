"""Tiny synthetic corpora, configs and a fake clock for the trainer tests.

Nothing is downloaded. "Speakers" are harmonic complexes under a syllable-rate envelope,
written through the real shard writer so the mixer reads the same format as on Kaggle.
Examples are 1 s long with batch 2, so a training step of S-GRU takes tens of
milliseconds on a CPU.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from earmark import constants as C
from earmark.data.embeddings import StubSpeakerEncoder, compute_speaker_embeddings
from earmark.data.mixer import Mixer, MixerPools
from earmark.data.shards import ShardedCorpus, ShardWriter, add_pools, finalize_dataset
from earmark.train.config import TrainConfig

SR = C.SAMPLE_RATE

#: Mixer overrides for tests: batch 2, 1 s examples, short synthetic RIRs.
TINY_MIXER: dict[str, Any] = {"batch_size": 2, "example_seconds": 1.0, "rir_max_seconds": 0.25}


def voice(rng: np.random.Generator, f0: float, seconds: float, peak_hz: float) -> np.ndarray:
    """A voiced, syllable-modulated harmonic signal (float32, peak about 0.2)."""
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1.0 + 0.03 * np.sin(2 * np.pi * 3.0 * t))) / SR
    x = np.zeros_like(t)
    for k in range(1, int(6000 // f0)):
        x += np.exp(-0.5 * ((k * f0 - peak_hz) / peak_hz) ** 2) / k * np.sin(k * phase)
    env = np.clip(np.sin(2 * np.pi * rng.uniform(3.0, 5.0) * t + rng.uniform(0, 2 * np.pi)), 0.0, None) ** 0.7
    x = x * env
    return (x / (np.abs(x).max() + 1e-12) * 0.2).astype(np.float32)


def _write(root: Path, rows: list[dict[str, Any]], name: str) -> ShardedCorpus:
    with ShardWriter(root, prefix="t", max_shard_bytes=400_000) as writer:
        for row in rows:
            row = dict(row)
            writer.add(row.pop("audio"), SR, **row)
    finalize_dataset(root, name=name)
    return ShardedCorpus(root)


@pytest.fixture(scope="session")
def tiny_pools(tmp_path_factory: pytest.TempPathFactory) -> MixerPools:
    """Four two-chapter speakers, two noises and stub embeddings (built once per session)."""
    root = tmp_path_factory.mktemp("train_corpora")
    rng = np.random.default_rng(11)
    rows: list[dict[str, Any]] = []
    for s in range(4):
        for chapter in range(2):
            for u in range(2):
                rows.append(
                    {
                        "audio": voice(rng, 110.0 + 30.0 * s, float(rng.uniform(1.0, 1.4)), 600.0 + 200.0 * s),
                        "utt_id": f"libri:{201 + s}_{chapter}_{u}",
                        "speaker": f"libri:{201 + s}",
                        "group": str(chapter),
                    }
                )
    _write(root / "speech", rows, "train_test_speech")
    add_pools(root / "speech", enrol_seconds=1.0)
    speech = ShardedCorpus(root / "speech")
    white = rng.standard_normal(2 * SR) * 0.1
    brown = np.cumsum(rng.standard_normal(2 * SR))
    brown = (brown - np.convolve(brown, np.ones(401) / 401, mode="same")) * 0.01
    noise_rows = [
        {"audio": x.astype(np.float32), "utt_id": f"synthetic:noise{j}", "speaker": f"synthetic:noise{j}",
         "group": f"n{j}", "source": "synthetic", "split": "train"}
        for j, x in enumerate((white, brown))
    ]  # fmt: skip
    noise = _write(root / "noise", noise_rows, "train_test_noise")
    embeddings = compute_speaker_embeddings(speech, StubSpeakerEncoder(), per_speaker=2, seed=0, clip_seconds=(0.5, 1.0))
    return MixerPools(speech=speech, noise=noise, embeddings=embeddings)


def tiny_config(**overrides: Any) -> TrainConfig:
    """S-GRU on the tiny mixer, fixed schedule, no AMP, no validation, inline batches."""
    settings: dict[str, Any] = {
        "name": "tiny",
        "model": "S-GRU",
        "mixer": dict(TINY_MIXER),
        "max_steps": 6,
        "warmup_steps": 2,
        "lr": 3e-3,
        "amp": False,
        "time_limit_hours": 10.0,
        "checkpoint_minutes": 1000.0,
        "log_every": 1,
        "val_every": 0,
        "trace_steps": 3,
        "prefetch": 0,
    }
    settings.update(overrides)
    return TrainConfig(**settings)


@pytest.fixture
def mixer_for(tiny_pools: MixerPools) -> Callable[..., Mixer]:
    """Build the mixer a config describes, over the tiny pools."""

    def factory(config: TrainConfig, seed_offset: int = 0) -> Mixer:
        return Mixer(tiny_pools, config.mixer_config(), seed=config.seed + seed_offset)

    return factory


class FakeClock:
    """Returns 0, dt, 2 dt, ... on successive calls (the trainer reads it once per step)."""

    def __init__(self, dt: float) -> None:
        self.dt = dt
        self.now = 0.0

    def __call__(self) -> float:
        value = self.now
        self.now += self.dt
        return value
