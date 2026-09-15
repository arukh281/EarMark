"""Deterministic synthetic signals for the evaluation tests (no datasets, no downloads)."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

SR = 16000


def speech_like(rng: np.random.Generator, seconds: float = 3.0, sr: int = SR, pauses: bool = True) -> np.ndarray:
    """A voiced, syllable-modulated harmonic signal with two formants and optional pauses.

    Close enough to speech for PESQ's level alignment and VAD, and fully deterministic
    for a seeded generator.
    """
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 120.0 + 25.0 * np.sin(2 * np.pi * 0.6 * t + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = np.zeros(n)
    for k in range(1, 40):
        x += np.sin(k * phase + rng.uniform(0, 2 * np.pi)) / k
    for fc, bw in ((700.0, 130.0), (1800.0, 200.0)):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * fc / sr
        x = lfilter([1.0 - r], [1.0, -2 * r * np.cos(theta), r * r], x)
    env = np.clip(np.sin(2 * np.pi * 3.2 * t), 0.0, None) ** 0.6
    x = x * (0.2 + env)
    if pauses:
        for start in (0.9, 2.1):
            x[int(start * sr) : int((start + 0.35) * sr)] = 0.0
    x += 1e-4 * rng.standard_normal(n)
    return 0.1 * x / np.sqrt(np.mean(x**2))


def add_noise(clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise at an exact SNR relative to ``clean``."""
    noise = rng.standard_normal(clean.size)
    scale = np.sqrt(np.sum(clean**2) / (np.sum(noise**2) * 10 ** (snr_db / 10)))
    return clean + scale * noise
