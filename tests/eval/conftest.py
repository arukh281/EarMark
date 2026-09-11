"""Fixtures for the evaluation tests; the signal generators live in :mod:`tests.eval.synth`."""

from __future__ import annotations

import numpy as np
import pytest

from tests.eval.synth import speech_like


@pytest.fixture
def speech(rng: np.random.Generator) -> np.ndarray:
    """Three seconds of deterministic speech-like audio at 16 kHz."""
    return speech_like(rng)
