"""Repository-wide pytest configuration for Earmark.

Unit tests are CPU-only, offline and fast. This file keeps them that way by default:
it hides CUDA devices, puts the Hugging Face hub in offline mode, keeps caches inside
``./.cache`` and seeds every random generator before each test. Set the variables
yourself (for example ``HF_HUB_OFFLINE=0 make gates``) to override.
"""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))

TEST_SEED = 1234


@pytest.fixture(autouse=True)
def _seed_everything() -> Iterator[None]:
    """Seed Python, NumPy and (if already imported) torch so every test is repeatable."""
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(TEST_SEED)
    yield


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh, seeded NumPy generator for synthetic fixtures."""
    return np.random.default_rng(TEST_SEED)
