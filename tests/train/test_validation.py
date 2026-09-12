"""Validation metrics: the AUC helper and a validation pass on the tiny mixer."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from earmark.data.mixer import Mixer
from earmark.model import build
from earmark.train.losses import TERMS, EarmarkLoss
from earmark.train.validation import VAL_SEED_OFFSET, binary_auc, validate

from .conftest import tiny_config


def test_binary_auc_edge_cases() -> None:
    assert binary_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert binary_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert binary_auc(np.full(4, 0.5), np.array([0, 1, 0, 1])) == 0.5
    assert math.isnan(binary_auc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_binary_auc_matches_pairwise_counting_with_ties() -> None:
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, 200).astype(float)  # many ties
    labels = rng.integers(0, 2, 200)
    pos, neg = scores[labels == 1], scores[labels == 0]
    pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    assert binary_auc(scores, labels) == pytest.approx(pairs)


def test_validation_is_repeatable_and_restores_training_mode(mixer_for: Callable[..., Mixer]) -> None:
    config = tiny_config()
    mixer = mixer_for(config, VAL_SEED_OFFSET)
    net = build("S-GRU")
    first = validate(net, EarmarkLoss(), mixer, 2)
    assert net.training
    assert {"loss", "si_sdr_db", "si_sdri_db", "vad_auc", "vad_auc_null"} | {f"term_{t}" for t in TERMS} <= set(first)
    assert math.isfinite(first["loss"])
    for key in ("vad_auc", "vad_auc_null"):
        assert math.isnan(first[key]) or 0.0 <= first[key] <= 1.0
    second = validate(net, EarmarkLoss(), mixer, 2)
    assert first == pytest.approx(second, nan_ok=True)
