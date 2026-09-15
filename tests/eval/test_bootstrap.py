"""Unit tests for the paired speaker-cluster and utterance bootstraps."""

from __future__ import annotations

import numpy as np
import pytest

from earmark.eval import bootstrap as BS


def test_identical_systems_give_a_degenerate_zero_interval(rng: np.random.Generator) -> None:
    a = rng.standard_normal(200)
    res = BS.paired_bootstrap(a, a.copy())
    assert res.estimate == 0.0 and res.low == 0.0 and res.high == 0.0
    assert res.unit == "utterance" and res.n_units == 200 and res.n_resamples == BS.DEFAULT_RESAMPLES


def test_clear_effect_excludes_zero_and_null_effect_does_not(rng: np.random.Generator) -> None:
    b = rng.standard_normal(300)
    a = b + 0.5 + 0.2 * rng.standard_normal(300)
    res = BS.paired_bootstrap(a, b)
    assert res.estimate == pytest.approx(np.mean(a - b))
    assert res.low < res.estimate < res.high
    assert res.excludes(0.0)
    null = BS.paired_bootstrap(b + 0.2 * rng.standard_normal(300), b)
    assert not null.excludes(0.0)


def test_pairing_removes_item_difficulty(rng: np.random.Generator) -> None:
    difficulty = 5.0 * rng.standard_normal(100)
    a = difficulty + 0.3 + 0.05 * rng.standard_normal(100)
    b = difficulty + 0.05 * rng.standard_normal(100)
    paired = BS.paired_bootstrap(a, b)
    assert paired.excludes(0.0)
    assert paired.high - paired.low < 0.1


def test_cluster_bootstrap_is_wider_for_correlated_clusters(rng: np.random.Generator) -> None:
    clusters = np.repeat([f"spk{i}" for i in range(6)], 50)
    effect = np.repeat(rng.normal(0.5, 1.0, 6), 50)  # the effect varies by speaker
    b = rng.standard_normal(300)
    a = b + effect + 0.1 * rng.standard_normal(300)
    utt = BS.paired_bootstrap(a, b)
    clu = BS.paired_bootstrap(a, b, clusters=list(clusters))
    assert clu.unit == "cluster" and clu.n_units == 6 and clu.n_items == 300
    assert clu.estimate == pytest.approx(utt.estimate)
    assert (clu.high - clu.low) > 3 * (utt.high - utt.low)


def test_rate_statistics_and_ratio(rng: np.random.Generator) -> None:
    minutes = rng.uniform(0.5, 2.0, 80)
    a_events = rng.poisson(0.5 * minutes)  # system A: ~0.5 per minute
    b_events = rng.poisson(2.0 * minutes)  # system B: ~2 per minute
    res = BS.paired_bootstrap(a_events, b_events, denominators=minutes, compare="ratio")
    assert res.estimate == pytest.approx(a_events.sum() / b_events.sum())
    assert res.high < 1.0  # A has clearly fewer false barge-ins per minute
    rate = BS.bootstrap_mean(a_events, denominators=minutes)
    assert rate.statistic == "rate"
    assert rate.estimate == pytest.approx(a_events.sum() / minutes.sum())


def test_ratio_with_zero_denominators_counts_invalid_resamples() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0])
    res = BS.paired_bootstrap(a, b, compare="ratio", n_resamples=200)
    assert res.n_invalid > 0


def test_reproducible_with_seed(rng: np.random.Generator) -> None:
    a, b = rng.standard_normal(50), rng.standard_normal(50)
    r1 = BS.paired_bootstrap(a, b, seed=3)
    r2 = BS.paired_bootstrap(a, b, seed=3)
    r3 = BS.paired_bootstrap(a, b, seed=4)
    assert (r1.low, r1.high) == (r2.low, r2.high)
    assert (r1.low, r1.high) != (r3.low, r3.high)
    assert set(r1.as_dict()) >= {"estimate", "low", "high", "unit", "n_units"}


def test_generic_cluster_bootstrap_median_delay(rng: np.random.Generator) -> None:
    delays_a = rng.normal(250, 20, 120)
    delays_b = rng.normal(400, 20, 120)
    clusters = [f"s{i % 8}" for i in range(120)]

    def median_gap(idx: np.ndarray) -> float:
        return float(np.median(delays_b[idx]) - np.median(delays_a[idx]))

    res = BS.cluster_bootstrap(median_gap, 120, clusters=clusters, n_resamples=300, name="median_delay_gap")
    assert res.statistic == "median_delay_gap" and res.n_units == 8
    assert res.low > 100.0


def test_bootstrap_mean_matches_resampling_theory(rng: np.random.Generator) -> None:
    x = rng.standard_normal(400)
    res = BS.bootstrap_mean(x, n_resamples=4000)
    se = x.std(ddof=0) / np.sqrt(x.size)
    assert (res.high - res.low) == pytest.approx(2 * 1.96 * se, rel=0.1)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        BS.paired_bootstrap([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        BS.paired_bootstrap([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError):
        BS.paired_bootstrap([1.0], [1.0], clusters=["a", "b"])
    with pytest.raises(ValueError):
        BS.paired_bootstrap([1.0], [1.0], compare="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BS.paired_bootstrap([1.0], [1.0], confidence=1.0)
    with pytest.raises(ValueError):
        BS.bootstrap_mean([])
