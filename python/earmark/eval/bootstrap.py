"""Paired bootstrap confidence intervals: speaker-cluster and utterance level.

Two systems are scored on the same items. Each resample draws *units* with replacement and
keeps every item of each drawn unit for both systems, so the comparison stays paired:

* ``clusters=None`` - utterance level: every item is its own unit (Suite B, where only two
  speakers exist and 'n=2 speakers' is printed next to every table).
* ``clusters=speaker_ids`` - speaker-cluster level: all items of a speaker move together
  (Suites A and R, and every pre-registered hypothesis).

System-level statistics are ratios of sums, which covers both plain means (denominator 1 per
item) and rates such as false barge-ins per minute (numerator = events, denominator =
minutes of target silence). Intervals are percentile intervals; the plan fixes 2,000
resamples and 95% confidence. For anything that is not a ratio of sums (a median delay, say)
use :func:`cluster_bootstrap` with your own statistic.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_RESAMPLES",
    "BootstrapResult",
    "bootstrap_mean",
    "cluster_bootstrap",
    "paired_bootstrap",
]

DEFAULT_RESAMPLES: int = 2000
DEFAULT_CONFIDENCE: float = 0.95

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate and percentile interval of one bootstrapped statistic."""

    statistic: str
    estimate: float
    low: float
    high: float
    confidence: float
    n_resamples: int
    n_units: int
    n_items: int
    unit: Literal["utterance", "cluster"]
    n_invalid: int = 0
    samples: FloatArray = field(repr=False, compare=False, default_factory=lambda: np.zeros(0))

    def excludes(self, value: float) -> bool:
        """True when ``value`` lies outside the interval (e.g. 0 for a difference, 1 for a ratio)."""
        return bool(value < self.low or value > self.high)

    def as_dict(self) -> dict[str, float | int | str]:
        """JSON-friendly summary (the resample array is left out)."""
        return {
            "statistic": self.statistic,
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "n_resamples": self.n_resamples,
            "n_units": self.n_units,
            "n_items": self.n_items,
            "unit": self.unit,
            "n_invalid": self.n_invalid,
        }


def _unit_index(n_items: int, clusters: Sequence[Hashable] | None) -> tuple[NDArray[np.int64], int]:
    """Map each item to a dense unit id."""
    if clusters is None:
        return np.arange(n_items, dtype=np.int64), n_items
    if len(clusters) != n_items:
        raise ValueError(f"clusters has {len(clusters)} entries for {n_items} items")
    _, inverse = np.unique(np.asarray([str(c) for c in clusters]), return_inverse=True)
    inverse = inverse.astype(np.int64)
    return inverse, int(inverse.max()) + 1 if n_items else 0


def _vector(x: ArrayLike | None, n: int, name: str, default: float) -> FloatArray:
    if x is None:
        return np.full(n, default, dtype=np.float64)
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or inf; drop or impute those items first")
    return a


def _resample_counts(n_units: int, n_resamples: int, seed: int) -> NDArray[np.int64]:
    """How many times each unit is drawn in each resample, shape (n_resamples, n_units)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=n_resamples)


def _interval(samples: FloatArray, confidence: float) -> tuple[float, float, int]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan"), int(samples.size)
    alpha = 1.0 - confidence
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high), int(samples.size - finite.size)


def _check(n_resamples: int, confidence: float) -> None:
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")


def paired_bootstrap(
    a: ArrayLike,
    b: ArrayLike,
    *,
    clusters: Sequence[Hashable] | None = None,
    denominators: ArrayLike | None = None,
    compare: Literal["difference", "ratio"] = "difference",
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> BootstrapResult:
    """Paired bootstrap of ``stat(A) - stat(B)`` or ``stat(A) / stat(B)``.

    ``a`` and ``b`` hold one value per item for systems A and B (same item order). Each
    system's statistic is ``sum(values) / sum(denominators)``: the mean when ``denominators``
    is omitted, a rate when it holds per-item exposure (minutes of target silence, frames...).
    Resamples whose ratio is undefined (zero denominator) are counted in ``n_invalid``.
    """
    _check(n_resamples, confidence)
    if compare not in ("difference", "ratio"):
        raise ValueError(f"compare must be 'difference' or 'ratio', got {compare!r}")
    va = np.asarray(a, dtype=np.float64)
    n = va.size
    va = _vector(va, n, "a", 0.0)
    vb = _vector(b, n, "b", 0.0)
    den = _vector(denominators, n, "denominators", 1.0)
    if n == 0:
        raise ValueError("need at least one item")
    unit, n_units = _unit_index(n, clusters)
    sum_a = np.bincount(unit, weights=va, minlength=n_units)
    sum_b = np.bincount(unit, weights=vb, minlength=n_units)
    sum_d = np.bincount(unit, weights=den, minlength=n_units)
    counts = _resample_counts(n_units, n_resamples, seed).astype(np.float64)
    ra, rb, rd = counts @ sum_a, counts @ sum_b, counts @ sum_d

    with np.errstate(divide="ignore", invalid="ignore"):
        stat_a, stat_b = ra / rd, rb / rd
        samples = stat_a - stat_b if compare == "difference" else stat_a / stat_b
        pa, pb = va.sum() / den.sum(), vb.sum() / den.sum()
        estimate = float(pa - pb) if compare == "difference" else float(pa / pb)
    low, high, invalid = _interval(samples, confidence)
    return BootstrapResult(
        statistic=f"paired_{compare}",
        estimate=estimate,
        low=low,
        high=high,
        confidence=confidence,
        n_resamples=n_resamples,
        n_units=n_units,
        n_items=n,
        unit="utterance" if clusters is None else "cluster",
        n_invalid=invalid,
        samples=samples,
    )


def bootstrap_mean(
    values: ArrayLike,
    *,
    clusters: Sequence[Hashable] | None = None,
    denominators: ArrayLike | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap interval of one system's mean (or rate, with ``denominators``)."""
    _check(n_resamples, confidence)
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        raise ValueError("need at least one item")
    v = _vector(v, n, "values", 0.0)
    den = _vector(denominators, n, "denominators", 1.0)
    unit, n_units = _unit_index(n, clusters)
    sum_v = np.bincount(unit, weights=v, minlength=n_units)
    sum_d = np.bincount(unit, weights=den, minlength=n_units)
    counts = _resample_counts(n_units, n_resamples, seed).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = (counts @ sum_v) / (counts @ sum_d)
    low, high, invalid = _interval(samples, confidence)
    return BootstrapResult(
        statistic="mean" if denominators is None else "rate",
        estimate=float(v.sum() / den.sum()),
        low=low,
        high=high,
        confidence=confidence,
        n_resamples=n_resamples,
        n_units=n_units,
        n_items=n,
        unit="utterance" if clusters is None else "cluster",
        n_invalid=invalid,
        samples=samples,
    )


def cluster_bootstrap(
    statistic: Callable[[NDArray[np.int64]], float],
    n_items: int,
    *,
    clusters: Sequence[Hashable] | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
    name: str = "custom",
) -> BootstrapResult:
    """Bootstrap an arbitrary statistic of item indices (with repeats).

    ``statistic(idx)`` receives the item indices of one resample (all items of each drawn
    unit, repeated as often as the unit was drawn) and returns a float; for a paired
    comparison, compute both systems' values from the same ``idx``. The point estimate is
    ``statistic(arange(n_items))``.
    """
    _check(n_resamples, confidence)
    if n_items < 1:
        raise ValueError("need at least one item")
    unit, n_units = _unit_index(n_items, clusters)
    members = [np.flatnonzero(unit == u) for u in range(n_units)]
    counts = _resample_counts(n_units, n_resamples, seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        drawn = np.repeat(np.arange(n_units), counts[r])
        idx = np.concatenate([members[u] for u in drawn]) if drawn.size else np.zeros(0, dtype=np.int64)
        samples[r] = float(statistic(idx))
    low, high, invalid = _interval(samples, confidence)
    return BootstrapResult(
        statistic=name,
        estimate=float(statistic(np.arange(n_items))),
        low=low,
        high=high,
        confidence=confidence,
        n_resamples=n_resamples,
        n_units=n_units,
        n_items=n_items,
        unit="utterance" if clusters is None else "cluster",
        n_invalid=invalid,
        samples=samples,
    )
