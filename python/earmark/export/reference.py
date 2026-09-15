"""NumPy reference implementations behind the engine goldens.

Each function is the float64 model of one engine layer:

* :class:`RingBufferModel`: the ``earmark::RingBuffer`` semantics (partial writes and
  reads, zero writes, discard, peek).
* :func:`design_resampler`, :func:`resampler_taps` and :func:`resample_reference`:
  the rational polyphase resampler. The Kaiser-window design here must match
  ``engine/src/resampler.cpp`` operation for operation: the same power series for
  ``I0``, ``math.sin`` rather than numpy's vectorised ``sin``, and the same summation
  order.
* :func:`gru_reference`: a stacked GRU with PyTorch's gate convention (tests check it
  against ``torch.nn.GRU``).
* :func:`matvec_reference` and :func:`grouped_matvec_reference`.

WOLA, ERB and normalisation goldens use :mod:`earmark.model.dsp` directly in float64, so
the engine is compared with the model's own code.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Final

import numpy as np

# ------------------------------------------------------------------------- ring buffer


class RingBufferModel:
    """Bounded FIFO of floats with the engine's partial-transfer semantics."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: deque[float] = deque()

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def space(self) -> int:
        return self.capacity - len(self._items)

    def write(self, values: np.ndarray) -> int:
        """Append as many of ``values`` as fit; returns how many were accepted."""
        accepted = min(len(values), self.space)
        self._items.extend(float(v) for v in values[:accepted])
        return accepted

    def write_zeros(self, count: int) -> int:
        accepted = min(count, self.space)
        self._items.extend([0.0] * accepted)
        return accepted

    def read(self, count: int) -> np.ndarray:
        """Pop up to ``count`` values, oldest first."""
        taken = min(count, self.size)
        return np.array([self._items.popleft() for _ in range(taken)], dtype=np.float64)

    def peek(self, count: int) -> np.ndarray:
        taken = min(count, self.size)
        return np.array([self._items[i] for i in range(taken)], dtype=np.float64)

    def discard(self, count: int) -> int:
        taken = min(count, self.size)
        for _ in range(taken):
            self._items.popleft()
        return taken


# --------------------------------------------------------------------------- resampler

#: Stop-band attenuation of the Kaiser design in dB (also the passband ripple scale).
RESAMPLER_ATTENUATION_DB: Final[float] = 120.0
#: Passband edge as a fraction of the lower of the two rates (7 kHz when 16 kHz is lower).
RESAMPLER_PASSBAND_FRACTION: Final[float] = 0.4375
#: Largest supported interpolation factor.
RESAMPLER_MAX_UP: Final[int] = 1024
#: Largest supported prototype filter length.
RESAMPLER_MAX_TAPS: Final[int] = 1 << 16


@dataclass(frozen=True)
class ResamplerDesign:
    """Parameters of the polyphase resampler ``in_rate -> out_rate``."""

    in_rate: int
    out_rate: int
    up: int
    down: int
    taps_per_phase: int
    prototype_taps: int
    cutoff: float  #: lower rate / (in_rate * up): normalised bandwidth of the prototype
    beta: float  #: Kaiser window shape

    @property
    def identity(self) -> bool:
        return self.up == 1 and self.down == 1

    @property
    def delay_seconds(self) -> float:
        """Group delay of the linear-phase prototype, in seconds."""
        if self.identity:
            return 0.0
        return (self.prototype_taps - 1) / (2.0 * self.in_rate * self.up)

    @property
    def delay_out_samples(self) -> float:
        return self.delay_seconds * self.out_rate


def design_resampler(in_rate: int, out_rate: int) -> ResamplerDesign:
    """Kaiser-window design for ``in_rate -> out_rate`` (mirrors ``earmark::design_resampler``)."""
    if in_rate <= 0 or out_rate <= 0:
        raise ValueError("rates must be positive")
    g = math.gcd(in_rate, out_rate)
    up, down = out_rate // g, in_rate // g
    if up == 1 and down == 1:
        return ResamplerDesign(in_rate, out_rate, 1, 1, 1, 1, 1.0, 0.0)
    if up > RESAMPLER_MAX_UP:
        raise ValueError(f"interpolation factor {up} exceeds {RESAMPLER_MAX_UP}")
    high = in_rate * up
    cutoff = min(in_rate, out_rate) / high
    transition = (1.0 - 2.0 * RESAMPLER_PASSBAND_FRACTION) * cutoff
    delta_omega = 2.0 * math.pi * transition
    order = math.ceil((RESAMPLER_ATTENUATION_DB - 7.95) / (2.285 * delta_omega))
    taps_per_phase = (order + 1 + up - 1) // up
    prototype_taps = taps_per_phase * up
    if prototype_taps > RESAMPLER_MAX_TAPS:
        raise ValueError(f"prototype needs {prototype_taps} taps (max {RESAMPLER_MAX_TAPS})")
    beta = 0.1102 * (RESAMPLER_ATTENUATION_DB - 8.7)
    return ResamplerDesign(in_rate, out_rate, up, down, taps_per_phase, prototype_taps, cutoff, beta)


def bessel_i0(x: float) -> float:
    """Modified Bessel function of the first kind, order 0, by its power series."""
    total = 1.0
    term = 1.0
    quarter = x * x / 4.0
    for k in range(1, 500):
        term *= quarter / (float(k) * float(k))
        total += term
        if term < total * 1e-17:
            break
    return total


def resampler_taps(design: ResamplerDesign) -> np.ndarray:
    """Prototype filter as float32 (length ``prototype_taps``), normalised to sum to ``up``."""
    n = design.prototype_taps
    if design.identity:
        return np.ones(1, dtype=np.float32)
    center = (n - 1) / 2.0
    i0_beta = bessel_i0(design.beta)
    values = [0.0] * n
    total = 0.0
    for k in range(n):
        x = design.cutoff * (k - center)
        sinc = 1.0 if x == 0.0 else math.sin(math.pi * x) / (math.pi * x)
        r = 2.0 * k / (n - 1) - 1.0
        window = bessel_i0(design.beta * math.sqrt(max(0.0, 1.0 - r * r))) / i0_beta
        values[k] = design.cutoff * sinc * window
        total += values[k]
    scale = design.up / total
    return np.array([v * scale for v in values], dtype=np.float32)


def resample_reference(x: np.ndarray, design: ResamplerDesign, taps: np.ndarray) -> np.ndarray:
    """Offline polyphase resampling in float64: ``y[m] = sum_k h[p + k up] x[i0 - k]``.

    ``t = m * down``, ``i0 = t // up``, ``p = t % up``; samples before the start are zero.
    Returns ``ceil(len(x) * up / down)`` outputs, exactly what the streaming engine emits.
    """
    x64 = np.asarray(x, dtype=np.float64)
    if design.identity:
        return x64.copy()
    n_out = (len(x64) * design.up + design.down - 1) // design.down
    m = np.arange(n_out, dtype=np.int64)
    t = m * design.down
    base = t // design.up
    phase = t % design.up
    k = np.arange(design.taps_per_phase, dtype=np.int64)
    index = base[:, None] - k[None, :]
    coef = np.asarray(taps, dtype=np.float64)[phase[:, None] + k[None, :] * design.up]
    samples = np.where(index >= 0, x64[np.clip(index, 0, None)], 0.0)
    return np.sum(coef * samples, axis=1)


# --------------------------------------------------------------------------------- GRU


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gru_reference(
    x: np.ndarray, layers: list[dict[str, np.ndarray]], h0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Stacked GRU in float64 with PyTorch's convention.

    Args:
        x: ``[T, input]`` inputs.
        layers: per layer ``{"weight_ih": [3H, in], "weight_hh": [3H, H], "bias_ih": [3H],
            "bias_hh": [3H]}``, gate rows ordered r, z, n.
        h0: ``[layers, H]`` initial hidden states.

    Returns:
        ``(y [T, H], h_final [layers, H])`` where ``y`` is the top layer's output.
    """
    h = np.array(h0, dtype=np.float64)
    hidden = h.shape[1]
    outputs = np.zeros((x.shape[0], hidden), dtype=np.float64)
    for t in range(x.shape[0]):
        inp = np.asarray(x[t], dtype=np.float64)
        for index, weights in enumerate(layers):
            gi = weights["weight_ih"].astype(np.float64) @ inp + weights["bias_ih"]
            gh = weights["weight_hh"].astype(np.float64) @ h[index] + weights["bias_hh"]
            r = _sigmoid(gi[:hidden] + gh[:hidden])
            z = _sigmoid(gi[hidden : 2 * hidden] + gh[hidden : 2 * hidden])
            n = np.tanh(gi[2 * hidden :] + r * gh[2 * hidden :])
            h[index] = (1.0 - z) * n + z * h[index]
            inp = h[index]
        outputs[t] = inp
    return outputs, h


# ------------------------------------------------------------------------------ matvec


def matvec_reference(w: np.ndarray, x: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """``W x + bias`` in float64 for row-major ``W`` ``[rows, cols]``."""
    y = np.asarray(w, dtype=np.float64) @ np.asarray(x, dtype=np.float64)
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64)
    return y


def grouped_matvec_reference(w: np.ndarray, x: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """GroupedLinear in float64: ``w`` ``[G, out/G, in/G]``, input group ``g`` feeds output group ``g``."""
    groups, out_g, in_g = w.shape
    xg = np.asarray(x, dtype=np.float64).reshape(groups, in_g)
    y = np.einsum("goi,gi->go", np.asarray(w, dtype=np.float64), xg).reshape(groups * out_g)
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64)
    return y


__all__ = [
    "RESAMPLER_ATTENUATION_DB",
    "RESAMPLER_MAX_TAPS",
    "RESAMPLER_MAX_UP",
    "RESAMPLER_PASSBAND_FRACTION",
    "ResamplerDesign",
    "RingBufferModel",
    "bessel_i0",
    "design_resampler",
    "grouped_matvec_reference",
    "gru_reference",
    "matvec_reference",
    "resample_reference",
    "resampler_taps",
]
