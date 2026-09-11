"""The interface every evaluated system implements, plus the trivial 'unprocessed' system.

An adapter (a baseline in :mod:`earmark.eval.baselines`, or the Earmark PyTorch-stream and
engine runners) turns a 16 kHz mono mixture into a time-aligned output of the same length
and, for gating systems, per-frame target-VAD probabilities at ``FRAME_RATE_HZ``. Scoring code
only depends on this protocol, so it never needs to know which runtime produced the audio.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from earmark import constants as C

__all__ = ["Enhancer", "EnhancerOutput", "SystemInfo", "Unprocessed", "as_output"]

AudioArray = NDArray[np.float32]


@dataclass(frozen=True)
class SystemInfo:
    """What every results row shows first: analytic size and compute, and the runtime used.

    ``inference_path`` is one of :data:`earmark.eval.runs_log.INFERENCE_PATHS`.
    """

    name: str
    inference_path: str
    params: int | None = None
    mmac_per_s: float | None = None
    training_data: str = ""
    licence: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EnhancerOutput:
    """System output for one mixture.

    ``audio`` has the mixture's length and is already compensated for the system's latency.
    ``vad`` (optional) holds one target-VAD probability per contract frame.
    """

    audio: AudioArray
    vad: NDArray[np.float32] | None = None


@runtime_checkable
class Enhancer(Protocol):
    """A system under evaluation."""

    info: SystemInfo
    sample_rate: int

    def enhance(self, mixture: AudioArray, embedding: NDArray[np.float32] | None = None) -> EnhancerOutput:
        """Process one mono mixture at ``sample_rate``; ``embedding`` is the enrolment vector."""
        ...


def as_output(result: EnhancerOutput | NDArray[np.floating], n_samples: int) -> EnhancerOutput:
    """Normalise an adapter's return value and check its length."""
    out = result if isinstance(result, EnhancerOutput) else EnhancerOutput(np.asarray(result, dtype=np.float32))
    audio = np.asarray(out.audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size != n_samples:
        raise ValueError(f"system output must be 1-D with {n_samples} samples, got shape {audio.shape}")
    if not np.all(np.isfinite(audio)):
        raise ValueError("system output contains NaN or inf")
    vad = None if out.vad is None else np.asarray(out.vad, dtype=np.float32)
    return EnhancerOutput(audio, vad)


class Unprocessed:
    """Identity system: the output is the mixture itself (the 'noisy' row of every table)."""

    sample_rate: int = C.SAMPLE_RATE

    def __init__(self) -> None:
        self.info = SystemInfo(name="unprocessed", inference_path="unprocessed", params=0, mmac_per_s=0.0)

    def enhance(self, mixture: AudioArray, embedding: NDArray[np.float32] | None = None) -> EnhancerOutput:
        return EnhancerOutput(np.asarray(mixture, dtype=np.float32))
