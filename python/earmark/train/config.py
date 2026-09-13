"""Training run configuration and the named presets.

A :class:`TrainConfig` fixes everything that determines a run: the model, the mixer
settings, the loss, the optimiser and its schedule, the precision, the session time limit
and the checkpoint cadence. The presets are the plan's runs:

=============  =======  ===============================================================
Preset         Model    What it is
=============  =======  ===============================================================
``smoke``      M        1 h pipeline check; checkpoints every 10 min so Hub pruning runs
``mini-full``  M        2 h, no agent-voice data (exploratory E2); scored on dev first
``M-v1``       M        about 20 GPU-h over two <= 11 h sessions
``S-GRU``      S-GRU    about 8 GPU-h (can share a session with M-v1 on the second T4)
``S-SSM``      S-SSM    about 8 GPU-h
``M-v2``       M        about 10 GPU-h fine-tune from M-v1 with re-targeted mixer weights
=============  =======  ===============================================================

Schedule length. Throughput on a T4 is not known in advance, so presets give a training
budget in hours instead of a step count: early in the first session (``calibrate_steps``,
inside the warm-up, where the learning rate does not depend on the schedule length) the
trainer measures steps per second and fixes ``max_steps`` so the cosine decay ends when
the budget is spent. The resolved value is stored in every checkpoint, so later sessions
and resumes follow the same schedule. Pass ``max_steps`` to fix it instead.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Final

from earmark.data.mixer import MixerConfig
from earmark.model.earmark_net import config_for
from earmark.train.losses import LossConfig

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_INIT_FROM = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]{0,63})(?:@(\d+))?$")


@dataclass(frozen=True)
class TrainConfig:
    """One training run. See the module docstring for the schedule and the presets."""

    #: Run name: the Hub path prefix ``runs/<name>/`` and the local log folder.
    name: str
    #: Model config name for :func:`earmark.model.build` (``M``, ``S-GRU``, ``S-SSM``, ``M-256``).
    model: str = "M"
    #: :class:`~earmark.model.earmark_net.EarmarkConfig` field overrides.
    model_overrides: Mapping[str, Any] = field(default_factory=dict)
    #: :class:`~earmark.data.mixer.MixerConfig` field overrides (for example ``batch_size``).
    mixer: Mapping[str, Any] = field(default_factory=dict)
    #: Train with the Kokoro agent-voice interferers (mini-full runs without them).
    use_agent_voice: bool = True
    #: Put MUSAN music beds under TV interferers.
    use_music: bool = True
    loss: LossConfig = field(default_factory=LossConfig)
    #: Seeds the model initialisation and the mixer's batch sequence.
    seed: int = 0
    lr: float = 5e-4
    #: Learning rate of the S4D ``log_dt``, ``log_a_real`` and ``a_imag`` parameters
    #: (no weight decay), following the S4 recipe. Only used by SSM bodies.
    ssm_lr: float = 1e-3
    #: Final learning rate as a fraction of the peak.
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 1000
    #: Fixed schedule length; ``None`` derives it from ``budget_hours`` (module docstring).
    max_steps: int | None = None
    #: Total training time across sessions, used when ``max_steps`` is ``None``.
    budget_hours: float | None = None
    #: Session-relative steps between which throughput is measured. The first few hundred
    #: steps run slower while the data pipeline fills, so measuring earlier undercounts
    #: throughput and ends the schedule early. The window ends inside every preset's warm-up.
    calibrate_steps: tuple[int, int] = (300, 500)
    grad_clip: float = 1.0
    #: fp16 autocast plus GradScaler for the network body (CUDA only; DSP and losses stay fp32).
    amp: bool = True
    #: Save and exit when the session has run this long.
    time_limit_hours: float = 11.0
    checkpoint_minutes: float = 20.0
    #: Checkpoints kept in storage (older ones are deleted).
    keep_last: int = 3
    log_every: int = 50
    #: Steps between validation passes on held-back mixer batches (0 disables them).
    val_every: int = 2000
    val_batches: int = 4
    #: Per-step losses logged after each checkpoint, the reference for the resume check.
    trace_steps: int = 100
    #: ``"<run>[@<step>]"``: start from another run's weights with a fresh optimiser.
    init_from: str | None = None
    #: Mixer batches drawn ahead on a background thread (0 draws inline).
    prefetch: int = 2

    def __post_init__(self) -> None:
        if not _RUN_NAME.match(self.name):
            raise ValueError(f"run name {self.name!r} must be letters, digits, '.', '_' or '-'")
        config_for(self.model)  # raises KeyError for an unknown model
        self.mixer_config()  # raises for unknown or invalid mixer settings
        if self.max_steps is None:
            if self.budget_hours is None or self.budget_hours <= 0:
                raise ValueError("set max_steps or a positive budget_hours")
            first, last = self.calibrate_steps
            if not 0 <= first < last:
                raise ValueError("calibrate_steps must be (first, last) with 0 <= first < last")
            if last > self.warmup_steps:
                raise ValueError("calibration must finish inside the warm-up (calibrate_steps[1] <= warmup_steps)")
        elif self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        positive = ("lr", "ssm_lr", "grad_clip", "time_limit_hours", "checkpoint_minutes")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        at_least_one = ("keep_last", "log_every", "val_batches")
        for name in at_least_one:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        for name in ("warmup_steps", "val_every", "trace_steps", "prefetch", "seed"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.min_lr_ratio <= 1.0 or self.weight_decay < 0:
            raise ValueError("min_lr_ratio must be in [0, 1] and weight_decay non-negative")
        if self.init_from is not None and not _INIT_FROM.match(self.init_from):
            raise ValueError(f"init_from {self.init_from!r} must look like 'M-v1' or 'M-v1@120000'")

    def mixer_config(self) -> MixerConfig:
        """The :class:`MixerConfig` with this run's overrides (JSON lists become tuples)."""
        known = {f.name for f in fields(MixerConfig)}
        unknown = set(self.mixer) - known
        if unknown:
            raise ValueError(f"unknown mixer settings {sorted(unknown)}")
        values = {k: tuple(v) if isinstance(v, list) else v for k, v in self.mixer.items()}
        return MixerConfig(**values)

    def init_source(self) -> tuple[str, int | None] | None:
        """``init_from`` as ``(run, step or None)``, or ``None``."""
        if self.init_from is None:
            return None
        match = _INIT_FROM.match(self.init_from)
        assert match is not None  # checked in __post_init__
        return match.group(1), int(match.group(2)) if match.group(2) else None

    def with_overrides(self, **changes: Any) -> TrainConfig:
        """A copy with fields replaced; ``mixer`` and ``model_overrides`` are merged, not replaced."""
        for key in ("mixer", "model_overrides"):
            if key in changes and changes[key] is not None:
                changes[key] = {**getattr(self, key), **changes[key]}
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form (stored in checkpoints and logs)."""
        data = asdict(self)
        data["mixer"] = dict(self.mixer)
        data["model_overrides"] = dict(self.model_overrides)
        return data


PRESETS: Final[dict[str, TrainConfig]] = {
    "smoke": TrainConfig(
        name="smoke", model="M", budget_hours=0.9, time_limit_hours=1.0, checkpoint_minutes=10.0,
        warmup_steps=300, calibrate_steps=(200, 300), log_every=25, val_every=500, val_batches=2,
    ),
    "mini-full": TrainConfig(
        name="mini-full", model="M", use_agent_voice=False, budget_hours=1.9, time_limit_hours=2.0,
        warmup_steps=1000, val_every=1000,
    ),
    "M-v1": TrainConfig(name="M-v1", model="M", budget_hours=20.0, warmup_steps=2000),
    "S-GRU": TrainConfig(name="S-GRU", model="S-GRU", budget_hours=8.0, lr=1e-3, warmup_steps=1000),
    "S-SSM": TrainConfig(name="S-SSM", model="S-SSM", budget_hours=8.0, lr=1e-3, warmup_steps=1000),
    # Fine-tune: fresh optimiser from M-v1's latest weights, a new batch sequence (seed 1)
    # and the mixer re-weighting from the week-2 listening pass (pass it with --mixer).
    "M-v2": TrainConfig(
        name="M-v2", model="M", init_from="M-v1", seed=1, budget_hours=10.0, lr=2e-4, warmup_steps=500,
    ),
}


def preset(name: str) -> TrainConfig:
    """Look up a preset; case and ``_``/``-`` are ignored (``m_v1`` == ``M-v1``)."""
    wanted = name.strip().lower().replace("_", "-")
    for key, config in PRESETS.items():
        if key.lower() == wanted:
            return config
    raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")


__all__ = ["PRESETS", "TrainConfig", "preset"]
