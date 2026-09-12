"""The Earmark trainer (M3): fp16 AMP for the network, a session clock, Hub checkpoints.

What one run does
-----------------
1. Refuses a CUDA device below compute capability 7.5. Kaggle's P100 is sm_60, which the
   current image does not support.
2. Resumes from the newest checkpoint in storage, restoring the model, optimiser,
   GradScaler, every RNG, the mixer's generator state, the step and the schedule. It
   refuses a different model, signal contract or GPU type. Without a checkpoint it starts
   fresh, optionally from another run's weights (``init_from``, the v2 fine-tune).
3. Trains. The network runs under fp16 autocast on CUDA; the model keeps its WOLA and ERB
   DSP in fp32 itself. The loss runs in fp32 with autocast disabled, and a GradScaler
   handles the fp16 gradients.
4. Every ``checkpoint_minutes`` it saves a checkpoint and pushes it, with the log, to
   storage, which keeps the newest ``keep_last``. After each checkpoint the next
   ``trace_steps`` per-step losses go to the log, as the reference for the resume check.
5. Stops at ``max_steps``, or when the session clock reaches ``time_limit_hours``, saving
   first. The clock starts at ``EARMARK_SESSION_START`` when set, so notebook setup counts.

Resume check
------------
``--verify-resume`` restores the newest stored checkpoint that has a complete trace,
trains ``--verify-steps`` steps without saving anything, and compares the per-step
losses with the trace. They are bit-identical on a CPU. On a GPU they must agree within
``--verify-tolerance`` (default 1e-3, relative to ``max(1, |loss|)``), because atomic
adds make some CUDA kernels non-deterministic.

Command line
------------
``python -m earmark.train.train --preset M-v1 --data-root /kaggle/input --storage hub
--repo-id <user>/earmark-checkpoints``; see ``--help``. The last line printed is
``EARMARK_RESULT {json}`` (or ``EARMARK_VERIFY {json}``) for the notebook to parse.

Exit status:

* 0 when the run finished, or hit the time limit, and its checkpoint was pushed;
* 2 when the final push failed (the checkpoint is then kept in ``<work-dir>/<run>/unpushed/``);
* 4 when the resume check failed;
* 1 on any other error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import torch
from torch import Tensor, nn

from earmark import constants as C
from earmark.data.mixer import Mixer
from earmark.model.earmark_net import build
from earmark.train.checkpoint import (
    CHECKPOINT_FORMAT,
    ResumeError,
    capture_rng_state,
    check_resume_compatible,
    device_info,
    load_checkpoint,
    provenance,
    restore_rng_state,
    save_checkpoint,
    seed_everything,
)
from earmark.train.config import TrainConfig, preset
from earmark.train.losses import TERMS, LossOutput, EarmarkLoss
from earmark.train.pools import discover_training_data
from earmark.train.storage import (
    CONFIG_NAME,
    LOG_NAME,
    CheckpointStorage,
    StorageError,
    checkpoint_name,
    open_storage,
)
from earmark.train.validation import VAL_SEED_OFFSET, validate

#: Kaggle's P100 (sm_60) is unsupported by the current image; T4 is 7.5.
MIN_CUDA_CAPABILITY: Final[tuple[int, int]] = (7, 5)
#: Fraction of the measured throughput used to size the schedule (checkpoints and
#: validation also take time).
CALIBRATION_MARGIN: Final[float] = 0.95
#: Consecutive non-finite losses tolerated before the run aborts.
MAX_NONFINITE_STEPS: Final[int] = 50
#: Environment variable with the notebook session's start time (Unix seconds).
SESSION_START_ENV: Final[str] = "EARMARK_SESSION_START"
#: S4D parameters that get ``ssm_lr`` and no weight decay.
SSM_PARAM_SUFFIXES: Final[tuple[str, ...]] = (".log_dt", ".log_a_real", ".a_imag")
#: Stats averaged into each ``train`` log record, in this order.
STAT_KEYS: Final[tuple[str, ...]] = ("si_sdr_db", "si_sdri_db", "absent_db", "n_active", "n_silent")


class UnsupportedDeviceError(RuntimeError):
    """The requested device cannot run the trainer (for example a P100 on Kaggle)."""


class NonFiniteLossError(RuntimeError):
    """The loss stayed NaN or infinite for too many consecutive steps."""


# ------------------------------------------------------------------------ building blocks


def check_device(device: torch.device) -> dict[str, Any]:
    """Refuse CUDA devices below compute capability 7.5; return the device description."""
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise UnsupportedDeviceError("CUDA was requested but is not available")
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) < MIN_CUDA_CAPABILITY:
            raise UnsupportedDeviceError(
                f"{torch.cuda.get_device_name(device)} has compute capability {major}.{minor}; the trainer "
                "needs 7.5 or newer. On Kaggle choose the 'GPU T4 x2' accelerator (P100 is sm_60)."
            )
    return device_info(device)


def lr_factor(step: int, warmup_steps: int, max_steps: int | None, min_ratio: float) -> float:
    """Linear warm-up to 1 over ``warmup_steps``, then cosine decay to ``min_ratio`` at ``max_steps``.

    ``step`` counts completed updates. Before ``max_steps`` is known the factor after
    warm-up is 1.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if max_steps is None or max_steps <= warmup_steps:
        return 1.0
    progress = min(1.0, (step - warmup_steps) / (max_steps - warmup_steps))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _group_hparams(config: TrainConfig) -> dict[str, tuple[float, float]]:
    return {
        "decay": (config.lr, config.weight_decay),
        "no_decay": (config.lr, 0.0),
        "ssm": (config.ssm_lr, 0.0),
    }


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    """AdamW with three groups: decayed weights; biases, norms and the NULL embedding
    (no decay); S4D ``log_dt``/``log_a_real``/``a_imag`` (``ssm_lr``, no decay).

    Each group records ``base_lr``; the schedule sets ``lr = base_lr * lr_factor(step)``.
    """
    members: dict[str, list[nn.Parameter]] = {"decay": [], "no_decay": [], "ssm": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(SSM_PARAM_SUFFIXES):
            members["ssm"].append(param)
        elif param.dim() < 2 or name.endswith("null_embedding"):
            members["no_decay"].append(param)
        else:
            members["decay"].append(param)
    hparams = _group_hparams(config)
    groups = [
        {"params": params, "name": name, "base_lr": hparams[name][0], "lr": hparams[name][0],
         "weight_decay": hparams[name][1]}
        for name, params in members.items()
        if params
    ]  # fmt: skip
    return torch.optim.AdamW(groups, lr=config.lr, betas=config.betas, eps=1e-8)


def make_grad_scaler(device: torch.device, enabled: bool) -> Any:
    """A GradScaler for ``device`` (inert when disabled)."""
    scaler_cls = getattr(torch.amp, "GradScaler", None)
    if scaler_cls is not None:
        return scaler_cls(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover - torch < 2.3


class WallClock:
    """Session time limit and checkpoint interval, both measured on one clock."""

    def __init__(
        self, limit_s: float, interval_s: float, clock: Callable[[], float], start: float | None = None
    ) -> None:
        self.clock = clock
        self.limit_s = limit_s
        self.interval_s = interval_s
        self.start = clock() if start is None else start
        self.last_checkpoint = self.start

    def now(self) -> float:
        return self.clock()

    def expired(self, now: float) -> bool:
        return now - self.start >= self.limit_s

    def checkpoint_due(self, now: float) -> bool:
        return now - self.last_checkpoint >= self.interval_s

    def mark_checkpoint(self, now: float) -> None:
        self.last_checkpoint = now


def _json_default(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=_json_default)
    return str(value)


class RunLog:
    """Append-only JSON-lines log of a run, echoed to the console.

    Per-step ``trace`` records (the resume-check reference) are written but not echoed.
    """

    def __init__(self, path: Path, *, run: str, echo: Callable[[str], None] | None = print) -> None:
        self.path = path
        self.run = run
        self.echo = echo
        path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self, source: Path | None) -> None:
        """Start from the stored log (a resumed run), or empty (a fresh run)."""
        if source is None:
            self.path.write_text("")
        elif Path(source).resolve() != self.path.resolve():
            shutil.copyfile(source, self.path)

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {"event": event, "run": self.run, "time": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")
        if self.echo is not None and event != "trace":
            shown = " ".join(
                f"{k}={_format_value(v)}" for k, v in fields.items() if k != "config" and v is not None
            )
            self.echo(f"[{self.run}] {event} {shown}")
        return record

    @staticmethod
    def read(path: str | Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------- results


@dataclass
class TrainResult:
    """Outcome of :meth:`Trainer.run`."""

    run: str
    status: str  #: "finished", "time_limit" or "stopped"
    step: int
    max_steps: int | None
    checkpoint_step: int | None
    pushed: bool
    train_seconds: float
    storage: str
    losses: list[float] = field(default_factory=list, repr=False)  #: per-step losses this session

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("losses")
        return data


@dataclass
class VerifyResult:
    """Outcome of :meth:`Trainer.verify`."""

    run: str
    checkpoint_step: int
    steps: int
    max_abs_diff: float
    max_rel_diff: float
    tolerance: float
    passed: bool
    losses: list[float] = field(default_factory=list, repr=False)
    reference: list[float] = field(default_factory=list, repr=False)

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("losses")
        data.pop("reference")
        return data


@dataclass
class _Step:
    total: float
    loss: LossOutput
    grad_norm: float
    lr: float


class _Window:
    """Running sums for one ``train`` log record."""

    def __init__(self) -> None:
        self.totals: list[float] = []
        self.terms: list[Tensor] = []
        self.stats: list[Tensor] = []
        self.grad_norms: list[float] = []
        self.data_s = 0.0
        self.busy_s = 0.0

    def add(self, step: _Step, data_s: float, busy_s: float) -> None:
        self.totals.append(step.total)
        self.terms.append(torch.stack([step.loss.terms[k] for k in TERMS]))
        self.stats.append(torch.stack([step.loss.stats[k] for k in STAT_KEYS]))
        self.grad_norms.append(step.grad_norm)
        self.data_s += data_s
        self.busy_s += busy_s

    def summary(self) -> dict[str, float]:
        terms = torch.stack(self.terms).mean(0).tolist()
        stats = torch.stack(self.stats).nanmean(0).tolist()
        finite = [g for g in self.grad_norms if math.isfinite(g)]
        out = {"loss": sum(self.totals) / len(self.totals)}
        out.update(dict(zip(TERMS, terms)))
        out.update(dict(zip(STAT_KEYS, stats)))
        out["grad_norm"] = sum(finite) / len(finite) if finite else math.nan
        out["steps_per_s"] = len(self.totals) / self.busy_s if self.busy_s > 0 else math.nan
        out["data_frac"] = self.data_s / self.busy_s if self.busy_s > 0 else math.nan
        return out


def _close(batches: Generator[Any, None, None]) -> None:
    batches.close()


# ---------------------------------------------------------------------------- trainer


class Trainer:
    """One training run over a :class:`~earmark.data.mixer.Mixer`, checkpointing to storage.

    ``clock`` (seconds) drives the session limit, the checkpoint interval and the
    training-time budget; it defaults to ``time.time`` with ``session_start`` and to
    ``time.monotonic`` otherwise. The trainer reads it once at construction, once when
    the loop starts and once after every step, so tests can drive it with a fake clock.
    """

    def __init__(
        self,
        config: TrainConfig,
        mixer: Mixer,
        storage: CheckpointStorage,
        *,
        device: str | torch.device = "cpu",
        val_mixer: Mixer | None = None,
        work_dir: str | Path | None = None,
        clock: Callable[[], float] | None = None,
        session_start: float | None = None,
        echo: Callable[[str], None] | None = print,
        allow_device_change: bool = False,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.device_info = check_device(self.device)
        if mixer.device.type != self.device.type:
            raise ValueError(f"the mixer renders on {mixer.device}, the trainer runs on {self.device}")
        self.mixer = mixer
        self.val_mixer = val_mixer
        self.storage = storage
        self.allow_device_change = allow_device_change
        self.echo = echo
        seed_everything(config.seed)
        self.model = build(config.model, **dict(config.model_overrides)).to(self.device)
        self.loss_fn = EarmarkLoss(config.loss)
        self.optimizer = build_optimizer(self.model, config)
        self.use_amp = config.amp and self.device.type == "cuda"
        self.scaler = make_grad_scaler(self.device, self.use_amp)
        self.step = 0
        self.train_seconds = 0.0
        self.max_steps = config.max_steps
        self.calibration: dict[str, float] | None = None
        if clock is None:
            clock = time.time if session_start is not None else time.monotonic
        self.timer = WallClock(config.time_limit_hours * 3600.0, config.checkpoint_minutes * 60.0, clock, session_start)
        base = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir()) / "earmark_runs"
        self.work_dir = base / config.name
        # Checkpoints are written here, pushed, then deleted. It is outside work_dir, which on
        # Kaggle is the notebook output; only an unpushed fallback copy is kept there.
        self.scratch_dir = Path(tempfile.gettempdir()) / f"earmark-checkpoints-{os.getpid()}" / config.name
        self.log = RunLog(self.work_dir / LOG_NAME, run=config.name, echo=echo)
        self.stamp = provenance()
        self._restored = False
        self._stop_requested = False
        self._nonfinite = 0

    # ------------------------------------------------------------------ utilities

    def _say(self, message: str) -> None:
        if self.echo is not None:
            self.echo(f"[{self.config.name}] {message}")

    def request_stop(self) -> None:
        """Ask the loop to save and exit after the current step (for SIGTERM)."""
        self._stop_requested = True

    def _apply_hparams(self) -> None:
        hparams = _group_hparams(self.config)
        for group in self.optimizer.param_groups:
            group["base_lr"], group["weight_decay"] = hparams[group["name"]]

    def _batches(self) -> Generator[dict[str, Tensor], None, None]:
        if self.config.prefetch > 0:
            return self.mixer.iterate(prefetch=self.config.prefetch)
        return iter(self.mixer)  # type: ignore[return-value]

    # ----------------------------------------------------------- restore and save

    def restore(self) -> str:
        """Resume from storage, else load ``init_from`` weights, else start fresh.

        Returns ``"resumed"``, ``"init_from"`` or ``"fresh"``. :meth:`run` calls it.
        """
        if self._restored:
            raise RuntimeError("restore() already ran for this trainer")
        self._restored = True
        steps = self.storage.steps()
        if steps:
            self._load(steps[-1])
            self.log.reset(self.storage.fetch_extra(LOG_NAME))
            return "resumed"
        self.log.reset(None)
        source = self.config.init_source()
        if source is not None:
            self._init_from(*source)
            return "init_from"
        return "fresh"

    def _load(self, step: int) -> dict[str, Any]:
        payload = load_checkpoint(self.storage.fetch(step), map_location=self.device)
        warnings = check_resume_compatible(
            payload, model_config=asdict(self.model.config), device=self.device_info,
            allow_device_change=self.allow_device_change,
        )  # fmt: skip
        for warning in warnings:
            self._say(f"warning: {warning}")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self._apply_hparams()
        if self.use_amp and payload["scaler"]:
            self.scaler.load_state_dict(payload["scaler"])
        self.mixer.load_state_dict(payload["mixer"])
        restore_rng_state(payload["rng"])
        self.step = int(payload["step"])
        self.train_seconds = float(payload["train_seconds"])
        saved_max = payload["max_steps"]
        if self.config.max_steps is not None:
            self.max_steps = self.config.max_steps
        else:
            self.max_steps = None if saved_max is None else int(saved_max)
        self.calibration = payload.get("calibration")
        return payload

    def _init_from(self, run: str, step: int | None) -> None:
        source = self.storage.for_run(run)
        steps = source.steps()
        if not steps:
            raise ResumeError(f"init_from: run {run!r} has no checkpoints in {source.describe()}")
        chosen = steps[-1] if step is None else step
        if chosen not in steps:
            raise ResumeError(f"init_from: run {run!r} has no checkpoint at step {chosen} (stored: {steps})")
        payload = load_checkpoint(source.fetch(chosen), map_location=self.device)
        if payload["contract_hash"] != C.CONTRACT_HASH:
            raise ResumeError(f"init_from: {run} was trained under contract {payload['contract_hash']}")
        if dict(payload["model_config"]) != asdict(self.model.config):
            raise ResumeError(f"init_from: {run} is a {payload['model_config']} model, this run builds {self.config.model}")
        self.model.load_state_dict(payload["model"])
        self.log.write("init_from", source=run, source_step=chosen, source_git_sha=payload["git_sha"])

    def _payload(self, status: str) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "status": status,
            "step": self.step,
            "max_steps": self.max_steps,
            "train_seconds": self.train_seconds,
            "calibration": self.calibration,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": capture_rng_state(),
            "mixer": self.mixer.state_dict(),
            "config": self.config.to_dict(),
            "model_config": asdict(self.model.config),
            "device": self.device_info,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **self.stamp,
        }

    def _save(self, status: str, now: float) -> bool:
        """Save a checkpoint at the current step and push it; returns whether the push worked."""
        local = save_checkpoint(self.scratch_dir / checkpoint_name(self.step), self._payload(status))
        config_path = self.work_dir / CONFIG_NAME
        config_path.write_text(json.dumps({"config": self.config.to_dict(), **self.stamp}, indent=2) + "\n")
        self.log.write("checkpoint", step=self.step, status=status, bytes=local.stat().st_size)
        started = time.perf_counter()
        try:
            self.storage.push(self.step, local, {LOG_NAME: self.log.path, CONFIG_NAME: config_path})
            pushed = True
        except StorageError as err:
            pushed = False
            kept = self.work_dir / "unpushed" / local.name
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, kept)
            self.log.write("push_failed", step=self.step, error=str(err), kept=str(kept))
        self.timer.mark_checkpoint(now)
        local.unlink(missing_ok=True)
        where = self.storage.describe()
        self._say(
            f"checkpoint {self.step} {'pushed to' if pushed else 'NOT pushed to'} {where} "
            f"in {time.perf_counter() - started:.1f} s"
        )
        return pushed

    # --------------------------------------------------------------------- training

    def _train_step(self, batch: Mapping[str, Tensor]) -> _Step:
        self.model.train()
        factor = lr_factor(self.step, self.config.warmup_steps, self.max_steps, self.config.min_lr_ratio)
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            out = self.model(batch["mixture"], batch["embedding"], null_mask=batch["null_embedding"])
        loss = self.loss_fn(out, batch)  # fp32, autocast disabled inside
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss.total).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip))
        total = float(loss.total.detach())
        if self.use_amp:
            self.scaler.step(self.optimizer)  # skipped internally when gradients overflowed
            self.scaler.update()
        elif math.isfinite(total) and math.isfinite(grad_norm):
            self.optimizer.step()
        self._nonfinite = 0 if math.isfinite(total) else self._nonfinite + 1
        if self._nonfinite >= MAX_NONFINITE_STEPS:
            raise NonFiniteLossError(f"loss was non-finite for {MAX_NONFINITE_STEPS} consecutive steps at step {self.step}")
        return _Step(total=total, loss=loss, grad_norm=grad_norm, lr=self.config.lr * factor)

    def _calibrate(self, mark: tuple[int, float], session_steps: int) -> None:
        first_steps, first_seconds = mark
        elapsed = self.train_seconds - first_seconds
        rate = (session_steps - first_steps) / elapsed if elapsed > 0 else math.inf
        assert self.config.budget_hours is not None  # max_steps is None only with a budget
        remaining = max(0.0, self.config.budget_hours * 3600.0 - self.train_seconds)
        planned = int(remaining * rate * CALIBRATION_MARGIN) if math.isfinite(rate) else 0
        self.max_steps = self.step + max(1, planned)
        self.calibration = {
            "steps_per_s": rate,
            "at_step": float(self.step),
            "budget_hours": float(self.config.budget_hours),
            "max_steps": float(self.max_steps),
        }
        self.log.write("calibrated", step=self.step, steps_per_s=rate, max_steps=self.max_steps,
                       budget_hours=self.config.budget_hours)  # fmt: skip

    def _log_train(self, window: _Window) -> None:
        summary = window.summary()
        rate = summary["steps_per_s"]
        eta = None
        if self.max_steps is not None and math.isfinite(rate) and rate > 0:
            eta = (self.max_steps - self.step) / rate / 3600.0
        scale = float(self.scaler.get_scale()) if self.use_amp else None
        lr = self.optimizer.param_groups[0]["lr"]
        self.log.write(
            "train", step=self.step, max_steps=self.max_steps, lr=lr, scale=scale,
            train_hours=self.train_seconds / 3600.0, eta_hours=eta, **summary,
        )  # fmt: skip

    def run(self, info: Mapping[str, Any] | None = None) -> TrainResult:
        """Train until ``max_steps`` or the session limit; always ends with a pushed checkpoint.

        ``info`` is added to the log's ``start`` record (for example the data paths).
        """
        self.storage.prepare()
        how = self.restore() if not self._restored else "restored"
        if self.max_steps is not None and self.step >= self.max_steps:
            self.log.write("end", status="finished", step=self.step, note="already at max_steps")
            return self._result("finished", self.step, True, [])
        self.log.write(
            "start", how=how, step=self.step, max_steps=self.max_steps, amp=self.use_amp,
            device=self.device_info, storage=self.storage.describe(), config=self.config.to_dict(),
            **self.stamp, **dict(info or {}),
        )  # fmt: skip
        last = self.timer.now()
        if self.timer.expired(last):
            self.log.write("end", status="time_limit", step=self.step, note="no session time left")
            return self._result("time_limit", self.step, True, [])
        losses: list[float] = []
        trace_after: int | None = self.step if how == "resumed" else None
        first_calib, final_calib = self.config.calibrate_steps
        mark: tuple[int, float] | None = None
        session_steps = 0
        window = _Window()
        status = "finished"
        batches = self._batches()
        try:
            while self.max_steps is None or self.step < self.max_steps:
                started = time.perf_counter()
                batch = next(batches)
                fetched = time.perf_counter()
                result = self._train_step(batch)
                self.step += 1
                session_steps += 1
                now = self.timer.now()
                self.train_seconds += now - last
                last = now
                losses.append(result.total)
                window.add(result, fetched - started, time.perf_counter() - started)
                if trace_after is not None and self.step - trace_after <= self.config.trace_steps:
                    self.log.write("trace", step=self.step, after=trace_after, loss=result.total)
                if self.max_steps is None:
                    if mark is None and session_steps >= first_calib:
                        mark = (session_steps, self.train_seconds)
                    elif mark is not None and (
                        session_steps >= final_calib
                        or (self.step >= self.config.warmup_steps and session_steps - mark[0] >= 10)
                    ):
                        self._calibrate(mark, session_steps)
                if self.step % self.config.log_every == 0:
                    self._log_train(window)
                    window = _Window()
                if self.val_mixer is not None and self.config.val_every and self.step % self.config.val_every == 0:
                    metrics = validate(self.model, self.loss_fn, self.val_mixer, self.config.val_batches, amp=self.use_amp)
                    self.log.write("val", step=self.step, **metrics)
                if self.max_steps is not None and self.step >= self.max_steps:
                    break
                if self.timer.expired(now):
                    status = "time_limit"
                    break
                if self._stop_requested:
                    status = "stopped"
                    break
                if self.timer.checkpoint_due(now):
                    self._save("running", now)
                    trace_after = self.step
        finally:
            _close(batches)
        self.log.write("end", status=status, step=self.step, max_steps=self.max_steps,
                       train_hours=self.train_seconds / 3600.0)  # fmt: skip
        pushed = self._save(status, last)
        return self._result(status, self.step, pushed, losses)

    def _result(self, status: str, checkpoint_step: int | None, pushed: bool, losses: list[float]) -> TrainResult:
        return TrainResult(
            run=self.config.name, status=status, step=self.step, max_steps=self.max_steps,
            checkpoint_step=checkpoint_step, pushed=pushed, train_seconds=self.train_seconds,
            storage=self.storage.describe(), losses=losses,
        )  # fmt: skip

    # ------------------------------------------------------------------ resume check

    def verify(self, steps: int = 100, tolerance: float = 1e-3, checkpoint_step: int | None = None) -> VerifyResult:
        """Resume check: restore a stored checkpoint, train ``steps`` steps without saving,
        and compare per-step losses with the ones the original run logged after it.

        Picks the newest stored checkpoint with a complete trace unless ``checkpoint_step``
        is given. Passes when every ``|new - logged| <= tolerance * max(1, |logged|)``.
        """
        if self._restored:
            raise RuntimeError("verify() needs a trainer that has not restored or run yet")
        self._restored = True
        log_path = self.storage.fetch_extra(LOG_NAME)
        if log_path is None:
            raise ResumeError(f"no {LOG_NAME} in {self.storage.describe()}")
        records = RunLog.read(log_path)
        traces: dict[tuple[int, int], float] = {}
        for record in records:
            if record.get("event") == "trace":
                traces.setdefault((int(record["after"]), int(record["step"])), float(record["loss"]))
        stored = self.storage.steps()
        complete = [k for k in stored if all((k, k + i) in traces for i in range(1, steps + 1))]
        if checkpoint_step is not None:
            if checkpoint_step not in complete:
                raise ResumeError(f"checkpoint {checkpoint_step} is not stored with {steps} traced steps after it")
            chosen = checkpoint_step
        elif complete:
            chosen = complete[-1]
        else:
            raise ResumeError(
                f"no stored checkpoint (of {stored}) has {steps} traced steps after it; "
                f"the run must continue at least {steps} steps past a checkpoint"
            )
        self._load(chosen)
        if self.max_steps is None:
            calibrated = [r for r in records if r.get("event") == "calibrated"]
            if calibrated:
                self.max_steps = int(calibrated[-1]["max_steps"])
        losses: list[float] = []
        batches = self._batches()
        try:
            for _ in range(steps):
                losses.append(self._train_step(next(batches)).total)
                self.step += 1
        finally:
            _close(batches)
        reference = [traces[(chosen, chosen + i)] for i in range(1, steps + 1)]
        abs_diff = [abs(a - b) for a, b in zip(losses, reference)]
        rel_diff = [d / max(1.0, abs(b)) for d, b in zip(abs_diff, reference)]
        finite = all(math.isfinite(x) for x in losses)
        result = VerifyResult(
            run=self.config.name, checkpoint_step=chosen, steps=steps, max_abs_diff=max(abs_diff),
            max_rel_diff=max(rel_diff), tolerance=tolerance, passed=finite and max(rel_diff) <= tolerance,
            losses=losses, reference=reference,
        )  # fmt: skip
        self._say(
            f"resume check from step {chosen} over {steps} steps: max |diff| {result.max_abs_diff:.3g}, "
            f"max relative {result.max_rel_diff:.3g} -> {'PASS' if result.passed else 'FAIL'}"
        )
        return result


# ------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m earmark.train.train",
        description="Train an Earmark model from a preset, checkpointing to a private HF repo or a local folder.",
    )
    parser.add_argument("--preset", required=True, help="smoke, mini-full, M-v1, S-GRU, S-SSM or M-v2")
    parser.add_argument("--run-name", help="run name (default: the preset's); a new name starts a fresh run")
    parser.add_argument("--data-root", default="/kaggle/input", help="folder searched for the prepared datasets")
    parser.add_argument("--storage", choices=("hub", "local"), default="hub")
    parser.add_argument("--repo-id", help="private HF model repo for checkpoints, e.g. <user>/earmark-checkpoints")
    parser.add_argument("--local-dir", help="checkpoint folder for --storage local")
    parser.add_argument("--work-dir", help="local folder for logs and temporary checkpoint files")
    parser.add_argument("--device", help="cuda or cpu (default: cuda when available)")
    parser.add_argument("--max-steps", type=int, help="fix the schedule length instead of calibrating it")
    parser.add_argument("--budget-hours", type=float, help="training-time budget across sessions")
    parser.add_argument("--time-limit-hours", type=float, help="save and exit after this long (default 11)")
    parser.add_argument("--checkpoint-minutes", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--init-from", help="RUN or RUN@STEP to fine-tune from")
    parser.add_argument("--mixer", type=json.loads, help='JSON MixerConfig overrides, e.g. \'{"p_interferer": 0.7}\'')
    parser.add_argument("--no-amp", action="store_true", help="train the network in fp32")
    parser.add_argument("--allow-device-change", action="store_true", help="resume on a different GPU type")
    parser.add_argument(
        "--allow-missing-heldout", action="store_true",
        help="train even if a speech dataset lacks heldout_speakers.json (only the default hold-outs are checked)",
    )  # fmt: skip
    parser.add_argument("--verify-resume", action="store_true", help="run the resume check instead of training")
    parser.add_argument("--verify-steps", type=int, default=100)
    parser.add_argument("--verify-tolerance", type=float, default=1e-3)
    parser.add_argument("--verify-checkpoint", type=int, help="checkpoint step to check (default: newest traced)")
    return parser


def resolve_config(args: argparse.Namespace) -> TrainConfig:
    """The preset with the command-line overrides applied."""
    config = preset(args.preset)
    changes: dict[str, Any] = {}
    for key in ("max_steps", "budget_hours", "time_limit_hours", "checkpoint_minutes", "lr", "seed", "init_from"):
        value = getattr(args, key)
        if value is not None:
            changes[key] = value
    if args.run_name:
        changes["name"] = args.run_name
    if args.no_amp:
        changes["amp"] = False
    if args.mixer is not None and not isinstance(args.mixer, dict):
        raise ValueError("--mixer must be a JSON object")
    mixer = dict(args.mixer or {})
    if args.batch_size is not None:
        mixer["batch_size"] = args.batch_size
    if mixer:
        changes["mixer"] = mixer
    return config.with_overrides(**changes)


def train_command(
    preset_name: str,
    *,
    data_root: str | Path,
    repo_id: str,
    work_dir: str | Path,
    extra: Sequence[str] = (),
    verify: bool = False,
    python: str = sys.executable,
) -> list[str]:
    """The command line the Kaggle notebook launches for one preset on one GPU."""
    cmd = [
        python, "-m", "earmark.train.train", "--preset", preset_name, "--data-root", str(data_root),
        "--storage", "hub", "--repo-id", repo_id, "--device", "cuda", "--work-dir", str(work_dir),
    ]  # fmt: skip
    if verify:
        cmd.append("--verify-resume")
    return cmd + list(extra)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    check_device(device)
    data = discover_training_data(
        args.data_root, use_agent_voice=config.use_agent_voice, use_music=config.use_music,
        allow_missing_heldout=args.allow_missing_heldout,
    )  # fmt: skip
    mixer_config = config.mixer_config()
    mixer = Mixer(data.pools, mixer_config, seed=config.seed, device=device, held_out=data.held_out)
    val_mixer = None
    if config.val_every:
        # Same pools (already checked for leaks above), another seed.
        val_mixer = Mixer(data.pools, mixer_config, seed=config.seed + VAL_SEED_OFFSET, device=device, held_out=None)
    storage = open_storage(
        args.storage, run=config.name, repo_id=args.repo_id, local_dir=args.local_dir, keep_last=config.keep_last
    )
    start = os.environ.get(SESSION_START_ENV)
    trainer = Trainer(
        config, mixer, storage, device=device, val_mixer=val_mixer, work_dir=args.work_dir,
        session_start=float(start) if start else None, allow_device_change=args.allow_device_change,
    )  # fmt: skip
    if args.verify_resume:
        verdict = trainer.verify(args.verify_steps, args.verify_tolerance, args.verify_checkpoint)
        print("EARMARK_VERIFY " + json.dumps(verdict.summary()), flush=True)
        return 0 if verdict.passed else 4
    try:
        previous = signal.signal(signal.SIGTERM, lambda _sig, _frame: trainer.request_stop())
    except ValueError:  # not the main thread
        previous = None
    try:
        result = trainer.run(info={"data": data.paths, "speakers": mixer.num_target_speakers})
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    print("EARMARK_RESULT " + json.dumps(result.summary()), flush=True)
    return 0 if result.pushed else 2


__all__ = [
    "CALIBRATION_MARGIN",
    "MIN_CUDA_CAPABILITY",
    "SESSION_START_ENV",
    "NonFiniteLossError",
    "RunLog",
    "TrainResult",
    "Trainer",
    "UnsupportedDeviceError",
    "VerifyResult",
    "WallClock",
    "build_optimizer",
    "build_parser",
    "check_device",
    "lr_factor",
    "main",
    "make_grad_scaler",
    "resolve_config",
    "train_command",
]


if __name__ == "__main__":
    sys.exit(main())
