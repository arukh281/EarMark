"""The trainer on CPU: bit-identical resume, the session clock, checkpoint rotation and
contents, schedule calibration, init_from, refusals, the capability check and the
resume check the Kaggle notebook runs.

The fake clock advances 1 s (or 100 s) per read; the trainer reads it once at
construction, once when the loop starts and once after every step.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from earmark import constants as C
from earmark.data.mixer import Mixer
from earmark.model import build
from earmark.train.checkpoint import (
    REQUIRED_KEYS,
    ResumeError,
    check_resume_compatible,
    load_checkpoint,
    save_checkpoint,
)
from earmark.train.config import TrainConfig
from earmark.train.storage import LOG_NAME, LocalDirStorage
from earmark.train.train import (
    RunLog,
    Trainer,
    UnsupportedDeviceError,
    WallClock,
    build_optimizer,
    check_device,
    lr_factor,
)

from .conftest import FakeClock, tiny_config

MixerFactory = Callable[..., Mixer]


def make_trainer(
    config: TrainConfig,
    mixer_for: MixerFactory,
    root: Path,
    *,
    clock: Callable[[], float] | None = None,
    work: str = "work",
    **kwargs: Any,
) -> Trainer:
    storage = LocalDirStorage(root / "store", config.name, keep_last=config.keep_last)
    return Trainer(
        config, mixer_for(config), storage, work_dir=root / work, clock=clock or FakeClock(1.0), echo=None, **kwargs
    )


def rng_states() -> tuple[torch.Tensor, np.ndarray, object]:
    return torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate()


@pytest.mark.parametrize("prefetch", [0, 2], ids=["inline", "prefetch2"])
def test_resume_is_bit_identical(tmp_path: Path, mixer_for: MixerFactory, prefetch: int) -> None:
    """Inline batches and the Kaggle default (a 2-batch prefetch thread) both resume exactly."""
    straight = make_trainer(tiny_config(prefetch=prefetch), mixer_for, tmp_path / "straight")
    full = straight.run()
    rng_after = rng_states()
    assert (full.status, full.step, len(full.losses)) == ("finished", 6, 6)

    # The same run split by the session clock: 3 steps, save and exit; then resume.
    split = tmp_path / "split"
    first = make_trainer(
        tiny_config(time_limit_hours=350 / 3600, prefetch=prefetch), mixer_for, split, clock=FakeClock(100.0)
    )
    part1 = first.run()
    assert (part1.status, part1.step, part1.pushed) == ("time_limit", 3, True)
    second = make_trainer(tiny_config(prefetch=prefetch), mixer_for, split, work="work2")
    random.seed(99)  # scramble every global generator: the checkpoint must restore them
    np.random.seed(99)
    torch.manual_seed(99)
    part2 = second.run()
    assert (part2.status, part2.step) == ("finished", 6)

    assert part1.losses + part2.losses == full.losses  # exact float equality, step by step
    for name, value in straight.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[name]), name
    ours, theirs = straight.optimizer.state_dict(), second.optimizer.state_dict()
    for index, state in ours["state"].items():
        for key, value in state.items():
            assert torch.equal(value, theirs["state"][index][key]), (index, key)
    assert second.mixer.next_index == straight.mixer.next_index == 6
    after = rng_states()
    assert torch.equal(after[0], rng_after[0]) and np.array_equal(after[1], rng_after[1])
    assert after[2] == rng_after[2]


def test_checkpoints_rotate_and_hold_the_full_state(tmp_path: Path, mixer_for: MixerFactory) -> None:
    config = tiny_config(max_steps=5, checkpoint_minutes=150 / 60, keep_last=2)
    trainer = make_trainer(config, mixer_for, tmp_path, clock=FakeClock(100.0))
    assert trainer.run().status == "finished"
    assert trainer.storage.steps() == [3, 5]  # saved at 1, 3 and 5; the oldest was pruned
    payload = load_checkpoint(trainer.storage.fetch(5))
    assert REQUIRED_KEYS <= set(payload)
    assert (payload["step"], payload["status"]) == (5, "finished")
    assert payload["mixer"]["next_index"] == 5 and payload["mixer"]["seed"] == config.seed
    assert {"python", "numpy", "torch"} <= set(payload["rng"])
    assert payload["contract_hash"] == C.CONTRACT_HASH and len(payload["source_sha256"]) == 64
    assert payload["git_sha"] is None or len(payload["git_sha"]) >= 7
    assert payload["optimizer"]["state"] and payload["scaler"] == {}
    assert payload["config"]["name"] == "tiny" and payload["model_config"]["hidden"] == 128
    records = RunLog.read(trainer.storage.fetch_extra(LOG_NAME))
    assert [r["step"] for r in records if r["event"] == "checkpoint"] == [1, 3, 5]
    traces = [(r["after"], r["step"]) for r in records if r["event"] == "trace"]
    assert traces == [(1, 2), (1, 3), (3, 4), (3, 5)]
    assert [r["event"] for r in records][-2:] == ["end", "checkpoint"]


def test_grad_scaler_state_survives_a_checkpoint(tmp_path: Path, mixer_for: MixerFactory) -> None:
    """The AMP path's GradScaler (scale and growth tracker) goes through ``_payload`` and
    ``_load`` intact. CPU trainers disable AMP, so an enabled CPU scaler stands in for CUDA's."""
    config = tiny_config()
    first = make_trainer(config, mixer_for, tmp_path)
    first.use_amp = True
    first.scaler = torch.amp.GradScaler("cpu", init_scale=2.0**16, growth_interval=2000)
    state = {"scale": 96.0, "growth_factor": 3.0, "backoff_factor": 0.25, "growth_interval": 7, "_growth_tracker": 5}
    first.scaler.load_state_dict(state)
    payload = first._payload("time_limit")
    assert payload["scaler"] == state
    first.storage.push(first.step, save_checkpoint(tmp_path / "scaler.pt", payload))

    second = make_trainer(config, mixer_for, tmp_path, work="work2")
    second.use_amp = True
    second.scaler = torch.amp.GradScaler("cpu")  # defaults: scale 65536, tracker 0
    assert second.scaler.get_scale() != 96.0
    second._load(first.step)
    assert second.scaler.get_scale() == 96.0
    assert second.scaler._get_growth_tracker() == 5
    assert second.scaler.state_dict() == state


def test_resume_check_reproduces_the_logged_losses(tmp_path: Path, mixer_for: MixerFactory) -> None:
    config = tiny_config(max_steps=8, checkpoint_minutes=350 / 60, trace_steps=3)
    original = make_trainer(config, mixer_for, tmp_path, clock=FakeClock(100.0))
    original.run()  # checkpoints at 3, 7 and 8; steps 4-6 are traced after step 3
    assert original.storage.steps() == [3, 7, 8]
    result = make_trainer(config, mixer_for, tmp_path, work="verify").verify(steps=3)
    assert result.checkpoint_step == 3 and result.passed and result.max_abs_diff == 0.0
    assert original.storage.steps() == [3, 7, 8]  # the check saved nothing
    assert not (tmp_path / "verify" / "tiny" / "checkpoints").exists()

    log = tmp_path / "store" / "tiny" / LOG_NAME
    records = RunLog.read(log)
    for record in records:
        if record["event"] == "trace" and record["step"] == 5:
            record["loss"] += 0.1
    log.write_text("".join(json.dumps(r) + "\n" for r in records))
    tampered = make_trainer(config, mixer_for, tmp_path, work="verify2").verify(steps=3)
    assert not tampered.passed and tampered.max_abs_diff == pytest.approx(0.1, rel=1e-6)
    with pytest.raises(ResumeError, match="traced steps"):
        make_trainer(config, mixer_for, tmp_path, work="verify3").verify(steps=4)


def test_a_finished_run_does_nothing_until_extended(tmp_path: Path, mixer_for: MixerFactory) -> None:
    make_trainer(tiny_config(max_steps=2), mixer_for, tmp_path).run()
    again = make_trainer(tiny_config(max_steps=2), mixer_for, tmp_path, work="w2").run()
    assert (again.status, again.step, again.losses) == ("finished", 2, [])
    longer = make_trainer(tiny_config(max_steps=3), mixer_for, tmp_path, work="w3").run()
    assert (longer.step, len(longer.losses)) == (3, 1)


def test_budget_calibrates_the_schedule_length(tmp_path: Path, mixer_for: MixerFactory) -> None:
    # 1 s per step: measured at 1 step/s between session steps 2 and 6, with 8.4 s of the
    # 14.4 s budget left -> 6 + int(8.4 * 0.95) = 13 steps.
    config = tiny_config(max_steps=None, budget_hours=14.4 / 3600, calibrate_steps=(2, 6), warmup_steps=6)
    trainer = make_trainer(config, mixer_for, tmp_path)
    result = trainer.run()
    assert (result.status, result.step, result.max_steps) == ("finished", 13, 13)
    payload = load_checkpoint(trainer.storage.fetch(13))
    assert payload["max_steps"] == 13 and payload["calibration"]["steps_per_s"] == pytest.approx(1.0)
    events = [r["event"] for r in RunLog.read(trainer.storage.fetch_extra(LOG_NAME))]
    assert events.count("calibrated") == 1


def test_init_from_copies_weights_with_a_fresh_optimiser(tmp_path: Path, mixer_for: MixerFactory) -> None:
    source = make_trainer(tiny_config(name="src", max_steps=2), mixer_for, tmp_path)
    source.run()
    tuned = make_trainer(tiny_config(name="ft", max_steps=2, init_from="src", seed=1), mixer_for, tmp_path)
    assert tuned.restore() == "init_from"
    for name, value in source.model.state_dict().items():
        assert torch.equal(value, tuned.model.state_dict()[name]), name
    assert not tuned.optimizer.state and tuned.step == 0
    missing = make_trainer(tiny_config(name="ft2", max_steps=2, init_from="nope"), mixer_for, tmp_path)
    with pytest.raises(ResumeError, match="no checkpoints"):
        missing.restore()


def test_resume_refuses_another_model(tmp_path: Path, mixer_for: MixerFactory) -> None:
    make_trainer(tiny_config(max_steps=1), mixer_for, tmp_path).run()
    other = make_trainer(tiny_config(max_steps=2, model_overrides={"hidden": 64}), mixer_for, tmp_path, work="w2")
    with pytest.raises(ResumeError, match="model config"):
        other.run()


def test_resume_compatibility_rules() -> None:
    payload = {
        "contract_hash": C.CONTRACT_HASH,
        "model_config": {"hidden": 1},
        "device": {"type": "cuda", "name": "Tesla T4", "capability": [7, 5]},
        "torch_version": torch.__version__,
    }
    t4 = {"type": "cuda", "name": "Tesla T4", "capability": [7, 5]}
    a100 = {"type": "cuda", "name": "NVIDIA A100", "capability": [8, 0]}
    assert check_resume_compatible(payload, model_config={"hidden": 1}, device=t4) == []
    with pytest.raises(ResumeError, match="GPU types"):
        check_resume_compatible(payload, model_config={"hidden": 1}, device=a100)
    assert check_resume_compatible(payload, model_config={"hidden": 1}, device=a100, allow_device_change=True)
    with pytest.raises(ResumeError, match="contract"):
        check_resume_compatible({**payload, "contract_hash": "0" * 16}, model_config={"hidden": 1}, device=t4)


def test_cuda_below_compute_capability_7_5_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Tesla P100-PCIE-16GB")
    with pytest.raises(UnsupportedDeviceError, match="T4 x2"):
        check_device(torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Tesla T4")
    assert check_device(torch.device("cuda")) == {"type": "cuda", "name": "Tesla T4", "capability": [7, 5]}


def test_lr_schedule_warms_up_then_decays_to_the_floor() -> None:
    assert lr_factor(0, 10, 100, 0.05) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100, 0.05) == pytest.approx(1.0)
    assert lr_factor(55, 10, 100, 0.05) == pytest.approx(0.525)
    assert lr_factor(100, 10, 100, 0.05) == pytest.approx(0.05)
    assert lr_factor(500, 10, None, 0.05) == 1.0  # schedule length not known yet


def test_wall_clock_limits_and_intervals() -> None:
    clock = WallClock(limit_s=25.0, interval_s=15.0, clock=FakeClock(10.0))  # starts at 0
    assert not clock.expired(24.9) and clock.expired(25.0)
    assert not clock.checkpoint_due(14.0) and clock.checkpoint_due(15.0)
    clock.mark_checkpoint(15.0)
    assert not clock.checkpoint_due(29.0) and clock.checkpoint_due(30.0)
    session = WallClock(limit_s=100.0, interval_s=10.0, clock=lambda: 1_000.0, start=950.0)
    assert not session.expired(1_000.0) and session.expired(1_050.0)


def test_optimizer_groups_follow_the_s4_recipe() -> None:
    net = build("S-SSM")
    opt = build_optimizer(net, tiny_config(model="S-SSM", lr=2e-3, ssm_lr=1e-3))
    groups = {g["name"]: g for g in opt.param_groups}
    names = {id(p): n for n, p in net.named_parameters()}
    assert {names[id(p)].rsplit(".", 1)[-1] for p in groups["ssm"]["params"]} == {"log_dt", "log_a_real", "a_imag"}
    assert (groups["ssm"]["base_lr"], groups["ssm"]["weight_decay"]) == (1e-3, 0.0)
    assert "conditioner.null_embedding" in {names[id(p)] for p in groups["no_decay"]["params"]}
    assert groups["no_decay"]["weight_decay"] == 0.0 and groups["decay"]["base_lr"] == 2e-3
    assert all(p.dim() >= 2 for p in groups["decay"]["params"])
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(net.parameters()))


def test_a_non_finite_loss_skips_the_update(tmp_path: Path, mixer_for: MixerFactory) -> None:
    trainer = make_trainer(tiny_config(), mixer_for, tmp_path)
    batch = dict(trainer.mixer.batch(0))
    batch["mixture"] = batch["mixture"].clone()
    batch["mixture"][0, 100] = float("nan")
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    step = trainer._train_step(batch)
    assert math.isnan(step.total)
    assert all(torch.equal(v, trainer.model.state_dict()[k]) for k, v in before.items())
