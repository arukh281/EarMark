"""Run configs and presets: the plan's runs, validation and overrides."""

from __future__ import annotations

import json

import pytest

from earmark.data.mixer import MixerConfig
from earmark.train.config import PRESETS, TrainConfig, preset


def test_the_plan_runs_exist_with_their_budgets() -> None:
    assert list(PRESETS) == ["smoke", "mini-full", "M-v1", "S-GRU", "S-SSM", "M-v2"]
    expect = {
        "smoke": ("M", 0.9, 1.0),
        "mini-full": ("M", 1.9, 2.0),
        "M-v1": ("M", 20.0, 11.0),
        "S-GRU": ("S-GRU", 8.0, 11.0),
        "S-SSM": ("S-SSM", 8.0, 11.0),
        "M-v2": ("M", 10.0, 11.0),
    }
    for name, (model, budget, limit) in expect.items():
        config = PRESETS[name]
        assert (config.name, config.model, config.budget_hours, config.time_limit_hours) == (name, model, budget, limit)
        assert config.keep_last == 3 and config.amp and config.max_steps is None
        assert config.checkpoint_minutes == (10.0 if name == "smoke" else 20.0)
    assert not PRESETS["mini-full"].use_agent_voice  # exploratory E2 compares against it
    assert all(PRESETS[n].use_agent_voice for n in PRESETS if n != "mini-full")
    assert PRESETS["M-v2"].init_from == "M-v1" and PRESETS["M-v2"].seed != PRESETS["M-v1"].seed
    assert TrainConfig(name="x", max_steps=1).time_limit_hours == 11.0  # the default session limit


@pytest.mark.parametrize("alias", ["m_v1", "M-V1", " m-v1 "])
def test_preset_lookup_ignores_case_and_separators(alias: str) -> None:
    assert preset(alias) is PRESETS["M-v1"]


def test_unknown_preset() -> None:
    with pytest.raises(KeyError, match="choose from"):
        preset("large")


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "has space", "max_steps": 1},
        {"name": "x", "model": "XL", "max_steps": 1},
        {"name": "x", "max_steps": 1, "mixer": {"no_such_field": 1}},
        {"name": "x", "max_steps": 1, "mixer": {"p_null": 2.0}},
        {"name": "x"},  # neither max_steps nor a budget
        {"name": "x", "budget_hours": 1.0, "calibrate_steps": (10, 5000)},  # beyond warm-up
        {"name": "x", "max_steps": 1, "init_from": "M v1"},
        {"name": "x", "max_steps": 1, "time_limit_hours": 0.0},
        {"name": "x", "max_steps": 1, "keep_last": 0},
    ],
)
def test_invalid_configs_are_rejected(bad: dict) -> None:
    with pytest.raises((ValueError, KeyError)):
        TrainConfig(**bad)


def test_overrides_merge_mixer_settings_and_json_lists_become_tuples() -> None:
    base = TrainConfig(name="x", max_steps=10, mixer={"batch_size": 16})
    changed = base.with_overrides(mixer={"sir_db": [0.0, 5.0]}, lr=1e-4)
    assert dict(changed.mixer) == {"batch_size": 16, "sir_db": [0.0, 5.0]} and changed.lr == 1e-4
    mixer = changed.mixer_config()
    assert isinstance(mixer, MixerConfig) and mixer.sir_db == (0.0, 5.0) and mixer.batch_size == 16
    assert changed.init_source() is None
    assert TrainConfig(name="x", max_steps=1, init_from="M-v1@1200").init_source() == ("M-v1", 1200)


def test_to_dict_is_plain_json() -> None:
    for config in PRESETS.values():
        data = json.loads(json.dumps(config.to_dict()))
        assert data["name"] == config.name and data["loss"]["asym_alpha"] == 10.0
