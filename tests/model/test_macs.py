"""MAC and parameter counts against the plan (within 10 %)."""

from __future__ import annotations

import pytest
import torch

from earmark import constants as C
from earmark.model import PLAN_TARGETS, build
from earmark.model.macs import ComplexityReport, complexity, count_params, dsp_macs_per_frame

TOLERANCE = 0.10


def _report(name: str) -> ComplexityReport:
    torch.manual_seed(0)
    return complexity(build(name))


@pytest.mark.parametrize("name", ["S-GRU", "S-SSM", "M", "M-256"])
def test_counts_within_ten_percent_of_plan(name: str) -> None:
    report = _report(name)
    target = PLAN_TARGETS[name]
    if target.params is not None:
        assert abs(report.params / target.params - 1) <= TOLERANCE, report.params
    if target.mmacs_per_s is not None:
        assert abs(report.mmacs_per_s / target.mmacs_per_s - 1) <= TOLERANCE, report.mmacs_per_s


def test_ssm_macs_match_gru() -> None:
    gru, ssm = _report("S-GRU"), _report("S-SSM")
    assert abs(ssm.macs_per_frame / gru.macs_per_frame - 1) <= TOLERANCE


@pytest.mark.parametrize("name", ["S-GRU", "M"])
def test_gru_body_count_matches_the_layer_weights(name: str) -> None:
    """Independent of the counter's formula: one multiply-add per GRU weight element, read
    off the model's real ``weight_ih_l*`` / ``weight_hh_l*`` tensors, plus the ``3 H``
    element-wise gate products of each layer."""
    torch.manual_seed(0)
    net = build(name)
    grus = [m for m in net.body.modules() if isinstance(m, torch.nn.GRU)]
    assert grus, f"{name} has no nn.GRU in its body"
    expected = 0
    for gru in grus:
        for layer in range(gru.num_layers):
            expected += getattr(gru, f"weight_ih_l{layer}").numel()
            expected += getattr(gru, f"weight_hh_l{layer}").numel()
            expected += 3 * gru.hidden_size
    assert complexity(net).macs_by_part["body"] == expected


def test_every_per_frame_part_is_counted() -> None:
    report = _report("S-SSM")
    expected = {"erb_enc", "df_enc", "enc_proj", "film_pre", "body", "vad_head", "film_post",
                "gain_head", "df_head"}
    assert expected <= {k for k, v in report.macs_by_part.items() if v > 0}
    assert "conditioner" not in report.macs_by_part  # computed once per embedding
    assert report.conditioning_macs > 0
    assert report.params == count_params(build("S-SSM"))
    assert sum(report.params_by_part.values()) == report.params


def test_report_fields() -> None:
    report = _report("M")
    assert report.frame_rate_hz == C.FRAME_RATE_HZ
    assert report.dsp_macs_per_frame == dsp_macs_per_frame() > 0
    assert report.total_mmacs_per_s > report.mmacs_per_s
    assert report.state_bytes == 4 * (832 + 2 * 384)
    assert set(report.as_dict()) >= {"params", "macs_per_frame", "mmacs_per_s"}
