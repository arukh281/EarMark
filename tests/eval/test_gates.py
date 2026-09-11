"""Suite B reproduction gates on the real VoiceBank+DEMAND test set (``make gates``).

These are excluded from ``make test`` (marker ``gate``) because they need the fetched data
(``scripts/fetch_eval_data.sh``, about 326 MB) and take a few tens of seconds on a laptop CPU.
A missing dataset or checkpoint FAILS the gate rather than skipping it, so an empty gate run
can never count as a pass.
"""

from __future__ import annotations

import pytest

from earmark.eval.assets import AssetError
from earmark.eval.suites import b_vbd as B

pytestmark = pytest.mark.gate


def _utterances() -> list[B.VbdUtterance]:
    try:
        return B.list_utterances()
    except B.DataMissingError as exc:
        pytest.fail(f"VoiceBank+DEMAND test set missing: {exc}")


def _assert_gates(system_name: str, summary: dict) -> None:
    gates = summary["gates"]
    assert gates, f"no gates evaluated for {system_name}"
    report = ", ".join(f"{g['metric']}={g['measured']:.4f} (target {g['target']}+/-{g['tolerance']})" for g in gates)
    print(f"\n{system_name}: {report}")
    assert all(g["passed"] for g in gates), report
    assert summary["n_items"] == B.N_UTTERANCES
    assert summary["n_rows_with_errors"] == 0


def test_gate_noisy_input_reproduces_published_scores() -> None:
    """Noisy VB-DEMAND test: PESQ-WB 1.97 and STOI 0.921, each within 0.02."""
    assert len(_utterances()) == B.N_UTTERANCES
    summary, _ = B.run_suite(B.make_system("unprocessed"), metrics=("pesq_wb", "stoi"), n_resamples=200)
    _assert_gates("unprocessed", summary)


def test_gate_gtcrn_vb_checkpoint_reproduces_published_pesq() -> None:
    """Official GTCRN VoiceBank checkpoint: PESQ-WB 2.87 within 0.05."""
    assert len(_utterances()) == B.N_UTTERANCES
    try:
        system = B.make_system("gtcrn-vb")
    except AssetError as exc:
        pytest.fail(f"GTCRN checkpoint missing or unverified: {exc}")
    summary, _ = B.run_suite(system, metrics=("pesq_wb", "stoi"), n_resamples=200)
    _assert_gates("gtcrn-vb", summary)
