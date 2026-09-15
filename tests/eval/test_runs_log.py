"""Unit tests for the append-only run log."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from earmark import constants as C
from earmark.eval import runs_log as RL


def _record(**overrides: object) -> dict:
    kw = {
        "suite": "B",
        "system": "unprocessed",
        "split": "gate",
        "inference_path": "unprocessed",
        "metrics": {"pesq_wb": np.float64(1.97), "n": np.int64(824), "bad": float("nan")},
    }
    kw.update(overrides)
    return RL.make_record(**kw)  # type: ignore[arg-type]


def test_append_read_and_verify_chain(tmp_path: Path) -> None:
    log = tmp_path / "results" / "runs.jsonl"
    first = RL.append_run(_record(), log)
    second = RL.append_run(_record(system="gtcrn-vb", inference_path="pytorch-offline"), log)
    assert first["seq"] == 0 and first["prev_sha256"] == RL.GENESIS_SHA256
    assert second["seq"] == 1 and second["prev_sha256"] != RL.GENESIS_SHA256
    runs = RL.read_runs(log)
    assert [r["system"] for r in runs] == ["unprocessed", "gtcrn-vb"]
    assert runs[0]["metrics"] == {"pesq_wb": 1.97, "n": 824, "bad": None}  # strict JSON, NaN -> null
    assert runs[0]["contract_hash"] == C.CONTRACT_HASH
    assert {"git_sha", "git_dirty", "split", "inference_path", "timestamp_utc", "argv"} <= set(runs[0])
    assert RL.verify_log(log) == 2


def test_tampering_is_detected(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"
    for system in ("a", "b", "c"):
        RL.append_run(_record(system=system), log)
    lines = log.read_text().splitlines()
    edited = json.loads(lines[1])
    edited["metrics"]["pesq_wb"] = 3.5
    lines[1] = json.dumps(edited, sort_keys=True, separators=(",", ":"))
    log.write_text("\n".join(lines) + "\n")
    with pytest.raises(RL.RunsLogError, match="hash chain"):
        RL.verify_log(log)
    log.write_text("\n".join([lines[0], lines[2]]) + "\n")  # a deleted line
    with pytest.raises(RL.RunsLogError):
        RL.verify_log(log)


def test_engine_test_numbers_need_parity() -> None:
    with pytest.raises(ValueError, match="parity"):
        _record(system="earmark-m", split="test", inference_path="engine")
    with pytest.raises(ValueError, match="parity"):
        _record(system="earmark-m", split="test", inference_path="engine", parity_passed=False)
    ok = _record(system="earmark-m", split="test", inference_path="engine", parity_passed=True)
    assert ok["parity_passed"] is True
    assert _record(split="test", inference_path="pytorch-stream")["inference_path"] == "pytorch-stream"
    assert _record(split="dev", inference_path="engine")["split"] == "dev"


def test_rejects_unknown_split_path_and_reserved_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        _record(split="train")
    with pytest.raises(ValueError, match="inference_path"):
        _record(inference_path="magic")
    rec = _record()
    rec["seq"] = 5
    with pytest.raises(ValueError, match="seq"):
        RL.append_run(rec, tmp_path / "x.jsonl")


def test_git_revision_outside_a_repo_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EARMARK_GIT_SHA", "abc123")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    rev = RL.git_revision(tmp_path)
    assert rev.sha == "abc123" and rev.dirty is None


def test_git_revision_inside_this_repo_is_a_sha(repo_root: Path) -> None:
    rev = RL.git_revision(repo_root)
    assert rev.sha == "unknown" or len(rev.sha) == 40


def test_to_jsonable_handles_numpy_paths_and_rejects_objects() -> None:
    out = RL.to_jsonable({"a": np.arange(3), "p": Path("/x"), "f": np.float32(0.5), "inf": math.inf, "b": np.bool_(True)})
    assert out == {"a": [0, 1, 2], "p": "/x", "f": 0.5, "inf": None, "b": True}
    with pytest.raises(TypeError):
        RL.to_jsonable({"x": object()})


def test_empty_or_missing_log(tmp_path: Path) -> None:
    assert RL.read_runs(tmp_path / "none.jsonl") == []
    assert RL.verify_log(tmp_path / "none.jsonl") == 0
