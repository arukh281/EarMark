"""Repository tooling: the CI workflow is well formed and the Makefile targets are wired.

These run in ``make test`` so that a broken workflow or a renamed Makefile target fails
locally, before a push. They never run the targets' real work: ``make -n`` only prints.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DISK_GUARD = REPO_ROOT / "scripts" / "disk_guard.sh"
MAKE = shutil.which("make")
NODE = shutil.which("node")
needs_make = pytest.mark.skipif(MAKE is None, reason="make is not on PATH")


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _make_targets() -> set[str]:
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    return {m.group(1) for m in re.finditer(r"^([A-Za-z][\w-]*):(?!=)", text, re.MULTILINE)}


def _steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


def _run_scripts(workflow: dict[str, Any]) -> str:
    return "\n".join(step["run"] for step in _steps(workflow) if "run" in step)


def _make(*args: str) -> subprocess.CompletedProcess[str]:
    assert MAKE is not None
    return subprocess.run(
        [MAKE, *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=False
    )


# ---- the CI workflow ---------------------------------------------------------------------


def test_workflow_parses_with_the_required_jobs(workflow: dict[str, Any]) -> None:
    triggers = workflow.get("on", workflow.get(True))  # PyYAML reads a bare `on:` key as True
    assert {"push", "pull_request"} <= set(triggers)
    assert {"contract", "python", "engine", "wasm", "gates"} <= set(workflow["jobs"])
    for name, job in workflow["jobs"].items():
        assert str(job["runs-on"]).startswith("ubuntu-"), name
        assert job["steps"], name
        assert job.get("timeout-minutes"), f"{name} has no timeout"
    assert workflow["env"]["PYTHON_VERSION"] == "3.12"
    assert workflow["permissions"] == {"contents": "read"}


def test_workflow_actions_are_pinned_to_commit_shas(workflow: dict[str, Any]) -> None:
    """Tags can be moved upstream; a full 40-character commit SHA cannot."""
    uses = [step["uses"] for step in _steps(workflow) if "uses" in step]
    assert uses
    for action in uses:
        _, _, ref = action.partition("@")
        assert re.fullmatch(r"[0-9a-f]{40}", ref), f"pin {action} to a full commit SHA"
    # Each pin keeps its release tag as a trailing comment, so bumps stay reviewable.
    lines = [line for line in WORKFLOW.read_text(encoding="utf-8").splitlines() if re.match(r"\s*(-\s+)?uses:", line)]
    assert len(lines) == len(uses)
    for line in lines:
        assert re.search(r"@[0-9a-f]{40} # v\d+", line), f"add the tag as a comment: {line.strip()}"
    assert re.fullmatch(r"\d+\.\d+\.\d+", workflow["env"]["EMSDK_VERSION"])


# ---- scripts/disk_guard.sh ---------------------------------------------------------------


def _disk_guard(min_gb: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "EARMARK_MIN_FREE_GB": min_gb}
    return subprocess.run(
        [str(DISK_GUARD), str(REPO_ROOT)], env=env, capture_output=True, text=True, timeout=30, check=False
    )


@pytest.mark.parametrize(
    ("min_gb", "status"),
    [("0", 0), ("0.0", 0), ("100000", 1), ("99999.5", 1), ("abc", 2), (".", 2), ("..", 2), ("1.2.3", 2),
     ("-1", 2), ("1e3", 2), ("5 ", 2)],
)  # fmt: skip
def test_disk_guard_exit_codes(min_gb: str, status: int) -> None:
    run = _disk_guard(min_gb)
    assert run.returncode == status, (run.stdout, run.stderr)
    if status == 0:
        assert run.stdout.startswith("disk_guard: OK")
    elif status == 1:
        assert "LOW DISK" in run.stderr
    else:
        assert "must be a number" in run.stderr


def test_disk_guard_rejects_a_missing_path() -> None:
    run = subprocess.run(
        [str(DISK_GUARD), str(REPO_ROOT / "no-such-dir")], capture_output=True, text=True, timeout=30, check=False
    )
    assert run.returncode == 2 and "no such path" in run.stderr


def test_workflow_make_calls_name_real_targets(workflow: dict[str, Any]) -> None:
    targets = _make_targets()
    called = set(re.findall(r"\bmake\s+([A-Za-z][\w-]*)", _run_scripts(workflow)))
    matrix = workflow["jobs"]["engine"]["strategy"]["matrix"]["include"]
    called |= {entry["target"] for entry in matrix}
    assert called, "the workflow calls no make targets"
    assert called <= targets, f"unknown make targets: {sorted(called - targets)}"
    assert {"contract-check", "test", "goldens-check", "engine-sanitize", "gates"} <= called


def test_workflow_script_paths_exist(workflow: dict[str, Any]) -> None:
    paths = re.findall(r"(?:engine|scripts|contract)/[\w./-]+\.(?:sh|mjs|py)", _run_scripts(workflow))
    assert paths
    for path in paths:
        assert (REPO_ROOT / path).is_file(), path


def test_workflow_installs_cpu_torch_and_uploads_the_wasm(workflow: dict[str, Any]) -> None:
    assert "download.pytorch.org/whl/cpu" in workflow["env"]["TORCH_INDEX"]
    python = "\n".join(s.get("run", "") for s in workflow["jobs"]["python"]["steps"])
    assert "$TORCH_INDEX" in python and "make test" in python
    wasm = workflow["jobs"]["wasm"]["steps"]
    assert any(s.get("uses", "").startswith("mymindstorm/setup-emsdk@") for s in wasm)
    uploads = [s for s in wasm if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert uploads and uploads[0]["with"]["path"].endswith("earmark.wasm")
    sanitize = workflow["jobs"]["engine"]["strategy"]["matrix"]["include"][0]
    assert sanitize["target"] == "engine-sanitize"


# ---- Makefile wiring ---------------------------------------------------------------------


def test_makefile_default_dev_runner_exists() -> None:
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    module = re.search(r"^DEV_RUNNER \?= ([\w.]+)$", text, re.MULTILINE)
    assert module is not None
    assert (REPO_ROOT / "python" / (module.group(1).replace(".", "/") + ".py")).is_file()


@needs_make
def test_make_eval_dev_without_args_prints_usage() -> None:
    run = _make("-s", "eval-dev", "EVAL_ARGS=")
    assert run.returncode != 0
    assert "usage: make eval-dev" in run.stderr


@needs_make
def test_make_eval_dev_runs_the_dev_runner_module() -> None:
    run = _make("-n", "eval-dev", "EVAL_ARGS=--config M --manifest dev/manifest.parquet")
    assert run.returncode == 0, run.stderr
    assert "PYTHONPATH=python" in run.stdout
    assert "-m earmark.eval.dev_runner --config M --manifest dev/manifest.parquet" in run.stdout
    assert "disk_guard.sh" in run.stdout


@needs_make
def test_make_gates_fetches_assets_then_runs_only_gate_tests() -> None:
    run = _make("-n", "gates")
    assert run.returncode == 0, run.stderr
    out = run.stdout
    assert "scripts/fetch_eval_data.sh vbd gtcrn" in out
    assert "-m gate" in out
    assert out.index("fetch_eval_data.sh") < out.index("-m gate")


def test_gate_tests_are_marked_and_excluded_from_make_test() -> None:
    gate_files = [p for p in (REPO_ROOT / "tests").rglob("test_*.py") if "pytest.mark.gate" in p.read_text("utf-8")]
    assert any(p.name == "test_gates.py" for p in gate_files)
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert '-m "not slow and not gate"' in makefile


@needs_make
def test_make_wasm_builds_then_smoke_tests() -> None:
    run = _make("-n", "wasm")
    assert run.returncode == 0, run.stderr
    assert "engine/wasm/build.sh engine/build-wasm" in run.stdout
    assert "engine/wasm/smoke.mjs engine/build-wasm/earmark.wasm" in run.stdout


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
def test_wasm_smoke_script_is_valid_javascript() -> None:
    assert NODE is not None
    run = subprocess.run(
        [NODE, "--check", str(REPO_ROOT / "engine" / "wasm" / "smoke.mjs")],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert run.returncode == 0, run.stderr
