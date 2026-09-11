"""Append-only run log (``results/runs.jsonl``): which code, split and inference path made a number.

Every scored run appends one JSON line with the git SHA (and whether the tree was dirty), the
split, the inference path, the contract hash, the metrics and the command line. Lines are
chained: each record stores the SHA-256 of the previous line (``prev_sha256``) and a sequence
number, so :func:`verify_log` detects any edit, deletion or reordering of earlier lines.
Appends take an exclusive ``flock`` so concurrent runners cannot interleave.

The plan's rule that test numbers come from the engine only when the engine-vs-PyTorch parity
gate passed is enforced here: a ``split="test"`` record with ``inference_path="engine"`` must
carry ``parity_passed=True``; otherwise report it as ``pytorch-stream``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from earmark import constants as C

__all__ = [
    "DEFAULT_RUNS_PATH",
    "GENESIS_SHA256",
    "INFERENCE_PATHS",
    "SPLITS",
    "GitRevision",
    "RunsLogError",
    "append_run",
    "git_revision",
    "make_record",
    "read_runs",
    "to_jsonable",
    "verify_log",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_PATH = REPO_ROOT / "results" / "runs.jsonl"
GENESIS_SHA256 = "0" * 64

#: How the numbers were produced. ``engine``: C++ engine (native or WASM); ``pytorch-stream``:
#: Earmark step() path; ``pytorch-offline``: a PyTorch forward over whole utterances;
#: ``onnxruntime``: native ONNX Runtime; ``cli``: a vendor binary; ``unprocessed``: the mixture;
#: ``oracle``: a reference-informed ceiling.
INFERENCE_PATHS: frozenset[str] = frozenset(
    {"engine", "pytorch-stream", "pytorch-offline", "onnxruntime", "cli", "unprocessed", "oracle"}
)
#: ``gate`` is for reproduction gates, ``smoke`` for throwaway checks.
SPLITS: frozenset[str] = frozenset({"dev", "test", "gate", "smoke"})

_RESERVED = ("seq", "prev_sha256")


class RunsLogError(RuntimeError):
    """The run log is malformed or its hash chain is broken."""


@dataclass(frozen=True)
class GitRevision:
    sha: str
    dirty: bool | None


def git_revision(repo_root: Path = REPO_ROOT) -> GitRevision:
    """HEAD commit and dirty flag (tracked files only), read-only.

    Falls back to ``$EARMARK_GIT_SHA`` (for Kaggle/Colab copies without ``.git``), else
    ``"unknown"``.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        return GitRevision(sha=sha, dirty=bool(status.strip()))
    except (OSError, subprocess.SubprocessError):
        return GitRevision(sha=os.environ.get("EARMARK_GIT_SHA", "unknown"), dirty=None)


def to_jsonable(obj: Any) -> Any:
    """Convert numpy types, paths and dataclass-like values to strict JSON (NaN/inf become null)."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, str):
        return obj
    if hasattr(obj, "as_dict"):
        return to_jsonable(obj.as_dict())
    raise TypeError(f"cannot store {type(obj).__name__} in the run log")


def make_record(
    *,
    suite: str,
    system: str,
    split: str,
    inference_path: str,
    metrics: Mapping[str, Any],
    parity_passed: bool | None = None,
    config: Mapping[str, Any] | None = None,
    notes: str = "",
    argv: Sequence[str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build a validated record; :func:`append_run` adds ``seq`` and ``prev_sha256``."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
    if inference_path not in INFERENCE_PATHS:
        raise ValueError(f"inference_path must be one of {sorted(INFERENCE_PATHS)}, got {inference_path!r}")
    if split == "test" and inference_path == "engine" and parity_passed is not True:
        raise ValueError(
            "test numbers may come from the engine only after the engine-vs-PyTorch parity gate "
            "passed (parity_passed=True); otherwise record them as 'pytorch-stream'"
        )
    rev = git_revision(repo_root)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "suite": suite,
        "system": system,
        "split": split,
        "inference_path": inference_path,
        "parity_passed": parity_passed,
        "git_sha": rev.sha,
        "git_dirty": rev.dirty,
        "contract_hash": C.CONTRACT_HASH,
        "metrics": to_jsonable(dict(metrics)),
        "config": to_jsonable(dict(config or {})),
        "notes": notes,
        "argv": list(sys.argv if argv is None else argv),
        "host": {"platform": platform.platform(), "python": platform.python_version(), "machine": platform.machine()},
    }


def _line_hash(line: bytes) -> str:
    return hashlib.sha256(line.rstrip(b"\n")).hexdigest()


def _dumps(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def append_run(record: Mapping[str, Any], path: Path = DEFAULT_RUNS_PATH) -> dict[str, Any]:
    """Append one record (from :func:`make_record`) and return it as written."""
    if any(k in record for k in _RESERVED):
        raise ValueError(f"record must not set {_RESERVED}; append_run fills them in")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
            prev = _line_hash(lines[-1]) if lines else GENESIS_SHA256
            written = {**to_jsonable(dict(record)), "seq": len(lines), "prev_sha256": prev}
            fh.seek(0, os.SEEK_END)
            fh.write((_dumps(written) + "\n").encode("ascii"))
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return written


def read_runs(path: Path = DEFAULT_RUNS_PATH) -> list[dict[str, Any]]:
    """All records, oldest first (an absent file is an empty log)."""
    if not path.exists():
        return []
    records = []
    for lineno, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RunsLogError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
    return records


def verify_log(path: Path = DEFAULT_RUNS_PATH) -> int:
    """Check the hash chain and sequence numbers; return the number of records."""
    if not path.exists():
        return 0
    prev = GENESIS_SHA256
    count = 0
    for lineno, raw in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RunsLogError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        if rec.get("seq") != count:
            raise RunsLogError(f"{path}:{lineno}: seq {rec.get('seq')} but expected {count}")
        if rec.get("prev_sha256") != prev:
            raise RunsLogError(f"{path}:{lineno}: hash chain broken (an earlier line was changed or removed)")
        prev = _line_hash(raw)
        count += 1
    return count
