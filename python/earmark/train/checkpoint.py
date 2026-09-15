"""Checkpoint payloads: RNG state, provenance stamps, atomic saving and safe loading.

A checkpoint is one ``torch.save`` file holding a plain dict (see :data:`REQUIRED_KEYS`):

* ``model``, ``optimizer`` and ``scaler`` (GradScaler) state dicts;
* ``rng``: the Python, NumPy, torch CPU and (when initialised) CUDA generator states;
* ``mixer``: the mixer's generator state (``seed``, ``next_index``, config). The mixer
  draws batch ``k`` from a generator seeded by ``(seed, k)``, so this pins the batch
  sequence exactly;
* ``step``, ``max_steps`` (the resolved schedule length) and ``train_seconds``;
* provenance: ``git_sha``, ``source_sha256`` (a digest of the Python sources and the
  signal contract, meaningful even when the code came from a zip), ``contract_hash``,
  ``torch_version``, the run ``config``, the ``model_config`` and the ``device``.

Files are loaded with ``weights_only=True``, so a checkpoint fetched from the Hub cannot
run code. Everything in the payload is therefore tensors and plain Python data.
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

import earmark
from earmark import constants as C
from earmark.model.earmark_net import EarmarkConfig, EarmarkNet

#: Bumped when the payload layout changes incompatibly.
CHECKPOINT_FORMAT: Final[int] = 1

REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "format", "step", "max_steps", "train_seconds", "model", "optimizer", "scaler", "rng",
        "mixer", "config", "model_config", "git_sha", "source_sha256", "contract_hash",
        "torch_version", "device",
    }
)  # fmt: skip

#: Repository root (``python/earmark/__init__.py`` -> three levels up).
SOURCE_ROOT: Final[Path] = Path(earmark.__file__).resolve().parents[2]


class CheckpointError(RuntimeError):
    """A checkpoint is malformed or incompatible with this run."""


class ResumeError(CheckpointError):
    """Resuming would silently change the run (model, contract or GPU type differ)."""


# ------------------------------------------------------------------------------ RNG


def capture_rng_state() -> dict[str, Any]:
    """Python, NumPy, torch CPU and (if initialised) CUDA generator states, as plain data."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    version, internal, gauss_next = random.getstate()
    state: dict[str, Any] = {
        "python": {"version": version, "internal": list(internal), "gauss_next": gauss_next},
        "numpy": {
            "name": str(name),
            "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Inverse of :func:`capture_rng_state`."""
    py = state["python"]
    random.setstate((py["version"], tuple(py["internal"]), py["gauss_next"]))
    nps = state["numpy"]
    keys = nps["keys"].cpu().numpy().astype(np.uint32)
    np.random.set_state((nps["name"], keys, nps["pos"], nps["has_gauss"], nps["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    cuda = state.get("cuda")
    if cuda and torch.cuda.is_available():
        count = min(len(cuda), torch.cuda.device_count())
        for index in range(count):
            torch.cuda.set_rng_state(cuda[index].cpu(), device=index)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------------ provenance


def git_sha(root: Path = SOURCE_ROOT) -> str | None:
    """The source's git commit, or ``None`` when unknown.

    Order: the ``EARMARK_GIT_SHA`` environment variable, a ``GIT_SHA`` file at the source
    root (put one in the zip that goes to Kaggle), then ``git rev-parse HEAD`` with a
    ``-dirty`` suffix when tracked files have uncommitted changes.
    """
    env = os.environ.get("EARMARK_GIT_SHA", "").strip()
    if env:
        return env
    stamp = root / "GIT_SHA"
    if stamp.is_file():
        text = stamp.read_text().strip()
        if text:
            return text.split()[0]
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    if not head:
        return None
    return f"{head}-dirty" if dirty else head


def source_digest(root: Path = SOURCE_ROOT) -> str:
    """sha256 over the Python package sources and the signal contract (paths and bytes)."""
    digest = hashlib.sha256()
    files = sorted((root / "python" / "earmark").rglob("*.py"))
    contract = root / "contract" / "signal.yaml"
    if contract.is_file():
        files.append(contract)
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def device_info(device: torch.device) -> dict[str, Any]:
    """Device type, name and (CUDA) compute capability."""
    if device.type == "cuda":
        return {
            "type": "cuda",
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    return {"type": device.type, "name": platform.machine() or device.type, "capability": None}


def provenance(root: Path = SOURCE_ROOT) -> dict[str, Any]:
    """Stamps recorded in every checkpoint and log header."""
    return {
        "git_sha": git_sha(root),
        "source_sha256": source_digest(root),
        "contract_hash": C.CONTRACT_HASH,
        # str(): torch.__version__ is a TorchVersion object, which weights_only loading refuses.
        "torch_version": str(torch.__version__),
        "earmark_version": earmark.__version__,
    }


# ---------------------------------------------------------------------- save and load


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` atomically (temporary file, then rename)."""
    missing = REQUIRED_KEYS - set(payload)
    if missing:
        raise CheckpointError(f"checkpoint payload is missing {sorted(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint safely (``weights_only=True``) and check its format."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"{path} is not an Earmark checkpoint of format {CHECKPOINT_FORMAT}")
    missing = REQUIRED_KEYS - set(payload)
    if missing:
        raise CheckpointError(f"{path} is missing {sorted(missing)}")
    return payload


def model_from_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> EarmarkNet:
    """Rebuild the trained :class:`EarmarkNet` from a checkpoint (for export and scoring)."""
    payload = load_checkpoint(path, map_location=device)
    if payload["contract_hash"] != C.CONTRACT_HASH:
        raise CheckpointError(
            f"checkpoint contract {payload['contract_hash']} differs from this code's {C.CONTRACT_HASH}"
        )
    config = EarmarkConfig(**payload["model_config"])
    net = EarmarkNet(config)
    net.load_state_dict(payload["model"])
    return net.to(device).train(False)


def check_resume_compatible(
    payload: Mapping[str, Any],
    *,
    model_config: Mapping[str, Any],
    device: Mapping[str, Any],
    allow_device_change: bool = False,
) -> list[str]:
    """Raise :class:`ResumeError` if resuming ``payload`` would change the run.

    Refuses a different signal contract, a different model config, and (unless
    ``allow_device_change``) a different device type or GPU model: the plan never resumes
    across GPU types. Returns warnings for differences that are allowed.
    """
    if payload["contract_hash"] != C.CONTRACT_HASH:
        raise ResumeError(
            f"checkpoint was trained under contract {payload['contract_hash']}, this code has {C.CONTRACT_HASH}"
        )
    if dict(payload["model_config"]) != dict(model_config):
        raise ResumeError(f"checkpoint model config {payload['model_config']} differs from {dict(model_config)}")
    saved = payload["device"]
    warnings: list[str] = []
    same = saved.get("type") == device.get("type") and (
        saved.get("type") != "cuda" or saved.get("name") == device.get("name")
    )
    if not same:
        message = f"checkpoint was saved on {saved.get('type')} {saved.get('name')}, this is {device.get('type')} {device.get('name')}"
        if not allow_device_change:
            raise ResumeError(message + "; the plan never resumes across GPU types (pass --allow-device-change to override)")
        warnings.append(message)
    if payload.get("torch_version") != torch.__version__:
        warnings.append(f"checkpoint used torch {payload.get('torch_version')}, this is {torch.__version__}")
    return warnings


__all__ = [
    "CHECKPOINT_FORMAT",
    "REQUIRED_KEYS",
    "SOURCE_ROOT",
    "CheckpointError",
    "ResumeError",
    "capture_rng_state",
    "check_resume_compatible",
    "device_info",
    "git_sha",
    "load_checkpoint",
    "model_from_checkpoint",
    "provenance",
    "restore_rng_state",
    "save_checkpoint",
    "seed_everything",
    "source_digest",
]
