"""Where checkpoints live: a private Hugging Face model repo, or a local directory.

Both backends implement :class:`CheckpointStorage` and use one layout per run::

    runs/<run>/ckpt-000012000.pt      (Hub)   or   <root>/<run>/ckpt-000012000.pt   (local)
    runs/<run>/train_log.jsonl                     <root>/<run>/train_log.jsonl
    runs/<run>/run_config.json                     <root>/<run>/run_config.json

A push adds the new checkpoint and the extra files and deletes checkpoints beyond the
newest ``keep_last``; on the Hub that is one commit. The Hub backend refuses a public
repository, reads the token from ``HF_TOKEN`` and never prints it, and retries network
calls with backoff. Kaggle runs use the Hub backend; the local one is for tests and
workstation runs, and it refuses ``/kaggle/working`` (the plan never resumes from there).

The Hub keeps deleted files in git history, so a long run's repo grows by the size of
every checkpoint ever pushed (about 25 MB each for M, 3 per hour). Squash the history
afterwards with ``HfApi().super_squash_history(repo_id)`` if storage matters.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, TypeVar

try:  # huggingface_hub >= 0.24
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:  # pragma: no cover - older hub releases on some notebook images
    from huggingface_hub.utils import RepositoryNotFoundError

_T = TypeVar("_T")

CHECKPOINT_RE: Final[re.Pattern[str]] = re.compile(r"^ckpt-(\d{9})\.pt$")
LOG_NAME: Final[str] = "train_log.jsonl"
CONFIG_NAME: Final[str] = "run_config.json"
#: Repo name used when only the account is known.
DEFAULT_REPO_NAME: Final[str] = "earmark-checkpoints"
KAGGLE_WORKING: Final[str] = "/kaggle/working"


class StorageError(RuntimeError):
    """A storage operation failed (after retries) or is not allowed."""


#: HTTP statuses that retrying cannot fix: bad request, bad or read-only token, missing
#: repo or file, payload too large, unprocessable. Timeouts, 5xx and commit races retry.
PERMANENT_HTTP_STATUS: Final[frozenset[int]] = frozenset({400, 401, 403, 404, 413, 422})


def _http_status(err: BaseException) -> int | None:
    """Status code of a huggingface_hub HTTP error (``err.response.status_code``), if any."""
    status = getattr(getattr(err, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def checkpoint_name(step: int) -> str:
    """File name of the checkpoint at ``step``."""
    return f"ckpt-{step:09d}.pt"


def parse_checkpoint_name(name: str) -> int | None:
    """Step of a checkpoint file name, or ``None`` for other files."""
    match = CHECKPOINT_RE.match(name)
    return int(match.group(1)) if match else None


class CheckpointStorage(Protocol):
    """What the trainer needs from a checkpoint store."""

    run: str
    keep_last: int

    def describe(self) -> str: ...

    def prepare(self) -> None:
        """Create the location and check it is allowed (for example, that a repo is private)."""
        ...

    def steps(self) -> list[int]:
        """Steps of the stored checkpoints, ascending."""
        ...

    def push(self, step: int, checkpoint: Path, extras: Mapping[str, Path] | None = None) -> None:
        """Store a checkpoint (plus extra files) and prune to the newest ``keep_last``."""
        ...

    def fetch(self, step: int) -> Path:
        """Local path of the checkpoint at ``step``."""
        ...

    def fetch_extra(self, name: str) -> Path | None:
        """Local path of an extra file (for example the log), or ``None`` if absent."""
        ...

    def for_run(self, run: str) -> CheckpointStorage:
        """The same location for another run (used by ``init_from``)."""
        ...


def _stale(existing: Sequence[int], new: int, keep_last: int) -> list[int]:
    keep = set(sorted(set(existing) | {new})[-keep_last:])
    return [step for step in existing if step not in keep]


# ----------------------------------------------------------------------------- local


class LocalDirStorage:
    """Checkpoints in ``<root>/<run>/`` on the local filesystem."""

    def __init__(self, root: str | Path, run: str, *, keep_last: int = 3) -> None:
        resolved = Path(root).expanduser().resolve()
        if resolved.as_posix().startswith(KAGGLE_WORKING):
            raise StorageError(
                f"{resolved} is under {KAGGLE_WORKING}, which does not survive a session; "
                "use the Hub backend on Kaggle"
            )
        if keep_last < 1:
            raise ValueError("keep_last must be at least 1")
        self.root = resolved
        self.run = run
        self.keep_last = keep_last
        self.dir = resolved / run

    def describe(self) -> str:
        return str(self.dir)

    def prepare(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def steps(self) -> list[int]:
        if not self.dir.is_dir():
            return []
        found = (parse_checkpoint_name(p.name) for p in self.dir.iterdir())
        return sorted(step for step in found if step is not None)

    def _copy(self, source: Path, name: str) -> None:
        tmp = self.dir / f".{name}.tmp"
        shutil.copyfile(source, tmp)
        os.replace(tmp, self.dir / name)

    def push(self, step: int, checkpoint: Path, extras: Mapping[str, Path] | None = None) -> None:
        self.prepare()
        existing = self.steps()
        self._copy(Path(checkpoint), checkpoint_name(step))
        for name, path in (extras or {}).items():
            self._copy(Path(path), name)
        for old in _stale(existing, step, self.keep_last):
            (self.dir / checkpoint_name(old)).unlink(missing_ok=True)

    def fetch(self, step: int) -> Path:
        path = self.dir / checkpoint_name(step)
        if not path.is_file():
            raise StorageError(f"no checkpoint for step {step} in {self.dir}")
        return path

    def fetch_extra(self, name: str) -> Path | None:
        path = self.dir / name
        return path if path.is_file() else None

    def for_run(self, run: str) -> LocalDirStorage:
        return LocalDirStorage(self.root, run, keep_last=self.keep_last)


# ------------------------------------------------------------------------------- hub


class HubStorage:
    """Checkpoints under ``runs/<run>/`` in a private Hugging Face model repository.

    ``api`` defaults to ``huggingface_hub.HfApi(token=...)`` with the token from the
    argument or the ``HF_TOKEN`` environment variable (a Kaggle secret); tests pass a fake.
    """

    def __init__(
        self,
        repo_id: str,
        run: str,
        *,
        keep_last: int = 3,
        token: str | None = None,
        api: Any | None = None,
        cache_dir: str | Path | None = None,
        retries: int = 5,
        retry_wait_s: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if "/" not in repo_id:
            raise ValueError(f"repo_id must be '<user>/<name>', got {repo_id!r}")
        if keep_last < 1 or retries < 1:
            raise ValueError("keep_last and retries must be at least 1")
        if api is None:
            token = token or os.environ.get("HF_TOKEN")
            if not token:
                raise StorageError(
                    "HF_TOKEN is not set: add a write token as the Kaggle secret HF_TOKEN "
                    "(the notebook exports it to the environment)"
                )
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        self.api = api
        self.repo_id = repo_id
        self.run = run
        self.keep_last = keep_last
        self.prefix = f"runs/{run}"
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.retries = retries
        self.retry_wait_s = retry_wait_s
        self._sleep = sleep

    def describe(self) -> str:
        return f"hf://{self.repo_id}/{self.prefix}"

    def _retry(self, what: str, fn: Callable[[], _T], *, fatal: tuple[type[BaseException], ...] = ()) -> _T:
        last: BaseException | None = None
        for attempt in range(self.retries):
            try:
                return fn()
            except fatal:
                raise
            except Exception as err:  # network and server errors: back off and retry
                status = _http_status(err)
                if status in PERMANENT_HTTP_STATUS:  # retrying cannot help; fail at once
                    hint = " (check that HF_TOKEN has write access to this repo)" if status in (401, 403) else ""
                    raise StorageError(f"{what} on {self.repo_id} failed with HTTP {status}{hint}: {err}") from err
                last = err
                if attempt + 1 < self.retries:
                    self._sleep(min(300.0, self.retry_wait_s * 2**attempt))
        raise StorageError(f"{what} on {self.repo_id} failed after {self.retries} attempts: {last}") from last

    def prepare(self) -> None:
        self._retry(
            "create_repo",
            lambda: self.api.create_repo(self.repo_id, private=True, exist_ok=True, repo_type="model"),
        )
        info = self._retry("repo_info", lambda: self.api.repo_info(self.repo_id, repo_type="model"))
        if not getattr(info, "private", False):
            raise StorageError(f"{self.repo_id} is public; checkpoints are only pushed to a private repo")

    def _files(self) -> list[str]:
        try:
            files = self._retry(
                "list_repo_files",
                lambda: self.api.list_repo_files(self.repo_id, repo_type="model"),
                fatal=(RepositoryNotFoundError,),
            )
        except RepositoryNotFoundError:
            return []
        return list(files)

    def _steps_in(self, files: Sequence[str]) -> list[int]:
        head = self.prefix + "/"
        found = (parse_checkpoint_name(f[len(head) :]) for f in files if f.startswith(head))
        return sorted(step for step in found if step is not None)

    def steps(self) -> list[int]:
        return self._steps_in(self._files())

    def push(self, step: int, checkpoint: Path, extras: Mapping[str, Path] | None = None) -> None:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        def attempt() -> object:
            # Re-list on every attempt: a commit that succeeded but whose reply was lost has
            # already deleted the stale files, and deleting them again would fail forever.
            existing = self._steps_in(self._files())
            ops: list[Any] = [
                CommitOperationAdd(
                    path_in_repo=f"{self.prefix}/{checkpoint_name(step)}", path_or_fileobj=str(checkpoint)
                )
            ]
            ops += [
                CommitOperationAdd(path_in_repo=f"{self.prefix}/{name}", path_or_fileobj=str(path))
                for name, path in (extras or {}).items()
            ]
            ops += [
                CommitOperationDelete(path_in_repo=f"{self.prefix}/{checkpoint_name(old)}")
                for old in _stale(existing, step, self.keep_last)
            ]
            return self.api.create_commit(
                self.repo_id, ops, commit_message=f"{self.run}: checkpoint at step {step}", repo_type="model"
            )

        self._retry("create_commit", attempt)

    def _download(self, path_in_repo: str) -> Path:
        local = self._retry(
            "hf_hub_download",
            lambda: self.api.hf_hub_download(
                self.repo_id, path_in_repo, repo_type="model", cache_dir=self.cache_dir
            ),
        )
        return Path(str(local))

    def fetch(self, step: int) -> Path:
        if step not in self.steps():
            raise StorageError(f"no checkpoint for step {step} in {self.describe()}")
        return self._download(f"{self.prefix}/{checkpoint_name(step)}")

    def fetch_extra(self, name: str) -> Path | None:
        path = f"{self.prefix}/{name}"
        if path not in self._files():
            return None
        return self._download(path)

    def for_run(self, run: str) -> HubStorage:
        return HubStorage(
            self.repo_id, run, keep_last=self.keep_last, api=self.api, cache_dir=self.cache_dir,
            retries=self.retries, retry_wait_s=self.retry_wait_s, sleep=self._sleep,
        )  # fmt: skip


# ----------------------------------------------------------------------------- helpers


def default_repo_id(token: str | None = None, api: Any | None = None) -> str:
    """``<account>/earmark-checkpoints`` for the account that owns the token."""
    if api is None:
        token = token or os.environ.get("HF_TOKEN")
        if not token:
            raise StorageError("HF_TOKEN is not set")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    return f"{api.whoami()['name']}/{DEFAULT_REPO_NAME}"


def open_storage(
    kind: str,
    *,
    run: str,
    repo_id: str | None = None,
    local_dir: str | Path | None = None,
    keep_last: int = 3,
) -> CheckpointStorage:
    """Build a backend: ``kind`` is ``"hub"`` (needs ``repo_id``) or ``"local"`` (needs ``local_dir``)."""
    if kind == "hub":
        if not repo_id:
            raise ValueError("the hub backend needs --repo-id (for example <user>/earmark-checkpoints)")
        return HubStorage(repo_id, run, keep_last=keep_last)
    if kind == "local":
        if local_dir is None:
            raise ValueError("the local backend needs --local-dir")
        return LocalDirStorage(local_dir, run, keep_last=keep_last)
    raise ValueError(f"unknown storage {kind!r}; use 'hub' or 'local'")


__all__ = [
    "CHECKPOINT_RE",
    "CONFIG_NAME",
    "DEFAULT_REPO_NAME",
    "LOG_NAME",
    "CheckpointStorage",
    "HubStorage",
    "LocalDirStorage",
    "StorageError",
    "checkpoint_name",
    "default_repo_id",
    "open_storage",
    "parse_checkpoint_name",
]
