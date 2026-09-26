"""Checkpoint storage: the Hub backend against a fake HfApi (no network), the local
backend, and a trainer resuming from the Hub backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

from earmark.data.mixer import Mixer
from earmark.train.config import TrainConfig
from earmark.train.storage import (
    CONFIG_NAME,
    LOG_NAME,
    HubStorage,
    LocalDirStorage,
    RepositoryNotFoundError,
    StorageError,
    checkpoint_name,
    default_repo_id,
    open_storage,
    parse_checkpoint_name,
)
from earmark.train.train import RunLog, Trainer

from .conftest import FakeClock, tiny_config


class _NotFoundResponse:
    """Just enough of an HTTP response for huggingface_hub's error types.

    The hub's HTTP client changes between major versions (httpx, then httpx2), but its
    errors only read ``headers``, ``request`` and ``status_code``, so a stub works on all.
    """

    status_code = 404
    text = ""

    def __init__(self, url: str) -> None:
        self.url = url
        self.headers: dict[str, str] = {}
        self.request = SimpleNamespace(method="GET", url=url)

    def json(self) -> dict[str, str]:
        return {}


def _not_found(repo_id: str) -> Exception:
    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        return RepositoryNotFoundError(f"{repo_id} not found", response=_NotFoundResponse(url))
    except TypeError:  # older huggingface_hub: (message, response=None)
        return RepositoryNotFoundError(f"{repo_id} not found")


class HttpError(Exception):
    """An HTTP error shaped like huggingface_hub's (``.response.status_code``)."""

    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


class FakeHfApi:
    """The slice of ``HfApi`` the Hub backend uses, kept in memory."""

    def __init__(self, download_dir: Path) -> None:
        self.files: dict[str, dict[str, bytes]] = {}
        self.private: dict[str, bool] = {}
        self.commits: list[tuple[str, list[str]]] = []
        self.fail_commits = 0  # raise a network error before applying the next N commits
        self.lose_reply = 0  # apply the next N commits, then raise as if the reply was lost
        self.error: Exception | None = None  # raised once by the next commit
        self.download_dir = download_dir

    def create_repo(
        self, repo_id: str, *, private: bool | None = None, exist_ok: bool = False, repo_type: str | None = None
    ) -> None:
        if repo_id in self.files and not exist_ok:
            raise RuntimeError("repo exists")
        self.files.setdefault(repo_id, {})
        self.private.setdefault(repo_id, bool(private))

    def repo_info(self, repo_id: str, *, repo_type: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(private=self.private[repo_id])

    def list_repo_files(self, repo_id: str, *, repo_type: str | None = None) -> list[str]:
        if repo_id not in self.files:
            raise _not_found(repo_id)
        return sorted(self.files[repo_id])

    def create_commit(
        self, repo_id: str, operations: Iterable[Any], *, commit_message: str, repo_type: str | None = None
    ) -> None:
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        if self.fail_commits:
            self.fail_commits -= 1
            raise ConnectionError("transient network error")
        files = self.files[repo_id]
        summary: list[str] = []
        for op in operations:
            if isinstance(op, CommitOperationAdd):
                files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
                summary.append(f"add:{op.path_in_repo}")
            elif isinstance(op, CommitOperationDelete):
                if op.path_in_repo not in files:
                    raise KeyError(f"cannot delete missing {op.path_in_repo}")
                del files[op.path_in_repo]
                summary.append(f"delete:{op.path_in_repo}")
        self.commits.append((commit_message, summary))
        if self.lose_reply:
            self.lose_reply -= 1
            raise ConnectionError("reply lost")

    def hf_hub_download(
        self, repo_id: str, filename: str, *, repo_type: str | None = None, cache_dir: Any = None
    ) -> str:
        path = self.download_dir / repo_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[repo_id][filename])
        return str(path)

    def whoami(self) -> dict[str, str]:
        return {"name": "someone"}


def _file(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_hub_push_keeps_the_newest_three_in_one_commit(tmp_path: Path) -> None:
    api = FakeHfApi(tmp_path / "downloads")
    store = HubStorage("me/ckpts", "M-v1", api=api)
    assert store.steps() == []  # the repo does not exist yet
    store.prepare()
    assert api.private["me/ckpts"] is True
    log = tmp_path / "log.jsonl"
    for step in (10, 20, 30, 40, 50):
        log.write_text(f"log at {step}\n")
        store.push(step, _file(tmp_path / f"c{step}.pt", f"weights {step}".encode()), {LOG_NAME: log})
    assert store.steps() == [30, 40, 50]
    assert store.fetch(50).read_bytes() == b"weights 50"
    assert store.fetch_extra(LOG_NAME).read_text() == "log at 50\n"
    assert store.fetch_extra(CONFIG_NAME) is None
    assert api.commits[3][1] == [
        f"add:runs/M-v1/{checkpoint_name(40)}", f"add:runs/M-v1/{LOG_NAME}", f"delete:runs/M-v1/{checkpoint_name(10)}",
    ]  # fmt: skip
    with pytest.raises(StorageError, match="no checkpoint"):
        store.fetch(10)
    sibling = store.for_run("M-v2")
    assert sibling.steps() == [] and sibling.api is api and sibling.describe() == "hf://me/ckpts/runs/M-v2"


def test_hub_refuses_a_public_repo(tmp_path: Path) -> None:
    api = FakeHfApi(tmp_path)
    api.create_repo("me/public", private=False)
    with pytest.raises(StorageError, match="public"):
        HubStorage("me/public", "run", api=api).prepare()


def test_hub_retries_transient_errors_then_gives_up(tmp_path: Path) -> None:
    api = FakeHfApi(tmp_path)
    api.create_repo("me/r", private=True)
    waits: list[float] = []
    store = HubStorage("me/r", "run", api=api, retries=3, retry_wait_s=2.0, sleep=waits.append)
    api.fail_commits = 2
    store.push(1, _file(tmp_path / "a.pt", b"x"))
    assert store.steps() == [1] and waits == [2.0, 4.0]
    api.fail_commits = 3
    with pytest.raises(StorageError, match="3 attempts"):
        store.push(2, _file(tmp_path / "b.pt", b"y"))


@pytest.mark.parametrize("status", [401, 403, 404])
def test_hub_fails_at_once_on_permanent_http_errors(tmp_path: Path, status: int) -> None:
    api = FakeHfApi(tmp_path)
    api.create_repo("me/r", private=True)
    waits: list[float] = []
    store = HubStorage("me/r", "run", api=api, sleep=waits.append)
    api.error = HttpError(status)
    with pytest.raises(StorageError, match=f"HTTP {status}") as info:
        store.push(1, _file(tmp_path / "a.pt", b"x"))
    assert waits == []  # no backoff
    assert ("write access" in str(info.value)) == (status in (401, 403))


def test_a_retried_commit_that_already_landed_still_succeeds(tmp_path: Path) -> None:
    api = FakeHfApi(tmp_path)
    api.create_repo("me/r", private=True)
    store = HubStorage("me/r", "run", keep_last=1, api=api, sleep=lambda _s: None)
    store.push(1, _file(tmp_path / "a.pt", b"1"))
    api.lose_reply = 1  # step 2 lands and deletes step 1, but the client sees an error
    store.push(2, _file(tmp_path / "b.pt", b"2"))
    assert store.steps() == [2]


def test_hub_needs_a_token_and_never_shows_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(StorageError, match="HF_TOKEN"):
        HubStorage("me/r", "run")
    with pytest.raises(StorageError, match="HF_TOKEN"):
        default_repo_id()
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    assert "not-a-real-token" not in HubStorage("me/r", "run").describe()
    with pytest.raises(ValueError):
        HubStorage("no-slash", "run", api=object())


def test_default_repo_id_uses_the_token_owner(tmp_path: Path) -> None:
    assert default_repo_id(api=FakeHfApi(tmp_path)) == "someone/earmark-checkpoints"


def test_local_storage_rotates_and_refuses_kaggle_working(tmp_path: Path) -> None:
    store = LocalDirStorage(tmp_path / "store", "run", keep_last=3)
    log = _file(tmp_path / "log", b"l")
    for step in range(1, 5):
        store.push(step, _file(tmp_path / f"{step}.pt", bytes([step])), {LOG_NAME: log})
    assert store.steps() == [2, 3, 4] and store.fetch(4).read_bytes() == b"\x04"
    assert store.fetch_extra(LOG_NAME).read_bytes() == b"l" and store.fetch_extra("missing") is None
    assert sorted(p.name for p in store.dir.iterdir()) == [checkpoint_name(s) for s in (2, 3, 4)] + [LOG_NAME]
    with pytest.raises(StorageError, match="kaggle/working"):
        LocalDirStorage("/kaggle/working/checkpoints", "run")


def test_open_storage_and_names(tmp_path: Path) -> None:
    assert isinstance(open_storage("local", run="r", local_dir=tmp_path), LocalDirStorage)
    for kind, kwargs in (("hub", {}), ("local", {}), ("s3", {"local_dir": tmp_path})):
        with pytest.raises(ValueError):
            open_storage(kind, run="r", **kwargs)
    assert parse_checkpoint_name(checkpoint_name(123)) == 123 and parse_checkpoint_name(LOG_NAME) is None


def test_trainer_resumes_from_the_hub_backend(tmp_path: Path, mixer_for: Callable[..., Mixer]) -> None:
    api = FakeHfApi(tmp_path / "downloads")

    def trainer(config: TrainConfig, work: str, clock: FakeClock) -> Trainer:
        store = HubStorage("me/ckpts", config.name, keep_last=config.keep_last, api=api)
        return Trainer(config, mixer_for(config), store, work_dir=tmp_path / work, clock=clock, echo=None)

    first = trainer(tiny_config(time_limit_hours=350 / 3600), "w1", FakeClock(100.0))
    assert (first.run().status, first.step) == ("time_limit", 3)
    second = trainer(tiny_config(), "w2", FakeClock(1.0))
    result = second.run()
    assert (result.status, result.step, result.pushed) == ("finished", 6, True)
    store = HubStorage("me/ckpts", "tiny", api=api)
    assert api.private["me/ckpts"] and store.steps() == [3, 6]
    records = RunLog.read(store.fetch_extra(LOG_NAME))
    assert [r["how"] for r in records if r["event"] == "start"] == ["fresh", "resumed"]
    assert [(r["after"], r["step"]) for r in records if r["event"] == "trace"] == [(3, 4), (3, 5), (3, 6)]
