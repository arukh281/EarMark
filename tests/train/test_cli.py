"""The command line end to end on CPU: dataset discovery in a Kaggle-like input tree, a
run to a local checkpoint folder, a resume, and the resume check."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from earmark.data.embeddings import StubSpeakerEncoder, compute_speaker_embeddings
from earmark.data.shards import ShardedCorpus, add_pools
from earmark.data.splits import HELDOUT_SPEAKERS_FILE, SplitLeakError, save_speaker_list
from earmark.train.pools import MissingHeldOutError, discover_training_data
from earmark.train.storage import LocalDirStorage
from earmark.train.train import main

from .conftest import SR, _write, voice


@pytest.fixture(scope="module")
def kaggle_inputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``/kaggle/input``-style tree: speech, noise and embeddings as separate datasets."""
    root = tmp_path_factory.mktemp("kaggle_input")
    rng = np.random.default_rng(5)
    speech_dir = root / "earmark-speech-16k" / "clean100_16k"
    rows: list[dict[str, Any]] = [
        {"audio": voice(rng, 120.0 + 40.0 * s, float(rng.uniform(1.0, 1.3)), 700.0 + 200.0 * s),
         "utt_id": f"libri:{401 + s}_{chapter}_{u}", "speaker": f"libri:{401 + s}", "group": str(chapter)}
        for s in range(3) for chapter in range(2) for u in range(2)
    ]  # fmt: skip
    _write(speech_dir, rows, "clean100_16k")
    add_pools(speech_dir, enrol_seconds=1.0)
    save_speaker_list(speech_dir / HELDOUT_SPEAKERS_FILE, ["libri:9001", "libri:9002"], name="heldout")
    noise = (rng.standard_normal(2 * SR) * 0.05).astype(np.float32)
    _write(
        root / "earmark-vctk-musan-noise-16k" / "noise_rir_16k" / "noise_train_demand",
        [{"audio": noise, "utt_id": "synthetic:n0", "speaker": "synthetic:n0", "group": "n0", "source": "synthetic",
          "split": "train"}],
        "noise_train_demand",
    )  # fmt: skip
    table = compute_speaker_embeddings(
        ShardedCorpus(speech_dir), StubSpeakerEncoder(), per_speaker=2, seed=0, clip_seconds=(0.5, 1.0)
    )
    table.save(root / "earmark-embeddings" / "embeddings" / "clean100_16k.npz")
    return root


def test_discovery_finds_the_datasets_and_held_out_speakers(kaggle_inputs: Path) -> None:
    messages: list[str] = []
    data = discover_training_data(kaggle_inputs, echo=messages.append)
    assert [Path(p).name for p in data.paths["speech"]] == ["clean100_16k"]
    assert [Path(p).name for p in data.paths["embeddings"]] == ["clean100_16k.npz"]
    assert [Path(p).name for p in data.paths["noise"]] == ["noise_train_demand"]
    assert data.paths["agent"] == data.paths["music"] == data.paths["rirs"] == []
    assert {"libri:9001", "libri:9002"} <= data.held_out.speakers
    assert any("kokoro_agent_16k" in m for m in messages)
    quiet = discover_training_data(kaggle_inputs, use_agent_voice=False, echo=messages.append)
    assert quiet.pools.agent is None


def test_discovery_needs_speech_noise_and_embeddings(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="earmark-embeddings"):
        discover_training_data(tmp_path, echo=None)


def test_discovery_refuses_a_speech_dataset_without_its_held_out_list(kaggle_inputs: Path, tmp_path: Path) -> None:
    """No ``heldout_speakers.json`` means dev/test disjointness cannot be proven: refuse,
    unless the caller explicitly accepts only the default hold-outs."""
    tree = tmp_path / "input"
    shutil.copytree(kaggle_inputs, tree)
    listing = tree / "earmark-speech-16k" / "clean100_16k" / HELDOUT_SPEAKERS_FILE
    listing.unlink()
    with pytest.raises(MissingHeldOutError, match=HELDOUT_SPEAKERS_FILE):
        discover_training_data(tree, echo=None)
    with pytest.raises(SplitLeakError):  # it is a leak error, so callers catching those see it
        discover_training_data(tree, echo=None)
    messages: list[str] = []
    data = discover_training_data(tree, allow_missing_heldout=True, echo=messages.append)
    assert any("--allow-missing-heldout" in m for m in messages)
    assert not {"libri:9001", "libri:9002"} & data.held_out.speakers  # only the defaults remain
    with pytest.raises(MissingHeldOutError):
        main(["--preset", "smoke", "--data-root", str(tree), "--storage", "local",
              "--local-dir", str(tmp_path / "store"), "--device", "cpu"])  # fmt: skip


def _last_json(text: str, marker: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.startswith(marker + " ")]
    assert lines, f"no {marker} line in the output"
    return json.loads(lines[-1].split(" ", 1)[1])


def test_the_command_line_trains_resumes_and_checks(
    kaggle_inputs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "store"
    base = [
        "--preset", "S-GRU", "--run-name", "cli", "--data-root", str(kaggle_inputs), "--storage", "local",
        "--local-dir", str(store), "--work-dir", str(tmp_path / "work"), "--device", "cpu", "--no-amp",
        "--batch-size", "2", "--mixer", json.dumps({"example_seconds": 1.0, "rir_max_seconds": 0.25}),
        "--checkpoint-minutes", "0.0001",  # a checkpoint after every step
    ]  # fmt: skip
    assert main([*base, "--max-steps", "2"]) == 0
    first = _last_json(capsys.readouterr().out, "EARMARK_RESULT")
    assert (first["status"], first["step"], first["pushed"]) == ("finished", 2, True)
    assert main([*base, "--max-steps", "4"]) == 0  # resumes at step 2
    second = _last_json(capsys.readouterr().out, "EARMARK_RESULT")
    assert (second["status"], second["step"]) == ("finished", 4)
    assert LocalDirStorage(store, "cli").steps() == [2, 3, 4]
    assert main([*base, "--max-steps", "4", "--verify-resume", "--verify-steps", "1"]) == 0
    verdict = _last_json(capsys.readouterr().out, "EARMARK_VERIFY")
    assert verdict["passed"] and verdict["max_abs_diff"] == 0.0 and verdict["checkpoint_step"] == 3
