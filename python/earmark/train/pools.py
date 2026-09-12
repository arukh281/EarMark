"""Find the prepared training datasets among the notebook inputs and open the mixer pools.

The M1 notebooks publish these folders (each with ``manifest.parquet``); Kaggle mounts
the attached datasets under ``/kaggle/input``, and the search is recursive, so the
dataset slugs and the nesting do not matter:

=========================  ===================================  ===========================
Folder                     Kaggle dataset                       Role
=========================  ===================================  ===========================
``clean100_16k``           ``earmark-speech-16k``               speech (targets, talkers)
``clean360cap_16k``        ``earmark-speech-16k``               speech
``vctk_16k``               ``earmark-vctk-musan-noise-16k``     speech (accents)
``noise_train_rirs``       ``earmark-vctk-musan-noise-16k``     noise
``noise_train_demand``     ``earmark-vctk-musan-noise-16k``     noise
``rir_sim``                ``earmark-vctk-musan-noise-16k``     simulated RIRs
``musan_music_16k/train``  ``earmark-vctk-musan-noise-16k``     music beds (TV interferer)
``kokoro_agent_16k``       ``earmark-kokoro-agent-16k``         agent-voice interferers
``<speech>.npz``           ``earmark-embeddings``               8 embeddings per speaker
=========================  ===================================  ===========================

Every speech folder's ``heldout_speakers.json`` (the dev/test speakers) is added to the
:class:`~earmark.data.mixer.HeldOut` set, so the mixer's leak check repeats offline.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from earmark.data.mixer import HeldOut, MixerPools, load_training_pools
from earmark.data.shards import MANIFEST_NAME
from earmark.data.splits import HELDOUT_SPEAKERS_FILE, SplitLeakError, load_speaker_list


class MissingHeldOutError(SplitLeakError):
    """A speech dataset has no ``heldout_speakers.json``, so training cannot prove that it
    is speaker-disjoint from the dev/test sets."""


SPEECH_DATASETS: Final[tuple[str, ...]] = ("clean100_16k", "clean360cap_16k", "vctk_16k")
NOISE_DATASETS: Final[tuple[str, ...]] = ("noise_train_rirs", "noise_train_demand")
RIR_DATASET: Final[str] = "rir_sim"
MUSIC_DATASET: Final[str] = "musan_music_16k/train"
AGENT_DATASET: Final[str] = "kokoro_agent_16k"


@dataclass(frozen=True)
class TrainingData:
    """Opened pools, what training must never contain, and where everything came from."""

    pools: MixerPools
    held_out: HeldOut
    paths: dict[str, list[str]]


def scan_inputs(root: str | Path) -> tuple[list[Path], list[Path]]:
    """All dataset folders (holding ``manifest.parquet``) and ``.npz`` files under ``root``."""
    datasets: list[Path] = []
    arrays: list[Path] = []
    for folder, _dirs, files in os.walk(root):
        if MANIFEST_NAME in files:
            datasets.append(Path(folder))
        arrays.extend(Path(folder) / f for f in files if f.endswith(".npz"))
    return sorted(datasets), sorted(arrays)


def match_dataset(folders: Sequence[Path], name: str) -> Path | None:
    """The folder whose trailing path components equal ``name`` (``a/b`` matches ``.../a/b``)."""
    parts = tuple(name.split("/"))
    hits = [f for f in folders if f.parts[-len(parts) :] == parts]
    return hits[0] if hits else None


def discover_training_data(
    root: str | Path,
    *,
    use_agent_voice: bool = True,
    use_music: bool = True,
    allow_missing_heldout: bool = False,
    echo: Callable[[str], None] | None = print,
) -> TrainingData:
    """Open every training dataset found under ``root`` (see the module docstring).

    Speech, noise and at least one embeddings table are required; RIRs, music and agent
    voices are optional (the mixer synthesises RIRs and skips missing interferer types).

    Every speech folder must hold its ``heldout_speakers.json`` (the dev/test speakers its
    prep notebook excluded), because without it the LibriTTS-R dev/test and LibriCSS
    disjointness cannot be enforced. A missing list raises :class:`MissingHeldOutError`
    unless ``allow_missing_heldout`` is set, in which case only the default hold-outs
    (:class:`~earmark.data.mixer.HeldOut`) are checked and a warning is printed.
    """
    say = echo or (lambda _msg: None)
    folders, arrays = scan_inputs(root)
    speech: list[Path] = []
    embeddings: list[Path] = []
    for name in SPEECH_DATASETS:
        folder = match_dataset(folders, name)
        table = next((a for a in arrays if a.name == f"{name}.npz"), None)
        if folder is None:
            say(f"warning: speech dataset {name} not found under {root}")
            continue
        speech.append(folder)
        if table is None:
            say(f"warning: no {name}.npz embeddings; {name} speakers serve only as interferers")
        else:
            embeddings.append(table)
    noise = [f for f in (match_dataset(folders, n) for n in NOISE_DATASETS) if f is not None]
    if not speech or not noise or not embeddings:
        raise FileNotFoundError(
            f"training data incomplete under {root}: speech={len(speech)}, noise={len(noise)}, "
            f"embeddings={len(embeddings)}. Attach earmark-speech-16k, earmark-vctk-musan-noise-16k "
            "and earmark-embeddings (plus earmark-kokoro-agent-16k for agent voices)."
        )
    rirs = match_dataset(folders, RIR_DATASET)
    music = match_dataset(folders, MUSIC_DATASET) if use_music else None
    agent = match_dataset(folders, AGENT_DATASET) if use_agent_voice else None
    if use_agent_voice and agent is None:
        say("warning: kokoro_agent_16k not found; training without agent-voice interferers")
    held: set[str] = set()
    missing: list[str] = []
    for folder in speech:
        listing = folder / HELDOUT_SPEAKERS_FILE
        if listing.is_file():
            held |= load_speaker_list(listing)
        else:
            missing.append(str(folder))
    if missing:
        problem = (
            f"no {HELDOUT_SPEAKERS_FILE} in {', '.join(missing)}, so the dev/test speakers (LibriTTS-R "
            "dev/test, LibriCSS, VoiceBank-DEMAND) cannot be checked against training"
        )
        if not allow_missing_heldout:
            raise MissingHeldOutError(
                f"{problem}. Re-run the prep notebook that built the dataset, or pass "
                "--allow-missing-heldout (allow_missing_heldout=True) to train with only the default hold-outs."
            )
        say(f"warning: {problem}; only the default hold-outs are checked (--allow-missing-heldout)")
    pools = load_training_pools(
        speech=speech, noise=noise, embeddings=embeddings, rirs=rirs, music=music, agent=agent
    )
    paths = {
        "speech": [str(p) for p in speech],
        "embeddings": [str(p) for p in embeddings],
        "noise": [str(p) for p in noise],
        "rirs": [str(rirs)] if rirs else [],
        "music": [str(music)] if music else [],
        "agent": [str(agent)] if agent else [],
    }
    return TrainingData(pools=pools, held_out=HeldOut().with_speakers(held), paths=paths)


__all__ = [
    "AGENT_DATASET",
    "MUSIC_DATASET",
    "MissingHeldOutError",
    "NOISE_DATASETS",
    "RIR_DATASET",
    "SPEECH_DATASETS",
    "TrainingData",
    "discover_training_data",
    "match_dataset",
    "scan_inputs",
]
