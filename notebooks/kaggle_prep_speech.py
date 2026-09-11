# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Earmark data prep 1 of 5: LibriTTS-R speech shards (Kaggle)
#
# Builds the speech that the training mixer and Earmark-Synth read, straight from the
# Hugging Face parquet mirror of LibriTTS-R (`mythicinfinity/libritts_r`, CC BY 4.0). A
# first pass reads only the small columns to plan. The second fetches only the row groups
# that hold selected rows, resamples to 16 kHz with soxr and writes int16 shards plus a
# parquet manifest. No parquet file or tarball is ever stored.
#
# | | |
# | --- | --- |
# | **Runs on** | Kaggle notebook |
# | **Accelerator** | None (CPU). Settings: Accelerator "None", Internet "On" (needs a phone-verified Kaggle account). |
# | **Secrets** | `HF_TOKEN`, optional: a read-only Hub token (authenticated reads are rate-limited less). Add-ons, Secrets, then tick it for this notebook. `GH_TOKEN` only when the source comes from the private GitHub repository. |
# | **Inputs** | The private dataset `earmark-src` (a zip of this repository), added with "Add Input". |
# | **Outputs** | `/kaggle/working/clean100_16k`: all of train.clean.100, about 6.2 GB (247 speakers, 53.8 h). `/kaggle/working/clean360cap_16k`: train.clean.360 capped at 3 min per speaker, about 5.2 GB (about 900 speakers). `/kaggle/working/devtest_16k`: the Earmark-Synth dev/test pools, about 0.4 GB. Total about 11.8 GB, under the 20 GB notebook-output limit. |
# | **Time** | About 2 to 3 hours on 4 CPU cores. train.clean.360 streams most of its 33 GB (every speaker needs at least one row group) but decodes only the selected rows. |
#
# **Publish.** Run with "Save Version", then "Save & Run All (Commit)". When the version
# finishes, open its Output tab, choose "New Dataset", name it `earmark-speech-16k` and
# keep it private. To rebuild one part, set `EARMARK_BUILD` below (for example
# `clean360cap`); finished parts are skipped.
#
# **Leak check.** Before any audio moves, every training speaker is checked against the
# LibriTTS-R dev and test speakers (Earmark-Synth, and LibriCSS, which replays test-clean
# speakers) and the VoiceBank-DEMAND speakers. That list is stored with each training set as
# `heldout_speakers.json`, so the trainer repeats the check offline.

# %% [markdown]
# ## Settings

# %%
import os
import pathlib

EARMARK_SRC = os.environ.get("EARMARK_SRC", "")  # folder or zip of the repository; empty: auto-detect
EARMARK_GITHUB_REPO = os.environ.get("EARMARK_GITHUB_REPO", "")  # e.g. github.com/<user>/earmark
EARMARK_GIT_REF = os.environ.get("EARMARK_GIT_REF", "main")
OUT = pathlib.Path(os.environ.get("EARMARK_OUT", "/kaggle/working"))
BUILD = [p.strip() for p in os.environ.get("EARMARK_BUILD", "clean100,clean360cap,devtest").split(",") if p.strip()]
WORKERS = int(os.environ.get("EARMARK_WORKERS", str(os.cpu_count() or 4)))
INDEX_THREADS = 16  # parallel small-column reads while indexing
REVISION = None  # pin a commit of mythicinfinity/libritts_r here to freeze the source exactly
ENROL_SECONDS = 40.0  # enrolment pool per speaker, from a chapter no target comes from
CAP_360_SECONDS = 180.0  # train.clean.360: 3 min per speaker
OVERSELECT = 1.15  # plan a little extra from text-length estimates; exact durations trim it
DEVTEST_SECONDS = {"dev.clean": 90.0, "test.clean": 90.0, "dev.other": 60.0, "test.other": 60.0}
DEVTEST_ENROL_SECONDS = 20.0
OUTPUT_LIMIT_GB = 19.5

# %% [markdown]
# ## Setup: find the Earmark source, install what is missing

# %%
import json
import subprocess
import sys
import time
import zipfile


def get_secret(name: str) -> str | None:
    """A secret from Kaggle Secrets, Colab Secrets or the environment (None when unset)."""
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(name)
        if value:
            return value
    except Exception:
        pass
    try:
        from google.colab import userdata

        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name) or None


def _source_in(root: pathlib.Path) -> pathlib.Path | None:
    for hit in (root, *root.glob("*"), *root.glob("*/*"), *root.glob("*/*/*")):
        if (hit / "python" / "earmark" / "constants.py").is_file():
            return hit
    return None


def find_source() -> pathlib.Path:
    """The Earmark repository: EARMARK_SRC (folder or zip), an attached input, or a git clone."""
    candidates = [pathlib.Path(EARMARK_SRC)] if EARMARK_SRC else []
    candidates += [pathlib.Path("/kaggle/input"), pathlib.Path("/content/drive/MyDrive/earmark")]
    for root in candidates:
        if not root.exists():
            continue
        found = _source_in(root) if root.is_dir() else None
        if found:
            return found
        archives = [root] if root.suffix == ".zip" else [*root.glob("*.zip"), *root.glob("*/*.zip")]
        for archive in archives:
            dest = pathlib.Path("/tmp/earmark-src") / archive.stem
            if not dest.exists():
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(dest)
            found = _source_in(dest)
            if found:
                return found
    if EARMARK_GITHUB_REPO:
        dest = pathlib.Path("/tmp/earmark-src/clone")
        if not dest.exists():
            env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
            cmd = ["git"]
            token = get_secret("GH_TOKEN")
            if token:
                # The token reaches git through the environment, never the command line.
                env["EARMARK_GH_TOKEN"] = token
                cmd += ["-c", 'credential.helper=!f() { echo username=x-access-token; echo "password=$EARMARK_GH_TOKEN"; }; f']
            cmd += ["clone", "--depth", "1", "--branch", EARMARK_GIT_REF, f"https://{EARMARK_GITHUB_REPO}.git", str(dest)]
            subprocess.run(cmd, env=env, check=True)
        return dest
    raise FileNotFoundError(
        "Earmark source not found: attach the earmark-src dataset (a zip of the repository) "
        "or set EARMARK_GITHUB_REPO and add a GH_TOKEN secret"
    )


def ensure(module: str, requirement: str | None = None) -> None:
    """Import a module, installing it with pip first when it is missing."""
    try:
        __import__(module)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", requirement or module], check=True)


for needed in ("soxr", "soundfile", "pyarrow", "fsspec", "huggingface_hub"):
    ensure(needed)
SRC = find_source()
sys.path.insert(0, str(SRC / "python"))
os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC / "python"), os.environ.get("PYTHONPATH", "")) if p)
hf_token = get_secret("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token  # read by the Hub client; never printed
OUT.mkdir(parents=True, exist_ok=True)
print("Earmark source:", SRC, "| HF token set:", bool(hf_token), "| workers:", WORKERS)

# %%
from concurrent.futures import ProcessPoolExecutor, as_completed

from earmark import constants as C
from earmark.data import hf_parquet_select as H
from earmark.data.shards import ShardedCorpus, add_pools, check_pools_disjoint, dataset_info, finalize_dataset
from earmark.data.splits import (
    HELDOUT_SPEAKERS_FILE,
    VB_TEST_SPEAKERS,
    check_speaker_disjoint,
    save_speaker_list,
)

fs = H.hf_filesystem()
print("signal contract", C.CONTRACT_HASH, "at", C.SAMPLE_RATE, "Hz")


def dir_gb(path: pathlib.Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


# %% [markdown]
# ## Held-out speakers
# Every LibriTTS-R dev and test speaker plus the two VoiceBank-DEMAND speakers. Only the
# `speaker_id` column is read.

# %%
t0 = time.time()
HELDOUT = set(H.libritts_r_heldout_speakers(fs, revision=REVISION, max_workers=INDEX_THREADS))
HELDOUT |= set(VB_TEST_SPEAKERS)
print(f"{len(HELDOUT)} held-out speakers ({time.time() - t0:.0f} s)")


# %%
def export_split(
    split: str,
    out_dir: pathlib.Path,
    *,
    cap_seconds: float | None,
    enrol_seconds: float,
    check_heldout: bool = True,
) -> None:
    """Index, select, plan and export one LibriTTS-R split (skipped when already built)."""
    if (out_dir / "manifest.parquet").exists():
        print(f"{out_dir.name}: already built, skipping")
        return
    files = H.libritts_r_parquet_files(fs, split, revision=REVISION)
    t0 = time.time()
    index = H.read_index(fs, files, max_workers=INDEX_THREADS)
    print(f"{split}: {index.num_rows} rows in {len(files)} files indexed in {time.time() - t0:.0f} s")
    if cap_seconds is None:
        selection = H.select_all(index)
    else:
        selection = H.select_speaker_capped(
            index, cap_seconds=cap_seconds, enrol_seconds=enrol_seconds, overselect=OVERSELECT
        )
    speakers = set(selection.column("speaker").to_pylist())
    if check_heldout:
        check_speaker_disjoint(speakers, {"LibriTTS-R dev/test and VoiceBank-DEMAND": HELDOUT})
    plan = H.plan_fetch(fs, selection)
    print(f"{split}: {len(speakers)} speakers; {plan.summary()}")
    jobs = H.make_export_jobs(
        plan, selection, str(out_dir), prefix=out_dir.name, split=split,
        cap_seconds=cap_seconds, enrol_seconds=enrol_seconds,
    )
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(H.run_export_job, job): job.prefix for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            stats = future.result()
            print(
                f"  [{done}/{len(jobs)}] {futures[future]}: {int(stats['rows'])} rows, "
                f"{stats['seconds'] / 3600:.2f} h, {int(stats['skipped_cap'])} past the cap "
                f"({time.time() - t0:.0f} s)"
            )
    info = {
        "source": H.LIBRITTS_R_REPO,
        "revision": REVISION,
        "split": split,
        "cap_seconds": cap_seconds,
        "enrol_seconds": enrol_seconds,
        "license": "CC BY 4.0",
    }
    finalize_dataset(out_dir, name=out_dir.name, info=info)
    if cap_seconds is None:
        add_pools(out_dir, enrol_seconds=enrol_seconds)
    corpus = ShardedCorpus(out_dir)
    check_pools_disjoint(corpus.speaker, corpus.group, corpus.column("pool"))
    print(f"{out_dir.name}: {corpus!r}, {dir_gb(out_dir):.2f} GB")


# %% [markdown]
# ## clean100_16k: all of train.clean.100
# Pools are assigned from exact durations after writing: the enrolment chapter gives up to
# 40 s, every other chapter gives targets.

# %%
if "clean100" in BUILD:
    export_split("train.clean.100", OUT / "clean100_16k", cap_seconds=None, enrol_seconds=ENROL_SECONDS)
    save_speaker_list(
        OUT / "clean100_16k" / HELDOUT_SPEAKERS_FILE, HELDOUT,
        name="LibriTTS-R dev/test and VoiceBank-DEMAND speakers", meta={"revision": REVISION},
    )

# %% [markdown]
# ## clean360cap_16k: train.clean.360 at 3 min per speaker
# Rows are chosen from text-length estimates so that each speaker's targets come from as
# few row groups as possible; exact durations then cap each pool while decoding.

# %%
if "clean360cap" in BUILD:
    export_split(
        "train.clean.360", OUT / "clean360cap_16k", cap_seconds=CAP_360_SECONDS, enrol_seconds=ENROL_SECONDS
    )
    save_speaker_list(
        OUT / "clean360cap_16k" / HELDOUT_SPEAKERS_FILE, HELDOUT,
        name="LibriTTS-R dev/test and VoiceBank-DEMAND speakers", meta={"revision": REVISION},
    )

# %% [markdown]
# ## devtest_16k: the Earmark-Synth dev and test pools
# dev.clean and test.clean hold the targets, with enrolment from a different chapter;
# dev.other and test.other hold the interferers. These speakers are the held-out set, so
# they skip the check above, but dev and test must not share a speaker.

# %%
if "devtest" in BUILD:
    for split, seconds in DEVTEST_SECONDS.items():
        export_split(
            split, OUT / "devtest_16k" / split.replace(".", "_"),
            cap_seconds=seconds, enrol_seconds=DEVTEST_ENROL_SECONDS, check_heldout=False,
        )
    dev_speakers, test_speakers = set(), set()
    for split in DEVTEST_SECONDS:
        speakers = set(ShardedCorpus(OUT / "devtest_16k" / split.replace(".", "_")).speaker.astype(str))
        (dev_speakers if split.startswith("dev") else test_speakers).update(speakers)
    check_speaker_disjoint(dev_speakers, {"Earmark-Synth test": test_speakers})
    print(f"devtest: {len(dev_speakers)} dev and {len(test_speakers)} test speakers, disjoint")

# %% [markdown]
# ## Summary and output-size check

# %%
summary = {}
for info_path in sorted(OUT.rglob("dataset_info.json")):
    info = dataset_info(info_path.parent)
    summary[str(info_path.parent.relative_to(OUT))] = {
        "rows": info["num_rows"],
        "speakers": info["num_speakers"],
        "hours": round(info["total_seconds"] / 3600, 2),
        "gb": round(info["total_bytes"] / 1e9, 3),
    }
(OUT / "speech_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
total_gb = dir_gb(OUT)
print(f"total output: {total_gb:.2f} GB")
if total_gb > OUTPUT_LIMIT_GB:
    raise RuntimeError("over the Kaggle output limit: build fewer parts per run with EARMARK_BUILD")
