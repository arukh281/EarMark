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
# # Earmark data prep 2 of 5: VCTK, MUSAN music, noise and RIRs (Kaggle)
#
# * **VCTK 0.92** (CC BY 4.0): the zip is downloaded once and its mic1 FLAC members are read
#   in place (no extraction). The VoiceBank-DEMAND speakers p232 and p257 are excluded,
#   silence is trimmed and each speaker is capped at 15 minutes.
# * **MUSAN music** (CC BY 4.0 corpus; per-track licences): the 11 GB tarball is streamed
#   through `tar` and only `musan/music` lands on disk. One 80 s excerpt per track, split
#   into `train` and `heldout` by artist.
# * **Noise and RIRs**: RIRS_NOISES (OpenSLR 28; simulated RIRs and noises for training,
#   real RIRs for evaluation only) and DEMAND channel 1 (the environments behind VoiceBank's
#   test noise are held out). Every noise clip passes the plan's excluded-class filter:
#   first on metadata, then with an AudioSet tagger, because the point-source noise files
#   carry no class labels at all.
#
# | | |
# | --- | --- |
# | **Runs on** | Kaggle notebook |
# | **Accelerator** | None (CPU). Settings: Accelerator "None", Internet "On". |
# | **Secrets** | `HF_TOKEN`, optional (downloads the AudioSet tagger `MIT/ast-finetuned-audioset-10-10-0.4593`, about 350 MB). `GH_TOKEN` only for the private GitHub source. |
# | **Inputs** | The private dataset `earmark-src` (a zip of this repository). |
# | **Outputs** | `/kaggle/working/vctk_16k`, about 2 GB (108 speakers). `/kaggle/working/musan_music_16k/{train,heldout}`, about 1.7 GB. `/kaggle/working/noise_rir_16k/`, about 1 GB: `noise_train_rirs`, `noise_train_demand` and `rir_sim` for training; `rir_real_eval` and `demand_eval` for evaluation only; `content_filter_log.json`. Total about 4.7 GB, under the 20 GB output limit. Temporary files (at most about 12 GB, mostly the 10.9 GB VCTK zip) go to `/kaggle/tmp` and are deleted. |
# | **Time** | About 2 to 3 hours (downloads, the tagger on CPU, resampling). |
#
# **Publish.** Save a version with "Save & Run All (Commit)"; from its Output tab choose
# "New Dataset", name it `earmark-vctk-musan-noise-16k`, keep it private. Set
# `EARMARK_BUILD` to rebuild one part (`vctk`, `musan` or `noise`).

# %% [markdown]
# ## Settings

# %%
import os
import pathlib

EARMARK_SRC = os.environ.get("EARMARK_SRC", "")  # folder or zip of the repository; empty: auto-detect
EARMARK_GITHUB_REPO = os.environ.get("EARMARK_GITHUB_REPO", "")  # e.g. github.com/<user>/earmark
EARMARK_GIT_REF = os.environ.get("EARMARK_GIT_REF", "main")
OUT = pathlib.Path(os.environ.get("EARMARK_OUT", "/kaggle/working"))
TMP = pathlib.Path(os.environ.get("EARMARK_TMP", "/kaggle/tmp"))
BUILD = [p.strip() for p in os.environ.get("EARMARK_BUILD", "vctk,musan,noise").split(",") if p.strip()]
WORKERS = int(os.environ.get("EARMARK_WORKERS", str(os.cpu_count() or 4)))
VCTK_CAP_SECONDS = 900.0  # 15 minutes per speaker
VCTK_ENROL_SECONDS = 40.0
MUSAN_EXCERPT_SECONDS = 80.0  # one excerpt per track keeps the music set near 1.7 GB
MUSAN_HELDOUT_FRACTION = 0.1  # share of artists held out for Earmark-Synth
RIRS_PER_ROOM = 20  # simulated RIRs kept per room (600 rooms)
NOISE_SEGMENT_SECONDS = 10.0
TAG_THRESHOLD = 0.2  # drop a clip when an excluded class scores at least this in any 10 s window
KEEP_DOWNLOADS = False
OUTPUT_LIMIT_GB = 19.5
VB_LOGFILES_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/logfiles.zip"

# %% [markdown]
# ## Setup: find the Earmark source, install what is missing

# %%
import json
import shutil
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
TMP.mkdir(parents=True, exist_ok=True)
print("Earmark source:", SRC, "| workers:", WORKERS)

# %%
from concurrent.futures import ProcessPoolExecutor, as_completed

from earmark.data import corpora as K
from earmark.data import noise_filter as NF
from earmark.data.shards import ShardedCorpus, add_pools, dataset_info, finalize_dataset
from earmark.data.splits import (
    HELDOUT_SPEAKERS_FILE,
    VB_TEST_SPEAKER_IDS,
    VB_TEST_SPEAKERS,
    check_speaker_disjoint,
    save_speaker_list,
)


def dir_gb(path: pathlib.Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def need_free_gb(path: pathlib.Path, gb: float) -> None:
    free = shutil.disk_usage(path).free / 1e9
    if free < gb:
        raise RuntimeError(f"{path} has {free:.1f} GB free; this step needs {gb:.0f} GB")


def download(url: str, dest: pathlib.Path, *, min_bytes: int) -> pathlib.Path:
    """Fetch a URL with curl unless a complete copy exists (partial files are replaced)."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest.unlink(missing_ok=True)
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10", "--silent", "--show-error", "-o", str(dest), url],
        check=True,
    )
    if dest.stat().st_size < min_bytes:
        raise RuntimeError(f"{dest.name}: only {dest.stat().st_size} bytes arrived")
    return dest


# %% [markdown]
# ## vctk_16k
# The download server answers HEAD with a small HTML page and has no range support, but a
# plain GET streams the zip; it is fetched whole and read in place. Speakers are split
# across worker processes, each reading the same zip.

# %%
if "vctk" in BUILD:
    vctk_out = OUT / "vctk_16k"
    if not (vctk_out / "manifest.parquet").exists():
        need_free_gb(TMP, 14)
        t0 = time.time()
        vctk_zip = download(K.VCTK_ZIP_URL, TMP / "VCTK-Corpus-0.92.zip", min_bytes=10_000_000_000)
        print(f"VCTK zip: {vctk_zip.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s")
        with zipfile.ZipFile(vctk_zip) as zf:
            vctk_speakers = sorted({m.speaker for m in K.vctk_members(zf.namelist())})
        print(f"{len(vctk_speakers)} speakers after excluding {VB_TEST_SPEAKER_IDS}")
        chunks = [vctk_speakers[k::WORKERS] for k in range(WORKERS)]
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = [
                pool.submit(K.export_vctk, vctk_zip, vctk_out, prefix=f"vctk-w{k}", speakers=chunk, cap_seconds=VCTK_CAP_SECONDS)
                for k, chunk in enumerate(chunks)
                if chunk
            ]
            for future in as_completed(futures):
                print("  ", future.result())
        finalize_dataset(
            vctk_out, name="vctk_16k",
            info={"source": "VCTK 0.92", "license": "CC BY 4.0", "excluded_speakers": list(VB_TEST_SPEAKER_IDS),
                  "cap_seconds": VCTK_CAP_SECONDS, "mic": "mic1"},
        )
        add_pools(vctk_out, enrol_seconds=VCTK_ENROL_SECONDS)
        check_speaker_disjoint(set(ShardedCorpus(vctk_out).speaker.astype(str)), {"VoiceBank-DEMAND": VB_TEST_SPEAKERS})
        save_speaker_list(vctk_out / HELDOUT_SPEAKERS_FILE, VB_TEST_SPEAKERS, name="VoiceBank-DEMAND speakers")
        if not KEEP_DOWNLOADS:
            vctk_zip.unlink()
    print(ShardedCorpus(vctk_out), f"{dir_gb(vctk_out):.2f} GB")

# %% [markdown]
# ## musan_music_16k
# `curl | tar` streams the tarball; only `musan/music` is written. Artists, not tracks, are
# split, so no held-out artist is ever heard in training.

# %%
if "musan" in BUILD:
    music_train = OUT / "musan_music_16k" / "train"
    music_heldout = OUT / "musan_music_16k" / "heldout"
    if not (music_train / "manifest.parquet").exists():
        need_free_gb(TMP, 8)
        music_root = TMP / "musan" / "music"
        if not music_root.exists():
            t0 = time.time()
            subprocess.run(
                ["bash", "-o", "pipefail", "-c",
                 f"curl -L --fail --retry 5 --silent --show-error '{K.MUSAN_URL}' | tar -xz -C '{TMP}' musan/music"],
                check=True,
            )
            print(f"MUSAN music extracted in {time.time() - t0:.0f} s")
        stats = K.export_musan_music(
            music_root, music_train, music_heldout,
            excerpt_seconds=MUSAN_EXCERPT_SECONDS, heldout_fraction=MUSAN_HELDOUT_FRACTION,
        )
        print("MUSAN:", stats)
        music_info = {"source": "MUSAN music", "license": "per-track (see the MUSAN annotations)",
                      "excerpt_seconds": MUSAN_EXCERPT_SECONDS, "heldout_fraction": MUSAN_HELDOUT_FRACTION}
        finalize_dataset(music_train, name="musan_music_16k_train", info=music_info)
        finalize_dataset(music_heldout, name="musan_music_16k_heldout", info=music_info)
        if not KEEP_DOWNLOADS:
            shutil.rmtree(TMP / "musan")
    for part in (music_train, music_heldout):
        print(ShardedCorpus(part), f"{dir_gb(part):.2f} GB")

# %% [markdown]
# ## noise_rir_16k
# The tagger's label names are matched against the plan's excluded classes with the same
# whole-word rule as the metadata filter, so the class list lives in one place
# (`earmark.data.noise_filter.EXCLUDED_TERMS`). Every decision is logged.

# %%
if "noise" in BUILD:
    noise_base = OUT / "noise_rir_16k"
    parts = {
        name: noise_base / name
        for name in ("noise_train_rirs", "noise_train_demand", "rir_sim", "rir_real_eval", "demand_eval")
    }
    if not all((p / "manifest.parquet").exists() for p in parts.values()):
        need_free_gb(TMP, 5)
        ensure("transformers")
        tagger = NF.AstAudioSetTagger()
        content = NF.ContentFilter(tagger, threshold=TAG_THRESHOLD)
        print("tagger labels treated as excluded:", content.excluded_labels)
        rirs_zip = download(K.RIRS_NOISES_URL, TMP / "rirs_noises.zip", min_bytes=1_300_000_000)
        t0 = time.time()
        stats = K.export_rirs_noises(
            rirs_zip, out_rir_sim=parts["rir_sim"], out_noise_train=parts["noise_train_rirs"],
            out_rir_real=parts["rir_real_eval"], rirs_per_room=RIRS_PER_ROOM,
            noise_segment_seconds=NOISE_SEGMENT_SECONDS, keep=content,
        )
        print(f"RIRS_NOISES: {stats} ({time.time() - t0:.0f} s)")
        demand_zips = {
            env: download(K.DEMAND_URL_TEMPLATE.format(env=env), TMP / f"{env}_16k.zip", min_bytes=1_000_000)
            for env in NF.DEMAND_16K_ENVIRONMENTS
        }
        stats = K.export_demand(
            demand_zips, out_train=parts["noise_train_demand"], out_heldout=parts["demand_eval"],
            held_out=NF.VB_HELDOUT_ENVIRONMENTS, segment_seconds=NOISE_SEGMENT_SECONDS, keep=content,
        )
        print("DEMAND:", stats)
        filter_card = content.summary()
        for name, path in parts.items():
            finalize_dataset(path, name=name, info={"tagger": tagger.model_id, "content_filter": filter_card})
        (noise_base / "content_filter_log.json").write_text(
            json.dumps({"summary": filter_card, "clips": content.log}, indent=2, default=str) + "\n"
        )
        print("content filter:", json.dumps(filter_card, indent=2))
        if not KEEP_DOWNLOADS:
            for path in [rirs_zip, *demand_zips.values()]:
                path.unlink(missing_ok=True)
    for path in parts.values():
        print(path.name, ShardedCorpus(path), f"{dir_gb(path):.3f} GB")

# %% [markdown]
# ## Optional: check the held-out environments against VoiceBank's own test log

# %%
vb_envs = None
try:
    vb_logs = download(VB_LOGFILES_URL, TMP / "vb_logfiles.zip", min_bytes=1000)
    with zipfile.ZipFile(vb_logs) as zf:
        log_name = next(n for n in zf.namelist() if n.endswith("log_testset.txt"))
        vb_envs = NF.vb_test_environments(zf.read(log_name).decode("utf-8", "replace"))
except (OSError, RuntimeError, StopIteration, subprocess.CalledProcessError, zipfile.BadZipFile) as err:
    print("VoiceBank log check skipped (download failed):", err)
if vb_envs is not None:
    if not vb_envs <= NF.VB_HELDOUT_ENVIRONMENTS:
        raise RuntimeError(f"VoiceBank test noise uses {sorted(vb_envs)}, not all held out")
    print("VoiceBank test noise environments, all held out:", sorted(vb_envs))

# %% [markdown]
# ## Summary and output-size check

# %%
summary = {}
for info_path in sorted(OUT.rglob("dataset_info.json")):
    info = dataset_info(info_path.parent)
    summary[str(info_path.parent.relative_to(OUT))] = {
        "rows": info["num_rows"],
        "hours": round(info["total_seconds"] / 3600, 2),
        "gb": round(info["total_bytes"] / 1e9, 3),
    }
(OUT / "vctk_musan_noise_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
total_gb = dir_gb(OUT)
print(f"total output: {total_gb:.2f} GB")
if total_gb > OUTPUT_LIMIT_GB:
    raise RuntimeError("over the Kaggle output limit: build fewer parts per run with EARMARK_BUILD")
