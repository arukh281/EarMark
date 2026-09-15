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
# # Earmark data prep 3 of 5: Kokoro "agent's own voice" utterances (Colab)
#
# Synthesises about 2,000 utterances with Kokoro-82M (Apache 2.0) from LibriTTS-R training
# text, one prompt per utterance, spread evenly over the 28 American and British English
# voices. Voices are split by ID: 21 `train` voices feed the training mixer, and the 7
# `test` voices are held out for Earmark-Synth's agent-voice condition. The split is the
# one `earmark.data.mixer.HeldOut` enforces.
#
# | | |
# | --- | --- |
# | **Runs on** | Google Colab |
# | **Accelerator** | GPU (T4) recommended: Runtime, Change runtime type, T4 GPU. CPU works but takes about 45 minutes. |
# | **Secrets** | Colab Secrets (the key icon): `HF_TOKEN`, optional (Hub reads); `KAGGLE_USERNAME` and `KAGGLE_KEY`, optional (publish straight to a private Kaggle dataset); `GH_TOKEN`, only for the private GitHub source. |
# | **Inputs** | The Earmark source: set `EARMARK_SRC` to a zip of the repository on Drive (for example `/content/drive/MyDrive/earmark/earmark-src.zip`), or `EARMARK_GITHUB_REPO` with a `GH_TOKEN` secret. |
# | **Outputs** | `/content/kokoro_agent_16k`, about 0.35 GB: int16 shards, `manifest.parquet` (with `voice_id`, `split`, `text`) and `dataset_info.json`. Published as the private Kaggle dataset `earmark-kokoro-agent-16k`, or zipped to `MyDrive/earmark/kokoro_agent_16k.zip` for upload by hand. Far under the 20 GB limit. |
# | **Time** | About 10 minutes on a T4. |
#
# The run is resumable: synthesis is done in chunks, and a finished chunk is skipped.

# %% [markdown]
# ## Settings

# %%
import os
import pathlib

EARMARK_SRC = os.environ.get("EARMARK_SRC", "")  # folder or zip of the repository (on Drive); empty: auto-detect
EARMARK_GITHUB_REPO = os.environ.get("EARMARK_GITHUB_REPO", "")  # e.g. github.com/<user>/earmark
EARMARK_GIT_REF = os.environ.get("EARMARK_GIT_REF", "main")
OUT = pathlib.Path(os.environ.get("EARMARK_OUT", "/content/kokoro_agent_16k"))
N_UTTERANCES = 2000
SEED = 0  # keep 0: the voice split must match earmark.data.mixer.HeldOut
CHUNKS = 4
KAGGLE_DATASET_SLUG = "earmark-kokoro-agent-16k"
DRIVE_DIR = pathlib.Path("/content/drive/MyDrive/earmark")

# %% [markdown]
# ## Setup: Drive, the Earmark source, Kokoro and espeak-ng

# %%
import json
import shutil
import subprocess
import sys
import time
import zipfile

try:
    from google.colab import drive

    drive.mount("/content/drive")
    ON_COLAB = True
except ImportError:
    ON_COLAB = False


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
    candidates += [pathlib.Path("/kaggle/input"), DRIVE_DIR]
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
        "Earmark source not found: set EARMARK_SRC to a zip of the repository on Drive, "
        "or set EARMARK_GITHUB_REPO and add a GH_TOKEN secret"
    )


def ensure(module: str, requirement: str | None = None) -> None:
    """Import a module, installing it with pip first when it is missing."""
    try:
        __import__(module)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", requirement or module], check=True)


if shutil.which("espeak-ng") is None:
    subprocess.run(["apt-get", "-qq", "-y", "install", "espeak-ng"], check=True, stdout=subprocess.DEVNULL)
for needed, requirement in (("kokoro", "kokoro>=0.9.2"), ("soxr", None), ("soundfile", None), ("pyarrow", None)):
    ensure(needed, requirement)
SRC = find_source()
sys.path.insert(0, str(SRC / "python"))
hf_token = get_secret("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token  # read by the Hub client; never printed
OUT.mkdir(parents=True, exist_ok=True)
print("Earmark source:", SRC, "| Colab:", ON_COLAB)

# %%
import torch

from earmark.data import agent_voice as AV
from earmark.data import hf_parquet_select as H
from earmark.data.mixer import HeldOut
from earmark.data.shards import ShardedCorpus, ShardWriter, finalize_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# %% [markdown]
# ## Prompts, voices and the plan
# Prompts are LibriTTS-R train.clean.100 transcripts (only the text column is read). Each
# prompt is spoken once, so train and test voices never share a sentence.

# %%
fs = H.hf_filesystem()
text_files = H.libritts_r_parquet_files(fs, "train.clean.100")
texts = H.read_index(fs, text_files, ("text_normalized",), max_workers=16).column("text_normalized").to_pylist()
prompts = AV.select_prompts(texts, N_UTTERANCES, seed=SEED)
voice_split = AV.split_voices(AV.ENGLISH_VOICES, seed=SEED)
test_voices = {v for v, s in voice_split.items() if s == "test"}
if test_voices != set(HeldOut().agent_voices):
    raise RuntimeError("this voice split differs from the one the training mixer holds out; keep SEED = 0")
plan = AV.plan_agent_utterances(voice_split, prompts, seed=SEED)
print(f"{len(plan)} utterances; test voices: {sorted(test_voices)}")

# %% [markdown]
# ## Synthesis (resumable in chunks)

# %%
tts = AV.kokoro_tts(device=DEVICE)
if not (OUT / "manifest.parquet").exists():
    for k in range(CHUNKS):
        prefix = f"kokoro-{k}"
        if (OUT / f"manifest-{prefix}.parquet").exists():
            print(prefix, "done earlier, skipping")
            continue
        for stale in OUT.glob(f"{prefix}-*.i16"):
            stale.unlink()  # left by an interrupted run
        part = plan[k::CHUNKS]
        t0 = time.time()

        def report(n: int, total: int = len(part), start: float = t0) -> None:
            if n % 100 == 0 or n == total:
                print(f"  {n}/{total} ({time.time() - start:.0f} s)")

        with ShardWriter(OUT, prefix=prefix) as writer:
            stats = AV.synthesize_plan(part, tts, writer, progress=report)
        print(prefix, stats)
    finalize_dataset(
        OUT, name="kokoro_agent_16k",
        info={"model": AV.KOKORO_REPO, "license": "Apache 2.0 (model); text from LibriTTS-R (CC BY 4.0)",
              "seed": SEED, "voice_split": voice_split, "n_prompts": len(prompts)},
    )
corpus = ShardedCorpus(OUT)
splits = corpus.column("split").astype(str)
print(corpus, {s: int((splits == s).sum()) for s in ("train", "test")})

# %% [markdown]
# ## Publish
# With `KAGGLE_USERNAME` and `KAGGLE_KEY` secrets the dataset goes straight to Kaggle
# (private). Without them a zip lands on Drive for upload by hand (Kaggle, Datasets, New
# Dataset, keep it private).

# %%
kaggle_user, kaggle_key = get_secret("KAGGLE_USERNAME"), get_secret("KAGGLE_KEY")
if kaggle_user and kaggle_key:
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = kaggle_user, kaggle_key
    ensure("kaggle")
    dataset_id = f"{kaggle_user}/{KAGGLE_DATASET_SLUG}"
    metadata = {"title": KAGGLE_DATASET_SLUG, "id": dataset_id, "licenses": [{"name": "other"}]}
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    exists = subprocess.run(["kaggle", "datasets", "status", dataset_id], capture_output=True).returncode == 0
    if exists:
        subprocess.run(["kaggle", "datasets", "version", "-p", str(OUT), "-m", "regenerated"], check=True)
    else:
        subprocess.run(["kaggle", "datasets", "create", "-p", str(OUT)], check=True)
    print("published privately as", dataset_id)
else:
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(DRIVE_DIR / "kokoro_agent_16k"), "zip", OUT)
    print("no Kaggle secrets; wrote", archive)
