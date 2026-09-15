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
# # Earmark data prep 5 of 5: speaker embeddings (Kaggle)
#
# Eight enrolment embeddings per training speaker from the frozen WeSpeaker ResNet34-LM
# (CC BY 4.0, 256-d, ONNX release pinned by sha256). Each comes from a 5-10 s clip built
# from that speaker's enrolment pool (chapters no target comes from), augmented with noise,
# a simulated RIR, random EQ and an Opus 16 kb/s round trip. Embeddings are L2-normalised;
# the mixer draws one per example.
#
# | | |
# | --- | --- |
# | **Runs on** | Kaggle notebook |
# | **Accelerator** | None (CPU) is enough, about 1 hour. Internet "On" (the ONNX model, 26.5 MB, comes from the Hub). |
# | **Secrets** | `HF_TOKEN`, optional. `GH_TOKEN` only for the private GitHub source. |
# | **Inputs** | `earmark-src`, `earmark-speech-16k` (clean100_16k, clean360cap_16k) and `earmark-vctk-musan-noise-16k` (vctk_16k, noise_rir_16k). |
# | **Outputs** | `/kaggle/working/embeddings/{clean100_16k,clean360cap_16k,vctk_16k}.npz`, about 10 MB in total (speakers x 8 x 256 float32), plus `embeddings_summary.json`. Far under the 20 GB limit. |
#
# **Publish.** Save a version, then Output, "New Dataset" named `earmark-embeddings`,
# private.
#
# The front end is WeSpeaker's own: Kaldi fbank with 80 mel bins, 25/10 ms, a Hamming
# window, no dither at inference, then per-utterance mean removal. The browser's enrolment
# graph must compute the same features.

# %% [markdown]
# ## Settings

# %%
import os
import pathlib

EARMARK_SRC = os.environ.get("EARMARK_SRC", "")  # folder or zip of the repository; empty: auto-detect
EARMARK_GITHUB_REPO = os.environ.get("EARMARK_GITHUB_REPO", "")  # e.g. github.com/<user>/earmark
EARMARK_GIT_REF = os.environ.get("EARMARK_GIT_REF", "main")
OUT = pathlib.Path(os.environ.get("EARMARK_OUT", "/kaggle/working/embeddings"))
SPEECH = ["clean100_16k", "clean360cap_16k", "vctk_16k"]
PER_SPEAKER = 8
SEED = 0
CLIP_SECONDS = (5.0, 10.0)

# %% [markdown]
# ## Setup: find the Earmark source, install onnxruntime

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


for needed in ("onnxruntime", "torchaudio", "soxr", "soundfile", "pyarrow", "huggingface_hub"):
    ensure(needed)
SRC = find_source()
sys.path.insert(0, str(SRC / "python"))
hf_token = get_secret("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token  # read by the Hub client; never printed
OUT.mkdir(parents=True, exist_ok=True)
print("Earmark source:", SRC)

# %%
import numpy as np

from earmark.data import embeddings as E
from earmark.data.shards import ShardedCorpus


def find_dataset(name: str) -> pathlib.Path:
    """A prepared dataset folder among the attached inputs."""
    for manifest in sorted(pathlib.Path("/kaggle/input").glob(f"**/{name}/manifest.parquet")):
        return manifest.parent
    raise FileNotFoundError(f"{name} is not among the attached inputs")


# %% [markdown]
# ## Encoder and augmentation

# %%
if not E.opus_available():
    subprocess.run(["apt-get", "-qq", "-y", "install", "ffmpeg"], check=False, stdout=subprocess.DEVNULL)
opus = E.opus_roundtrip if E.opus_available() else None
if opus is None:
    print("WARNING: ffmpeg with libopus is unavailable; enrolment clips skip the Opus round trip")
noise = ShardedCorpus.concat(
    [ShardedCorpus(find_dataset("noise_train_rirs")), ShardedCorpus(find_dataset("noise_train_demand"))]
)
rirs = ShardedCorpus(find_dataset("rir_sim"))
encoder = E.WeSpeakerOnnxEncoder.from_hub(intra_op_threads=os.cpu_count())
augment = E.EnrolAugment(E.EnrolAugmentConfig(), noise=noise, rirs=rirs, opus=opus)
print("encoder:", encoder.name, encoder.providers, "| noise clips:", len(noise), "| RIRs:", len(rirs))

# %% [markdown]
# ## Embeddings per corpus (a finished corpus is skipped)

# %%
summary = {}
for name in SPEECH:
    dest = OUT / f"{name}.npz"
    if not dest.exists():
        corpus = ShardedCorpus(find_dataset(name))
        t0 = time.time()

        def report(done: int, total: int, start: float = t0) -> None:
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total} speakers ({time.time() - start:.0f} s)")

        table = E.compute_speaker_embeddings(
            corpus, encoder, per_speaker=PER_SPEAKER, seed=SEED, clip_seconds=CLIP_SECONDS,
            augment=augment, progress=report,
        )
        table.save(dest)
    table = E.SpeakerEmbeddings.load(dest)
    emb = table.table.astype(np.float64)
    within = float(np.mean([(e @ e.T)[np.triu_indices(len(e), 1)].mean() for e in emb]))
    centroids = emb.mean(axis=1)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    between = float((centroids @ centroids.T)[np.triu_indices(len(centroids), 1)].mean())
    summary[name] = {"speakers": len(table.speakers), "per_speaker": table.per_speaker,
                     "within_cosine": round(within, 3), "between_cosine": round(between, 3)}
    print(name, summary[name])
    if within <= between:
        raise RuntimeError(f"{name}: embeddings do not separate speakers; check the encoder")
(OUT / "embeddings_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
