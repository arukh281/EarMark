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
# # Earmark training (Kaggle, GPU T4 x2)
#
# Runs one or two presets from `earmark.train.config.PRESETS`, one per T4:
#
# - `smoke`: 1 h pipeline check;
# - `mini-full`: M, 2 h, no agent voice;
# - `M-v1`: about 20 GPU-h over two sessions;
# - `S-GRU` and `S-SSM`: about 8 GPU-h each;
# - `M-v2`: a fine-tune from M-v1, about 10 GPU-h.
#
# Each run pushes a checkpoint to a private Hugging Face model repo every 20 minutes (10 in
# the smoke run) and keeps the last 3. It saves and exits after 11 hours, and resumes from
# the Hub when this notebook runs again. Nothing is ever resumed from `/kaggle/working`.
#
# | | |
# | --- | --- |
# | **Runs on** | Kaggle notebook. For anything longer than a quick look, use Save Version, then "Save & Run All (Commit)", so the run continues with the browser closed. |
# | **Accelerator** | GPU T4 x2. Never P100: the trainer refuses GPUs below compute capability 7.5 at start-up. |
# | **Secrets** | `HF_TOKEN` (required): a Hugging Face token with write access, used to create and push to the private checkpoint repo. `GH_TOKEN` is needed only for a private GitHub source. |
# | **Inputs** | `earmark-src`, `earmark-speech-16k`, `earmark-vctk-musan-noise-16k`, `earmark-kokoro-agent-16k` and `earmark-embeddings`. |
# | **Outputs** | `/kaggle/working/earmark_logs/`, holding the JSON-lines and console logs of each run: a few MB, far under the 20 GB limit. Checkpoints (about 25 MB each for M) go only to the private Hub repo. |
#
# **Continuing a run.** Run the notebook again with the same `RUNS`, and each run picks up
# from its newest Hub checkpoint. A finished run exits at once. To start over, give it a new
# name with `EXTRA_ARGS = "--run-name smoke-2"`.
#
# **Resume check.** After the smoke run, the last cell restores a checkpoint from the Hub
# in a fresh process, trains 100 steps, and checks that the losses match the original run's
# within 1e-3.

# %% [markdown]
# ## Settings

# %%
import os
import pathlib
import time

SESSION_START = time.time()  # the 11 h session limit counts from here, not from the first step
RUNS = os.environ.get("EARMARK_RUNS", "smoke")  # comma-separated presets; the first uses GPU 0, a second GPU 1
HF_REPO = os.environ.get("EARMARK_HF_REPO", "")  # empty: <your HF account>/earmark-checkpoints, created private
EXTRA_ARGS = os.environ.get("EARMARK_TRAIN_ARGS", "")  # e.g. "--max-steps 20000"; passed to every run
VERIFY_RESUME = os.environ.get("EARMARK_VERIFY_RESUME", "auto")  # auto: after smoke; 1: always; 0: never
EARMARK_SRC = os.environ.get("EARMARK_SRC", "")  # folder or zip of the repository; empty: auto-detect
EARMARK_GITHUB_REPO = os.environ.get("EARMARK_GITHUB_REPO", "")  # e.g. github.com/<user>/earmark
EARMARK_GIT_REF = os.environ.get("EARMARK_GIT_REF", "main")
DATA_ROOT = pathlib.Path(os.environ.get("EARMARK_DATA_ROOT", "/kaggle/input"))
LOGS = pathlib.Path(os.environ.get("EARMARK_LOGS", "/kaggle/working/earmark_logs"))

# %% [markdown]
# ## Setup: find the Earmark source, read the token

# %%
import json
import shlex
import subprocess
import sys
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


for module, requirement in (("soxr", None), ("soundfile", None), ("pyarrow", None), ("huggingface_hub", None),
                            ("yaml", "pyyaml"), ("scipy", None)):
    ensure(module, requirement)
SRC = find_source()
sys.path.insert(0, str(SRC / "python"))
hf_token = get_secret("HF_TOKEN")
if not hf_token:
    raise RuntimeError("add a Hugging Face token with write access as the Kaggle secret HF_TOKEN (Add-ons, Secrets)")
os.environ["HF_TOKEN"] = hf_token  # read by the trainer's Hub client; never printed
LOGS.mkdir(parents=True, exist_ok=True)
print("Earmark source:", SRC)

# %% [markdown]
# ## GPUs and runs

# %%
import torch

from earmark.train.config import preset
from earmark.train.storage import default_repo_id
from earmark.train.train import MIN_CUDA_CAPABILITY, train_command

runs = [preset(name).name for name in RUNS.split(",") if name.strip()]
gpus = torch.cuda.device_count()
for index in range(gpus):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}, compute capability {torch.cuda.get_device_capability(index)}")
if gpus == 0:
    raise RuntimeError("no GPU: set the accelerator to GPU T4 x2")
if any(torch.cuda.get_device_capability(i) < MIN_CUDA_CAPABILITY for i in range(gpus)):
    raise RuntimeError("a GPU below compute capability 7.5 (a P100?): switch the accelerator to GPU T4 x2")
if len(runs) > gpus:
    raise RuntimeError(f"{len(runs)} runs need {len(runs)} GPUs; this session has {gpus}")
if not HF_REPO:
    HF_REPO = default_repo_id()
print("runs:", runs, "| checkpoints: hf://" + HF_REPO)

# %% [markdown]
# ## Train: one process per GPU, logs stream below every minute

# %%
env_base = dict(os.environ, PYTHONPATH=str(SRC / "python"), PYTHONUNBUFFERED="1", EARMARK_SESSION_START=str(SESSION_START))
extra = shlex.split(EXTRA_ARGS)
jobs = []
for gpu, run in enumerate(runs):
    console = LOGS / f"{run}.console.log"
    cmd = train_command(run, data_root=DATA_ROOT, repo_id=HF_REPO, work_dir=LOGS, extra=extra)
    handle = console.open("w")
    proc = subprocess.Popen(cmd, env=dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=handle, stderr=subprocess.STDOUT)
    jobs.append({"run": run, "proc": proc, "console": console, "handle": handle, "offset": 0})
    print(f"started {run} on GPU {gpu} (console log: {console})")


def pump(job: dict) -> None:
    """Print the console lines a job wrote since the last call."""
    with job["console"].open("r", errors="replace") as fh:
        fh.seek(job["offset"])
        new = fh.read()
        job["offset"] = fh.tell()
    for line in new.splitlines():
        print(f"{job['run']:>9} | {line}")


while any(job["proc"].poll() is None for job in jobs):
    time.sleep(60)
    for job in jobs:
        pump(job)
results = {}
for job in jobs:
    pump(job)
    job["handle"].close()
    marks = [ln for ln in job["console"].read_text(errors="replace").splitlines() if ln.startswith("EARMARK_RESULT ")]
    results[job["run"]] = json.loads(marks[-1].split(" ", 1)[1]) if marks else {"status": "crashed"}
    results[job["run"]]["returncode"] = job["proc"].returncode
print(json.dumps(results, indent=2))
failed = [run for run, res in results.items() if res["returncode"] != 0]
if failed:
    raise RuntimeError(f"runs {failed} failed; read their console logs in {LOGS}")

# %% [markdown]
# ## Resume check (fresh process, checkpoint fetched from the Hub)

# %%
if VERIFY_RESUME == "1" or (VERIFY_RESUME == "auto" and runs[0] == "smoke"):
    cmd = train_command(runs[0], data_root=DATA_ROOT, repo_id=HF_REPO, work_dir=LOGS / "verify", extra=extra, verify=True)
    check = subprocess.run(cmd, env=dict(env_base, CUDA_VISIBLE_DEVICES="0"), capture_output=True, text=True)
    print(check.stdout[-6000:])
    marks = [ln for ln in check.stdout.splitlines() if ln.startswith("EARMARK_VERIFY ")]
    if not marks:
        print(check.stderr[-6000:])
        raise RuntimeError("the resume check did not run; see the output above")
    verdict = json.loads(marks[-1].split(" ", 1)[1])
    print(json.dumps(verdict, indent=2))
    if not verdict["passed"]:
        raise RuntimeError("resume check FAILED: losses after resuming differ by more than the tolerance")
    print("resume check PASSED")

# %% [markdown]
# ## What next

# %%
for run, res in results.items():
    if res["status"] == "finished":
        print(f"{run}: finished at step {res['step']}; checkpoints in {res['storage']}")
    else:
        print(f"{run}: {res['status']} at step {res['step']} of {res['max_steps']}; run this notebook again to continue")
