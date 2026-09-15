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
# # Earmark data prep 4 of 5: LibriCSS for Suite R (Colab)
#
# LibriCSS replays LibriSpeech test-clean utterances over loudspeakers in a meeting room
# (10 one-hour sessions, 7 channels, 16 kHz). This notebook keeps channel 0 of every
# recording as 16 kHz FLAC, reading the release zip in place, and saves the released
# timings, the per-mini-session enrolment lists and each speaker's unused LibriSpeech
# enrolment utterances. Session 0 is dev and sessions 1-9 are test. Evaluation only: the
# audio is never redistributed.
#
# | | |
# | --- | --- |
# | **Runs on** | Google Colab |
# | **Accelerator** | None (CPU runtime). About 9 GB of free disk for the downloads. |
# | **Secrets** | None needed. `GH_TOKEN` only for the private GitHub source. |
# | **Inputs** | The Earmark source: `EARMARK_SRC` pointing at a zip of the repository on Drive, or `EARMARK_GITHUB_REPO` with a `GH_TOKEN` secret. Downloads `for_release.zip` (6.4 GB, Google Drive id in `earmark.data.libricss`) and LibriSpeech test-clean (0.35 GB). |
# | **Outputs** | `/content/libricss_16k`, about 1 GB: `audio/` (channel-0 FLAC, about 0.7 GB), `enrol/` (LibriSpeech enrolment FLAC, about 0.3 GB), `sessions.parquet`, `utterances.parquet`, `suite_r_cases.parquet`, `libricss_speaker_info.jsonl`. Copied to `MyDrive/earmark/libricss_16k`; download it to the Mac as `data/libricss_16k`. Far under the 20 GB limit. |
# | **Time** | About 30 minutes, mostly the download. |
#
# If Google Drive refuses the download ("too many users"), open the file id in a browser,
# add a shortcut to your Drive, and set `LOCAL_ZIP` below to the shortcut's path.

# %% [markdown]
# ## Settings

# %%
import os
import pathlib

EARMARK_SRC = os.environ.get("EARMARK_SRC", "")  # folder or zip of the repository (on Drive); empty: auto-detect
EARMARK_GITHUB_REPO = os.environ.get("EARMARK_GITHUB_REPO", "")  # e.g. github.com/<user>/earmark
EARMARK_GIT_REF = os.environ.get("EARMARK_GIT_REF", "main")
OUT = pathlib.Path(os.environ.get("EARMARK_OUT", "/content/libricss_16k"))
TMP = pathlib.Path(os.environ.get("EARMARK_TMP", "/content/tmp"))
LOCAL_ZIP = os.environ.get("EARMARK_LIBRICSS_ZIP", "")  # an existing for_release.zip, if any
DRIVE_DIR = pathlib.Path("/content/drive/MyDrive/earmark")

# %% [markdown]
# ## Setup: Drive and the Earmark source

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


for needed in ("gdown", "soxr", "soundfile", "pyarrow"):
    ensure(needed)
SRC = find_source()
sys.path.insert(0, str(SRC / "python"))
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
print("Earmark source:", SRC, "| Colab:", ON_COLAB, "| free disk:", round(shutil.disk_usage(TMP).free / 1e9), "GB")

# %%
import urllib.request

import gdown
import pyarrow as pa
import pyarrow.parquet as pq

from earmark.data import libricss as L

# %% [markdown]
# ## Channel 0 of every recording
# The 6.4 GB zip is read member by member; nothing is extracted.

# %%
zip_path = pathlib.Path(LOCAL_ZIP) if LOCAL_ZIP else TMP / "for_release.zip"
if not (zip_path.exists() and zip_path.stat().st_size == L.LIBRICSS_ZIP_BYTES):
    gdown.download(id=L.LIBRICSS_GDRIVE_ID, output=str(zip_path), quiet=False)
if zip_path.stat().st_size != L.LIBRICSS_ZIP_BYTES:
    raise RuntimeError(f"for_release.zip is {zip_path.stat().st_size} bytes, expected {L.LIBRICSS_ZIP_BYTES}")
t0 = time.time()
sessions = L.export_channel0(zip_path, OUT, progress=lambda k, n, name: print(f"  [{k}/{n}] {name}"))
print(f"{len(sessions)} mini-sessions in {time.time() - t0:.0f} s")

# %% [markdown]
# ## Enrolment lists and LibriSpeech enrolment audio
# The released lists give, per mini-session, each speaker's LibriSpeech utterances that are
# not replayed in it. Only those FLACs are copied out of the test-clean tarball, which is
# streamed once.

# %%
info_path = OUT / "libricss_speaker_info.jsonl"
urllib.request.urlretrieve(L.LIBRICSS_SPEAKER_INFO_URL, info_path)
speaker_info = L.parse_speaker_info_jsonl(info_path.read_text())
missing = sorted({s.name for s in sessions} - set(speaker_info))
if missing:
    raise RuntimeError(f"no enrolment list for {missing}")
needed_utts = L.needed_enrolment_utts(speaker_info, sessions=sessions)
wanted = {u for ids in needed_utts.values() for u in ids}
tar_path = TMP / "test-clean.tar.gz"
if not (tar_path.exists() and tar_path.stat().st_size == L.LIBRISPEECH_TEST_CLEAN_BYTES):
    urllib.request.urlretrieve(L.LIBRISPEECH_TEST_CLEAN_URL, tar_path)
found = L.extract_librispeech_utts(tar_path, wanted, OUT / "enrol")
lost = sorted(wanted - set(found))
if lost:
    raise RuntimeError(f"{len(lost)} enrolment utterances missing from test-clean, e.g. {lost[:5]}")
print(f"{len(needed_utts)} speakers, {len(found)} enrolment utterances")

# %% [markdown]
# ## Suite R cases
# Each speaker of each mini-session is the target in turn; the others are interferers.

# %%
cases = L.suite_r_cases(sessions, speaker_info)
pq.write_table(pa.Table.from_pylist(cases), OUT / "suite_r_cases.parquet")
by_split = {split: sum(c["split"] == split for c in cases) for split in ("dev", "test")}
hours = sum(pq.read_table(OUT / L.SESSIONS_FILE).column("num_samples").to_pylist()) / 16000 / 3600
summary = {"sessions": len(sessions), "hours": round(hours, 2), "cases": by_split,
           "gb": round(sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e9, 3)}
(OUT / "libricss_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

# %% [markdown]
# ## Copy to Drive

# %%
if ON_COLAB:
    dest = DRIVE_DIR / OUT.name
    shutil.copytree(OUT, dest, dirs_exist_ok=True)
    print("copied to", dest)
