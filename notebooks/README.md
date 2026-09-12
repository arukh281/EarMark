# Notebooks

Every notebook here is a [jupytext](https://jupytext.readthedocs.io/) percent-format `.py`
file, so it diffs and reviews like code. Convert one for upload with
`uvx jupytext --to ipynb notebooks/<name>.py`. Never commit `.ipynb` files or cell outputs.

Run them in this order. Each one publishes a **private** Kaggle dataset, and later notebooks
attach those datasets as inputs.

| # | Notebook | Runs on | Publishes (private) | Size |
| --- | --- | --- | --- | --- |
| 1 | `kaggle_prep_speech.py` | Kaggle CPU, Internet on | `earmark-speech-16k`: `clean100_16k`, `clean360cap_16k` and `devtest_16k/{dev_clean,dev_other,test_clean,test_other}` | about 11.8 GB |
| 2 | `kaggle_prep_vctk_musan_noise.py` | Kaggle CPU, Internet on | `earmark-vctk-musan-noise-16k`: `vctk_16k`, `musan_music_16k/{train,heldout}`, `noise_rir_16k` (with `content_filter_log.json`) | about 4.7 GB |
| 3 | `colab_kokoro_agent_voice.py` | Colab T4 | `earmark-kokoro-agent-16k`: Kokoro agent-voice interferers (train and test voices) | about 0.35 GB |
| 4 | `colab_libricss_prep.py` | Colab CPU | `MyDrive/earmark/libricss_16k`: LibriCSS channel 0, timings and enrolment lists (download to the Mac) | about 1 GB |
| 5 | `kaggle_embeddings.py` | Kaggle CPU, after 1 and 2 | `earmark-embeddings`: eight WeSpeaker embeddings per training speaker | about 10 MB |
| 6 | `kaggle_train.py` | Kaggle GPU **T4 x2** | checkpoints in the private HF model repo `<hf-user>/earmark-checkpoints` | |
| - | `kaggle_serve_bench.py` | Kaggle GPU (stretch, not written yet) | batched streams-per-T4 benchmark | |

`docs/WEEK1_STATUS.md` has the exact settings, secrets and inputs for each run.

## Getting the source into a notebook

Each notebook looks for the repository in this order:

1. `EARMARK_SRC`: a folder or a zip. On Kaggle, attach a dataset named `earmark-src` made from
   `zip -r /tmp/earmark-src.zip python notebooks contract pyproject.toml -x '*/__pycache__/*'`.
   Add a `GIT_SHA` file to the zip so that checkpoints record the commit.
2. `EARMARK_GITHUB_REPO` (for example `github.com/<user>/EarMark`) and `EARMARK_GIT_REF`
   (default `main`). A private repository also needs a read-only `GH_TOKEN` secret.

## Rules

- Notebooks are thin: they install the repository, then call `earmark.*` functions. Logic
  that matters lives in `python/earmark/`, where it is unit-tested. `tests/data` and
  `tests/train` check every notebook statically (header cell, no tokens, every `earmark` call
  and its arguments, Python 3.10 syntax for the Kaggle and Colab images).
- Tokens come only from Kaggle Secrets or Colab Secrets (`HF_TOKEN`, `KAGGLE_USERNAME`,
  `KAGGLE_KEY`, `GH_TOKEN`), read at runtime into environment variables. Never paste a token
  into a cell.
- Each published dataset stays well under the 20 GB notebook-output limit.
- Training runs resume from the Hugging Face Hub checkpoint repo, never from
  `/kaggle/working`, and never across GPU types.
- Signal constants (sample rate, hop, bins, embedding size) are imported from
  `earmark.constants`, never retyped.
