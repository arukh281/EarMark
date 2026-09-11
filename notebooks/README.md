# Notebooks

Every notebook here is a [jupytext](https://jupytext.readthedocs.io/) percent-format `.py`
file, so it diffs and reviews like code. Open it in Jupyter with jupytext installed, or
upload it to Kaggle or Colab, which both import `.py` notebooks. Never commit `.ipynb`
files or cell outputs.

| Notebook | Runs on | Produces |
| --- | --- | --- |
| `kaggle_prep_speech.py` | Kaggle CPU | `clean100_16k`, `clean360cap_16k` private datasets (parquet column projection, soxr, int16 shards) |
| `kaggle_prep_vctk_musan_noise.py` | Kaggle CPU | `vctk_16k`, `musan_music_16k`, `noise_rir_16k` private datasets |
| `colab_kokoro_agent_voice.py` | Colab | `kokoro_agent_16k` agent-voice interferers |
| `colab_libricss_prep.py` | Colab | LibriCSS channel-0 sessions, timings and enrolment lists |
| `kaggle_embeddings.py` | Kaggle | eight WeSpeaker embeddings per training speaker |
| `kaggle_train.py` | Kaggle GPU T4 x2 | M, S-GRU and S-SSM training runs (config flags), v2 fine-tune |
| `kaggle_serve_bench.py` | Kaggle GPU (stretch) | batched streams-per-T4 benchmark |

## Rules

- Notebooks are thin: they install the repository, then call `earmark.*` functions. Logic
  that matters lives in `python/earmark/` where it is unit-tested.
- Tokens come only from Kaggle Secrets or Colab Secrets (for example `HF_TOKEN`), read at
  runtime into environment variables. Never paste a token into a cell.
- Each dataset a notebook publishes stays well under the 20 GB notebook-output limit.
- Training runs resume from the Hugging Face Hub checkpoint repo, never from
  `/kaggle/working`, and never across GPU types.
- Signal constants (sample rate, hop, bins, embedding size) are imported from
  `earmark.constants`, never retyped.
