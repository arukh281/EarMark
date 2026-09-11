# Week 1 status (11 Sep 2026)

## Built and tested

- **Signal contract (M0).** `contract/signal.yaml` generates the Python, C++ and JavaScript
  constants, and a test proves all three agree. See [CONTRACT.md](CONTRACT.md).
- **Data and mixer (M1).** Parquet selection from Hugging Face, int16 shards, the noise
  content filter, the GPU mixer (other talkers, a synthetic agent voice, TV speech over music,
  loudspeaker chains) with VAD frame labels, speaker embeddings, the Earmark-Synth dev/test
  generator, loaders for LibriCSS and the self-recorded set, and five Kaggle/Colab prep
  notebooks.
- **Model (M2).** The causal network: grouped-convolution encoder, FiLM conditioning, a GRU
  or S4D body, ERB gains, the order-3 deep filter and the personal-VAD head. The streaming
  `step()` matches the offline `forward()`.
- **Export and engine (M4).** The flat weight blob with its manifest and golden tensors, and
  the C++17 engine: resampler, STFT, ERB features, GRU and matvec kernels, the weight loader
  and the C API. The network is not wired into `em_process` yet: gains are fixed at 1 and the
  VAD output is 0.
- **Evaluation core (M8).** Metrics (PESQ-WB, ESTOI, SI-SDRi, TSOS, interferer suppression),
  barge-in events, bootstrap confidence intervals, the append-only runs log, a streaming
  scorer, the GTCRN baseline, Suite B (VoiceBank+DEMAND) and the evaluation-data fetch script.

**Tests:** `make test` gives 422 passed and 2 skipped (both need ffmpeg with libopus).
`make engine` passes all 10 ctest suites, including zero allocations after `em_create`.
`make contract-check` is in sync.

## Not done yet

These parts of the week-1 build stopped on a usage limit and will be rerun:

- the M3 trainer (losses, mixed precision, Hub checkpoints and resume);
- the dev runner behind `make eval-dev`;
- integration and CI: GitHub Actions for the CPU tests, ASan/UBSan on Linux, the WebAssembly
  build and the Playwright browser tests;
- two review passes and a fix pass.

ASan cannot run on this Mac (the macOS 26 runtime hangs before `main`), so it has to run in
Linux CI.

## Decisions to review

- **Contract additions:** the ERB band widths (DeepFilterNet rule, at least 2 bins per band),
  the periodic sqrt-Hann window, backward FFT normalisation, no centring, and the barge-in
  event constants.
- **TSOS:** implemented in a level-independent form instead of the paper's equation 3.
  Review it before the pre-registration commit, because hypothesis H2 depends on it.
- **Enrolment front end:** the training embeddings use WeSpeaker's Kaldi fbank with a
  Hamming window. The browser enrolment graph must match it, or the embeddings must be
  recomputed.
- **Engine core:** ringbuf, the resampler, the state struct, the SIMD128 matvec and the GRU
  step were agent-written and should be reviewed line by line.

## Your steps on Kaggle and Colab

**Getting the code into a notebook.** Once this week's PR is merged, set
`EARMARK_GITHUB_REPO=github.com/arukh281/EarMark` in the notebook. The repo is public, so no
`GH_TOKEN` is needed (`EARMARK_GIT_REF` picks the branch). Before the merge, zip the source
from the repo root with
`zip -r /tmp/earmark-src.zip python notebooks contract pyproject.toml -x '*/__pycache__/*'`
and upload it as a private Kaggle dataset named `earmark-src`, or to Google Drive as
`MyDrive/earmark/earmark-src.zip`.

Convert each notebook with `uvx jupytext --to ipynb notebooks/<name>.py` before uploading it,
and don't commit the `.ipynb` files.

1. **Kaggle, `kaggle_prep_speech`.** Accelerator None, Internet on (the account needs phone
   verification), and optionally a read-only `HF_TOKEN` secret. Save & Run All; it takes
   2-3 hours. Publish the output as the private dataset `earmark-speech-16k` (about 11.8 GB).
2. **Kaggle, `kaggle_prep_vctk_musan_noise`.** Same settings, 2-3 hours. Publish the output as
   `earmark-vctk-musan-noise-16k` (about 4.7 GB), then read
   `noise_rir_16k/content_filter_log.json` to see which clips the filter dropped.
3. **Colab, `colab_kokoro_agent_voice`.** T4 GPU. Add `KAGGLE_USERNAME` and `KAGGLE_KEY`
   secrets to publish `earmark-kokoro-agent-16k` directly; without them it writes a zip to
   Drive for you to upload. About 10 minutes. Keep `SEED = 0`.
4. **Colab, `colab_libricss_prep`.** CPU runtime, about 30 minutes. If Drive blocks the
   6.4 GB download, add a shortcut to the file in your Drive and set `EARMARK_LIBRICSS_ZIP`.
   Then download `MyDrive/earmark/libricss_16k` (about 1 GB) to the Mac.
5. **Kaggle, `kaggle_embeddings`** (after 1 and 2). Inputs: the source, `earmark-speech-16k`
   and `earmark-vctk-musan-noise-16k`. About 1 hour. Publish the output as
   `earmark-embeddings` (about 10 MB), and check that within-speaker cosine beats
   between-speaker cosine (the notebook stops if it doesn't).
