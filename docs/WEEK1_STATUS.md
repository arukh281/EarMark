# Week 1 status (11 Sep 2026)

All week-1 code is built, and every local check passes. What is left needs your accounts:
the Kaggle and Colab data runs, the training smoke run, and the first CI run after you push.
Nothing has been trained yet.

## Done-when checklist

| Week-1 criterion | Status | Evidence |
| --- | --- | --- |
| CI is green on the contract, data, model, trainer and random-weight engine goldens | **Written; runs on your first push.** | Locally, every command CI runs passes (see the next table). ASan and the WASM build run only in Linux CI, so they have never run. |
| Both VoiceBank+DEMAND gates pass | **Done** (local CPU, full 824-utterance test set) | Noisy input: PESQ-WB **1.9673** (target 1.97 ± 0.02), STOI **0.9211** (0.921 ± 0.02). GTCRN-VB checkpoint: PESQ-WB **2.8678** (2.87 ± 0.05). `make gates`: 2 passed in 24.5 s, after verifying every pinned download. |
| The private Kaggle datasets exist, each under 20 GB | **Your step** | The five data notebooks are written and statically tested, but none has been run. Expected sizes: 11.8 GB, 4.7 GB, 0.35 GB and 10 MB, plus LibriCSS at 1 GB on Drive. |
| On Earmark-Synth dev, mini-full improves SI-SDRi over unprocessed and its VAD AUC is at least 0.9 | **Blocked** on the datasets and the mini-full run | The dev runner is built and tested. Run the check with `make eval-dev EVAL_ARGS="... --check"` (step 9 below). |
| Resume from the Hub works on Kaggle | **Your step** (the smoke run) | On CPU, resume is bit-identical: the losses, weights, AdamW state, mixer index and every RNG match exactly. The Hub path is tested only against a fake Hub. The smoke notebook runs the Kaggle check (within 1e-3 over 100 steps) automatically. |
| `disk_guard` reports at least 5 GB free | **Done** | 12.4 GiB free. |

### Local checks (the commands CI runs)

| Command | Result |
| --- | --- |
| `make test` | 547 passed and 2 skipped (both need ffmpeg with libopus; CI installs it) in 17 s |
| `make contract-check` | in sync |
| `make goldens-check` | goldens are in sync (0.6 s) |
| `make engine` (Apple clang 17, RelWithDebInfo, `-Werror`) | 10/10 ctest suites (31 Catch2 cases) |
| `make engine-sanitize SANITIZERS=-DEARMARK_UBSAN=ON` | 10/10 under UBSan (ASan hangs in the macOS 26 runtime, so it runs on Linux only) |
| Engine with GCC 15 (`-Werror`) | 10/10, no warnings. This is a rehearsal for CI's GCC job. |
| `tests/test_tooling.py` | The workflow parses; every `make` target it calls exists; the smoke script parses under Node. |

Default tests per folder:

| Folder | Tests |
| --- | --- |
| `tests/data` | 206 |
| `tests/eval` | 106 |
| `tests/train` | 87 |
| `tests/model` | 70 |
| `tests/export` | 43 |
| `tests/` root (contract and tooling) | 37 |

Also excluded from `make test`: 2 gate tests and 1 slow MPS test.

## The nine week-1 work items

| # | Item | Status |
| --- | --- | --- |
| 1 | M0 signal contract | Done. One YAML file generates the Python, C++ and JS constants. `CONTRACT_HASH` is `875b5abc86eafabb`. |
| 2 | M1 data: prep notebooks, GPU mixer with VAD labels, property tests, Earmark-Synth generator | Done in code. The notebooks have not run. The Opus paths have not run locally. |
| 3 | M2: S-GRU, S-SSM and M, with streaming = offline, causality, perfect reconstruction and MAC tests | Done. Streaming matches offline to about 2e-7. |
| 4 | M3 trainer: capability check, 11 h timer, Hub checkpoints, CPU bit-identical resume | Done on CPU. The CUDA, AMP and real Hub paths first run on Kaggle. |
| 5 | M4 random-weight export and goldens | Done. The random M blob is 7.94 MB with 39 tensors; all 7 goldens are in sync. |
| 6 | Engine skeleton passing the per-layer goldens (ringbuf, resampler, STFT, ERB, GRU) | Done. The network is not wired into `em_process` yet: gains are fixed at 1 and the VAD output is 0. |
| 7 | M8 harness core and the two VB gates | Done (the gates are above). |
| 8 | PyTorch-stream dev runner | Done. Tested on synthetic data with random weights. |
| 9 | CI, with WASM built in Actions | Written: `.github/workflows/ci.yml` plus the `engine/wasm/smoke.mjs` smoke test. The first run happens on push. |

## Measured numbers

**Model size and compute.** From `PYTHONPATH=python .venv/bin/python -m earmark.model.macs`;
network MACs are counted on the `step()` path.

| Config | Params | Plan | MMAC/s (+DSP) | Plan | State per stream |
| --- | --- | --- | --- | --- | --- |
| S-GRU | 287,953 | 0.3M | 30.5 (31.9) | 30 | 4,352 B |
| S-SSM | 259,985 | matched MACs | 31.1 (32.4) | 30 | 68,864 B |
| M | 1,984,209 | 2M | 196.6 (198.0) | 200 | 6,400 B |
| M-256 (fallback) | 933,329 | about 1M | 93.3 (94.6) | | 5,376 B |

**Engine** (native, week-1 skeleton):

- Golden errors are at most 1.9e-6 on the STFT and 1.5e-5 on ERB power with a peak of 266.
  The GRU is within 8.9e-8 and matvec within 3.0e-7; ringbuf and the resampler taps are
  bit-exact.
- The resampler round trip is 129.9 dB (16k to 48k and back) and 126.4 dB (44.1k); the target
  is above 90 dB.
- Latency is 319 samples at 16 kHz and 1,145 at 48 kHz (23.9 ms). At 44.1 kHz it is 1,053.3
  samples, which is fractional, so align with `em_latency_seconds()`.
- There are 0 allocations after `em_create` (block sizes 1 to 4096, 6 device rates).
- A random-weight M blob loads with a 7.97 MB arena.

**Evaluation harness.** The Suite B scores above match the other published values too:
- noisy input: SI-SDR 8.45, CSIG 3.33, CBAK 2.44, COVL 2.62;
- GTCRN-VB: STOI 0.940, SI-SDR 18.80.

A full 824-utterance run takes 7 s (noisy) or 21 s (GTCRN) on 8 CPU workers.

## Your next steps

Do these in order. Numbered notebooks match `notebooks/README.md`, and each notebook's first
cell repeats its settings.

**0. Push and check CI.**
1. Commit the week-1 work, push the branch and open a pull request.
2. In the Actions tab, confirm the `contract`, `python`, `engine` (both jobs) and `wasm` jobs
   are green, and download the `earmark-wasm` artefact. The WASM job prints the module's
   imports; the web loader must satisfy each of them.
3. The `gates` job runs only on `main` and on manual dispatch.
4. If a job fails, the log names the exact step. The engine and WASM jobs have never run
   anywhere yet, so they are the most likely to need a fix.

**1. Accounts and secrets (once).**
- **Kaggle.** Verify your phone number (needed for Internet access in notebooks). Under
  Settings, API, create a token; this gives `KAGGLE_USERNAME` and `KAGGLE_KEY`.
- **Hugging Face.** Create two access tokens:
  - a **read** token, stored as the Kaggle secret `HF_TOKEN` for the data notebooks;
  - a **write** token, stored as the Kaggle secret `HF_TOKEN` attached to the training
    notebook only.

  The trainer creates the private repo `<hf-user>/earmark-checkpoints` itself and refuses a
  public one.
- **Colab.** Add the Colab secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`, so the Kokoro notebook
  can publish its dataset directly.

**2. Get the source into Kaggle and Colab.**
1. From the repo root, run:
   ```
   git rev-parse HEAD > /tmp/GIT_SHA
   zip -r /tmp/earmark-src.zip python notebooks contract pyproject.toml -x '*/__pycache__/*'
   zip -j /tmp/earmark-src.zip /tmp/GIT_SHA
   ```
2. Upload the zip as the private Kaggle dataset `earmark-src`, and copy it to Google Drive as
   `MyDrive/earmark/earmark-src.zip`.
3. Alternatively, once the code is on GitHub, set
   `EARMARK_GITHUB_REPO=github.com/arukh281/EarMark` in each notebook. A private repo also
   needs a read-only `GH_TOKEN` secret.

Convert each notebook with `uvx jupytext --to ipynb notebooks/<name>.py`, and do not commit
the `.ipynb` files.

**3. Kaggle, `kaggle_prep_speech`.**
- Settings: Accelerator None, Internet on. Inputs: `earmark-src`. Secret: `HF_TOKEN` (read).
- Run: Save Version, then Save & Run All. It takes about 2-3 h.
- Output: New Dataset, private, named **`earmark-speech-16k`** (about 11.8 GB). It contains
  `clean100_16k`, `clean360cap_16k` and `devtest_16k`.

**4. Kaggle, `kaggle_prep_vctk_musan_noise`.**
- Same settings; about 2-3 h.
- Output: private dataset **`earmark-vctk-musan-noise-16k`** (about 4.7 GB).
- Then read `noise_rir_16k/content_filter_log.json`. The AudioSet tagger may drop more DEMAND
  segments than planned.

**5. Colab, `colab_kokoro_agent_voice`.**
- Runtime: T4 GPU. Set `EARMARK_SRC` to the Drive zip, then Run all (about 10 min).
- Keep `SEED = 0`, because the mixer's held-out voices depend on it.
- It publishes **`earmark-kokoro-agent-16k`** (about 0.35 GB). Without the Kaggle secrets, it
  writes `MyDrive/earmark/kokoro_agent_16k.zip` for you to upload yourself.

**6. Colab, `colab_libricss_prep`.**
- Runtime: CPU. Takes about 30 min.
- If Drive blocks the 6.4 GB download, add a shortcut to the file in your Drive and set
  `EARMARK_LIBRICSS_ZIP` to its path.
- Then download `MyDrive/earmark/libricss_16k` (about 1 GB) to the Mac.

**7. Kaggle, `kaggle_embeddings`** (after steps 3 and 4).
- Inputs: `earmark-src`, `earmark-speech-16k` and `earmark-vctk-musan-noise-16k`. Takes about
  1 h.
- Output: private dataset **`earmark-embeddings`** (about 10 MB).
- Check its summary: within-speaker cosine must beat between-speaker cosine (the notebook
  stops otherwise). Also check whether it warns that the Opus round trip was skipped.

**8. Kaggle, `kaggle_train`: the smoke run, then mini-full.**
- Settings: Accelerator **GPU T4 x2** (never P100), Internet on, secret `HF_TOKEN` (write).
- Inputs: `earmark-src`, `earmark-speech-16k`, `earmark-vctk-musan-noise-16k`,
  `earmark-kokoro-agent-16k` and `earmark-embeddings`.
- Leave `RUNS = "smoke"`, then Save & Run All. It takes about 1 h 10 min.
- It passes when:
  - the log ends with `resume check PASSED`;
  - the private repo holds `runs/smoke/ckpt-*.pt` (3 files), `train_log.jsonl` and
    `run_config.json`.
- From the log, record:
  - `calibrated ... steps_per_s`, which decides M versus M-256;
  - `data_frac`, where a value above 0.3 means the CPU mixer is the bottleneck.
- Then run `RUNS = "mini-full"` (2 h).

**9. Earmark-Synth dev and the week-1 score (on the Mac).**
1. Download `devtest_16k`, `noise_rir_16k/{demand_eval,rir_real_eval}`,
   `musan_music_16k/heldout` and the Kokoro dataset.
2. Fetch ESC-50 (`corpora.ESC50_URL`) and run `export_esc50`.
3. Build the dev manifest with `load_bench_pools`, then `design_suite` (with the union of the
   training speakers), then `write_manifest`. Put a `pools.yaml` next to it.
4. Personal mode needs WeSpeaker embeddings. Either install `onnxruntime` and `torchaudio`
   locally, or run the dev runner once on Kaggle with `--embeddings-out emb.npz`.
5. Then run:
   ```
   make eval-dev EVAL_ARGS="--config M --checkpoint <mini_full.pt> --manifest <dev>/manifest.parquet --embeddings emb.npz --check"
   ```
   It exits 0 only if the SI-SDRi 95% interval is above 0 dB and the VAD AUC is at least 0.9.

**10. Recordings (Earmark-Real, weeks 2-3).**
- Book volunteers.
- Print `docs/CONSENT_TEMPLATE.md` and fill in its bracketed fields.
- Record with a laptop microphone and a close-talk microphone together, starting each take
  with a clap for alignment (`earmark_real.align_by_clap`).
- Keep the signed forms offline, and enter each volunteer's choices in the dataset manifest.

**Also:**
- Source `scripts/env.sh` once by hand; it has only been syntax-checked.
- Review the open items in `docs/DECISIONS.md`, especially the TSOS definition, before the
  PREREG commit.
- Review line by line the engine parts the plan says you own (ringbuf, resampler, `state.h`,
  the SIMD128 matvec and the GRU step).
- Confirm the LibriCSS licence before any LibriCSS audio appears in public
  (`docs/DATA_AND_LICENSES.md`).

## Known gaps carried into week 2

- **Engine:** the network (convs, FiLM, GRU body, heads, deep filter) is not wired into
  `em_process`, and there is no end-to-end engine-vs-PyTorch golden yet. The ctypes binding
  and the `earmark_bench` CLI are not built.
- **CUDA never run:** fp16 AMP, GradScaler, cuDNN GRU and the real Hub push first run in the
  Kaggle smoke run. The GPU resume tolerance of 1e-3 is an estimate.
- **Loss weights** were calibrated on synthetic batches; recheck them on mini-full.
- **Not built yet:**
  - the web app (`web/`) and its Playwright CI job;
  - the WebRTC APM scoring job;
  - the RNNoise, DFN3, cascade and raw-gate adapters, ASR/WER, and Suites A, R, D and E;
  - the enrolment ONNX graph.
