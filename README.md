# Earmark

Personal voice isolation and barge-in gating for voice agents, running live in the browser.

> **Status: work in progress (end of week 1).** The foundations are built and tested:
>
> - the signal contract;
> - the data pipeline and Kaggle/Colab notebooks;
> - three model sizes;
> - the trainer with Hugging Face checkpointing;
> - the weight export;
> - the native C++ engine skeleton;
> - the evaluation harness and its dev runner;
> - CI, which also builds the WebAssembly module.
>
> The harness reproduces the published VoiceBank+DEMAND figures. **Nothing has been trained
> yet, so there are no Earmark results here.** Numbers will be added only when they come from
> logged runs in `results/runs.jsonl`. See [docs/WEEK1_STATUS.md](docs/WEEK1_STATUS.md) for
> exactly what is built, what was measured, and what comes next.

## The problem

Voice agents get interrupted by the wrong voice. A TV in the background, a colleague talking
nearby, or the agent's own voice leaking back from the laptop speaker can all trigger a
"barge-in", and the agent stops mid-sentence. Ordinary noise suppression and voice-activity
detection can't fix this, because every one of those sounds is real speech.

## What Earmark does

You record about 5 seconds of your voice. After that, a small causal model runs in your
browser tab and, every 10 ms:

1. **Isolates your voice.** It removes noise and other talkers, including the agent's own
   voice coming back through the speakers.
2. **Gates barge-in.** It outputs a personal voice-activity signal that is on only while
   *you* are speaking, so the agent stops for you and nobody else.

One set of weights serves three modes:

| Mode     | What the speech recogniser hears     | What Earmark drives        |
| -------- | ------------------------------------ | -------------------------- |
| Personal | your isolated voice                  | barge-in and endpointing   |
| Gate     | the raw microphone audio             | barge-in and endpointing   |
| Denoise  | all speech, denoised (no enrolment)  | nothing                    |

By design, audio stays on your machine: the model runs in the page, and nothing is uploaded.

## How it works

```
microphone (48 or 44.1 kHz)
  -> resample to 16 kHz mono
  -> 20 ms sqrt-Hann window, 10 ms hop (161 frequency bins)
  -> features: 32 ERB bands + complex low band (0-3.15 kHz, 64 bins)
  -> grouped-convolution encoder
  -> FiLM conditioning on your speaker embedding
  -> 2-layer GRU
  -> heads: 32 ERB gains | order-3 deep filter on the low band | personal VAD
  -> inverse STFT (overlap-add)
  => your voice, plus a per-frame "you are speaking" signal
```

- **Causal and low-latency.** The model never looks at future audio: 20 ms of algorithmic
  latency with zero lookahead.
- **Enrolment.** A frozen WeSpeaker ResNet34-LM model turns about 5 s of your speech into a
  256-dimensional speaker embedding. In the browser it will run once, in a Web Worker.
- **Model sizes (measured).**

  | Model | Parameters | Compute |
  | --- | --- | --- |
  | M (main) | 1.98M | 197 MMAC/s |
  | S-GRU | 0.29M | 30.5 MMAC/s |
  | S-SSM (S4D state-space) | 0.26M | 31.1 MMAC/s |

  All three run the same streaming `step()` path.
- **Engine.** A dependency-free C++17 streaming engine behind a small C API (pocketfft is the
  only header it uses). It already runs the full signal path: resampling, STFT, ERB features
  and synthesis. Tests check it against Python goldens, and check that it allocates no memory
  after start-up. The network itself is not wired in yet. The WebAssembly build (SIMD128,
  fixed memory) runs in GitHub Actions; the AudioWorklet that runs it in the browser comes
  next.
- **One signal contract.** `contract/signal.yaml` generates the constants for Python, C++ and
  JavaScript, so the three implementations cannot drift apart.

## Training data

Training mixtures are generated on the fly from public speech (LibriTTS-R and VCTK) plus
noise, room impulse responses and music beds. A synthetic "agent voice" (Kokoro TTS) is used
as an interferer, because an agent hearing itself is one of the main causes of false
barge-ins. Training runs on Kaggle GPUs, with checkpoints in a private Hugging Face repo.
Evaluation speakers never appear in training, and the mixer refuses to start if they do.
Every source and its licence is listed in [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md).

## How it will be evaluated

Earmark is judged on **real room recordings**, not only on synthetic mixtures:

- **Test data:** LibriCSS (real meeting-room recordings with overlapping talkers), plus a
  short laptop-microphone set that will be recorded with volunteers' consent
  ([consent form](docs/CONSENT_TEMPLATE.md)) and released.
- **Baselines:** DeepFilterNet3 followed by a speaker-verification gate, and a raw personal
  gate.
- **Metrics:**
  - false barge-ins per minute and barge-in onset delay;
  - PESQ-WB, ESTOI and SI-SDR improvement;
  - Whisper word error rate (raw vs enhanced vs gated);
  - end-to-end added latency, measured with an acoustic loopback.
- **Honest numbers:** every threshold is frozen on the dev split before any test run, and
  every scored run is logged with its commit SHA and inference path. The harness was first
  checked against published figures. On the VoiceBank+DEMAND test set, noisy input scores
  PESQ-WB 1.967 and STOI 0.921 (published 1.97 and 0.921), and the official GTCRN checkpoint
  scores PESQ-WB 2.868 (published 2.87).

## Roadmap

- **Week 1 (done):** signal contract, data pipeline, model, trainer, export, engine skeleton,
  evaluation harness, dev runner and CI.
- **Week 2:** training runs on Kaggle, the network wired into the engine, and the in-browser
  demo. Code freezes at the end of the week.
- **Week 3:** all evaluation suites.
- **Week 4:** release: results table, model card and a Hugging Face Space demo.

## Development

Local tooling:
- Python 3.12 in a uv-managed `.venv`, with `cmake` and `ninja` inside it;
- a C++17 compiler;
- Node 22 for the JavaScript checks.

WebAssembly is built only in GitHub Actions.

```sh
source scripts/env.sh     # keep HF, torch, uv, pip, npm and Playwright caches in ./.cache
make disk-guard           # fails when less than 5 GiB is free; run it before big downloads

make test                 # CPU unit tests (about 550, under 20 s); slow and gate tests excluded
make gates                # fetch + verify VoiceBank+DEMAND and GTCRN, then the Suite B gates
make contract-check       # fail if the generated constants are stale (make codegen rewrites them)
make goldens-check        # fail if the engine goldens are stale (make goldens rewrites them)
make engine               # configure (Ninja), build and ctest the C++ engine
make engine-sanitize      # the same under ASan + UBSan (Linux; on macOS pass SANITIZERS=-DEARMARK_UBSAN=ON)
make eval-dev EVAL_ARGS="--config M --checkpoint CKPT --manifest DEV/manifest.parquet --embeddings EMB.npz --check"
                          # score Earmark-Synth dev with the PyTorch-stream runner
make clean-caches         # delete ./.cache downloads and Python caches
make help                 # every target
```

**CI** (`.github/workflows/ci.yml`, ubuntu-24.04, Python 3.12) runs:
- the contract check;
- the Python CPU tests, with CPU-only torch and ffmpeg for the Opus tests;
- the engine's Catch2 tests, under clang ASan + UBSan and as a GCC release build;
- the Emscripten build, uploaded as the `earmark-wasm` artefact and smoke-tested in Node;
- on `main` only, the Suite B gates.

Layout:

- `contract/`: `signal.yaml`, the single source of every signal constant, and `codegen.py`,
  which generates `python/earmark/constants.py`, `engine/include/earmark_constants.h` and
  `web/src/constants.js`. See [docs/CONTRACT.md](docs/CONTRACT.md).
- `python/earmark/`: the `data`, `model`, `train`, `export` and `eval` packages.
- `engine/`: the C++17 streaming engine, its Catch2 tests and the WASM build. See
  [engine/README.md](engine/README.md). Every engine file was agent-written in week 1; the
  core parts are marked for line-by-line review by the author.
- `notebooks/`: jupytext `.py` notebooks for Kaggle and Colab, in run order (see
  [notebooks/README.md](notebooks/README.md)).
- `web/`: the browser app. For now it holds only the generated constants.
- `tests/`: CPU-only, offline pytest suites, one folder per package.
- `results/`: suite JSON, latency JSON and `runs.jsonl`.
- `docs/`:
  - [WEEK1_STATUS.md](docs/WEEK1_STATUS.md);
  - [CONTRACT.md](docs/CONTRACT.md);
  - [DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md);
  - [DECISIONS.md](docs/DECISIONS.md);
  - [FAILURES.md](docs/FAILURES.md);
  - [CONSENT_TEMPLATE.md](docs/CONSENT_TEMPLATE.md).

Test rules:
- Tests never download datasets and never need a GPU.
- Mark anything slower than about a minute with `@pytest.mark.slow`.
- Mark anything that needs fetched evaluation data with `@pytest.mark.gate`.
- Tokens (for example `HF_TOKEN`) come only from environment variables or from Kaggle and
  Colab Secrets.

## Licence

MIT, copyright 2026 Aradhya Khandelwal. See [LICENSE](LICENSE).

Third-party code:
- pocketfft (BSD-3-Clause, see `engine/third_party/POCKETFFT_LICENSE.md`);
- the GTCRN baseline model code (MIT, see `python/earmark/eval/baselines/GTCRN_LICENSE.txt`);
- Catch2, for the tests (Boost Software License 1.0).

Datasets and pretrained models are used under their own licences and are not redistributed
here. See [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md).
