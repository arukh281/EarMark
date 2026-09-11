# Earmark

Personal voice isolation and barge-in gating for voice agents, running live in the browser.

> **Status: work in progress.** The week-1 foundations are in: the signal contract, the data
> pipeline, the model, the evaluation harness and the native C++ engine, each with its own
> CPU tests. Next come the trainer, the WebAssembly build, CI and the browser demo. Nothing
> has been trained or measured yet, so there are no results here. Numbers will be added only
> when they come from logged runs in `results/runs.jsonl`. See
> [docs/WEEK1_STATUS.md](docs/WEEK1_STATUS.md) for exactly what is built.

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
- **Model size.** The main model has about 2M parameters (about 200 MMAC/s). A small 0.3M
  variant and a state-space (S4D) variant are planned for comparison.
- **Engine.** A dependency-free C++17 streaming engine behind a small C API (pocketfft is the
  only header it uses). It builds natively today and runs the full signal path (resampling,
  STFT, ERB features and synthesis), with tests that include a check that it allocates no
  memory after start-up. The network itself is not wired in yet. The WebAssembly build with
  SIMD, and the AudioWorklet that runs it in the browser, come next.
- **One signal contract.** `contract/signal.yaml` generates the constants for Python, C++ and
  JavaScript, so the three implementations cannot drift apart.

## Training data

Training mixtures are generated on the fly from public speech (LibriTTS-R and VCTK) plus
noise, room impulse responses and music beds. A synthetic "agent voice" (Kokoro TTS) is used
as an interferer, because an agent hearing itself is one of the main causes of false
barge-ins. Training runs on Kaggle GPUs. Evaluation speakers never appear in training.

## How it will be evaluated

Earmark is judged on **real room recordings**, not only on synthetic mixtures:

- **Test data:** LibriCSS (real meeting-room recordings with overlapping talkers), plus a
  short laptop-microphone set that will be recorded with volunteers' consent and released.
- **Baselines:** DeepFilterNet3 followed by a speaker-verification gate, and Silero VAD
  followed by the same gate.
- **Metrics:** false barge-ins per minute, barge-in onset delay, PESQ-WB, ESTOI, SI-SDR
  improvement, Whisper word error rate (raw vs enhanced vs gated), and end-to-end added
  latency measured with an acoustic loopback.
- **Honest numbers:** every threshold is frozen on the dev split before any test run, and
  every scored run is logged with its commit SHA. Before any Earmark number is trusted, the
  harness must reproduce published VoiceBank+DEMAND figures (noisy input PESQ 1.97 and
  STOI 0.921, GTCRN 2.87).

## Roadmap

- **Week 1:** signal contract, data pipeline, model, evaluation harness and native engine
  (done); trainer, WebAssembly build and CI (next).
- **Week 2:** training runs and the in-browser demo.
- **Week 3:** all evaluation suites.
- **Week 4:** release: results table, model card and a Hugging Face Space demo.

## Development

Local tooling: Python 3.12 in a uv-managed `.venv` (with `cmake` and `ninja` inside it), a
C++17 compiler, and Node 22 for the web tests. WebAssembly is built only in GitHub Actions.

```sh
source scripts/env.sh   # keep HF, torch, uv, pip, npm and Playwright caches in ./.cache
scripts/disk_guard.sh   # fails when less than 5 GiB is free; run it before big downloads

make test               # CPU unit tests; the slow and gate markers are excluded
make gates              # acceptance gates (need fetched evaluation data)
make codegen            # regenerate constants from contract/signal.yaml
make contract-check     # fail if the generated constants are stale
make engine             # configure (Ninja), build and ctest the C++ engine
make eval-dev           # score the dev split with the PyTorch-stream runner (lands next)
make clean-caches       # delete ./.cache downloads and Python caches
```

Layout:

- `contract/`: `signal.yaml`, the single source of every signal constant, and `codegen.py`,
  which generates `python/earmark/constants.py`, `engine/include/earmark_constants.h` and
  `web/src/constants.js`. See [docs/CONTRACT.md](docs/CONTRACT.md).
- `python/earmark/`: `data`, `model`, `train`, `export` and `eval` packages (tests import `earmark`).
- `engine/`: the dependency-free C++17 streaming engine and its tests. See
  [engine/README.md](engine/README.md).
- `web/`: the browser app. For now it holds only the generated constants.
- `notebooks/`: jupytext `.py` notebooks for Kaggle and Colab (see `notebooks/README.md`).
- `tests/`: CPU-only, offline pytest suites, one folder per package.
- `results/`: suite JSON, latency JSON and `runs.jsonl`.
- `docs/`: the signal contract and the week-1 status.

Tests never download datasets and never need a GPU. Mark anything slower than about a
minute with `@pytest.mark.slow` and anything that needs fetched evaluation data with
`@pytest.mark.gate`. Tokens (for example `HF_TOKEN`) come only from environment variables
or Kaggle and Colab Secrets.

## Licence

MIT, copyright 2026 Aradhya Khandelwal. See [LICENSE](LICENSE).

Third-party code: pocketfft (BSD-3-Clause, see `engine/third_party/POCKETFFT_LICENSE.md`),
the GTCRN baseline model code (MIT, see `python/earmark/eval/baselines/GTCRN_LICENSE.txt`) and
Catch2 for tests (Boost Software License 1.0). Datasets and pretrained models are used under
their own licences and are not redistributed here.
