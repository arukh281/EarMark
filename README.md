# Earmark

Personal voice isolation and barge-in gating for voice agents, running live in the browser.

> **Status: work in progress.** The foundations are being built now: the signal contract,
> data pipeline, model, trainer, C++ engine and evaluation harness. Nothing has been trained
> or measured yet, so there are no results here. Numbers will be added only when they come
> from logged runs in `results/runs.jsonl`.

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

Audio stays on your machine: the model runs in the page, and nothing is uploaded.

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
  256-dimensional speaker embedding. It runs once, in a Web Worker.
- **Model size.** The main model targets about 2M parameters (about 200 MMAC/s). A small
  0.3M variant and a state-space (S4D) variant are planned for comparison.
- **Engine.** A dependency-free C++17 streaming engine (pocketfft is the only header it uses),
  compiled to WebAssembly with SIMD and run inside an AudioWorklet. It allocates no memory
  after start-up, and the same code builds natively for tests and benchmarks.
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

- **Week 1 (now):** signal contract, data pipeline, model, trainer, engine, evaluation
  harness and CI.
- **Week 2:** training runs and the in-browser demo.
- **Week 3:** all evaluation suites.
- **Week 4:** release: results table, model card and a Hugging Face Space demo.

## Licence

MIT, copyright 2026 Aradhya Khandelwal. See [LICENSE](LICENSE).

Third-party code: pocketfft (BSD-3-Clause, see `engine/third_party/POCKETFFT_LICENSE.md`) and
Catch2 for tests (Boost Software License 1.0). Datasets and pretrained models are used under
their own licences and are not redistributed here.
