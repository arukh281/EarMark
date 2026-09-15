# The signal contract (M0)

Earmark runs the same network in three places: PyTorch (training, the dev runner),
the C++17 engine (native and WASM) and the browser (the AudioWorklet and UI). Every
signal-level number those three share lives in one file, `contract/signal.yaml`.
`contract/codegen.py` validates it, derives the dependent values and generates:

| Generated file | Used by |
| --- | --- |
| `python/earmark/constants.py` | `earmark.data`, `earmark.model`, `earmark.train`, `earmark.export`, `earmark.eval` |
| `engine/include/earmark_constants.h` | the engine (`earmark::contract::*` in C++, `EARMARK_*` macros for the C ABI) |
| `web/src/constants.js` | the worklet, the Workers and the UI (ES module) |

A constant has the same UPPER_SNAKE name in every language, so `HOP_LENGTH` in Python is
`earmark::contract::HOP_LENGTH` (and `EARMARK_HOP_LENGTH`) in C++ and `HOP_LENGTH` in JS.
Nobody retypes these numbers anywhere else. That removes a whole class of
PyTorch/C++/JS parity bugs (an off-by-one bin count, a different window, a different
hangover) at almost no cost.

## Values

| Constant | Value | Meaning |
| --- | --- | --- |
| `SAMPLE_RATE` | 16000 | Internal rate. The engine resamples device audio (48 kHz 3:1, 44.1 kHz 160:441). |
| `NUM_CHANNELS` | 1 | Mono at the model boundary. |
| `WINDOW` | `sqrt_hann_periodic` | `w[n] = sin(pi * n / 320)`, `n = 0..319`, for analysis and synthesis. |
| `WINDOW_LENGTH`, `HOP_LENGTH`, `N_FFT` | 320, 160, 320 | 20 ms window, 10 ms hop, no zero padding. |
| `N_BINS` | 161 | One-sided bins, `N_FFT // 2 + 1` (derived). Bin width 50 Hz. |
| `FFT_NORM` | `backward` | Forward DFT unscaled, inverse scaled by `1 / N_FFT` (numpy, torch and pocketfft with `fct = 1/N` on the inverse). |
| `STFT_CENTER` | false | Frame `t` covers samples `[160 t, 160 t + 320)`; no reflect padding. |
| `HOP_MS`, `WINDOW_MS`, `FRAME_RATE_HZ` | 10, 20, 100 | Derived. |
| `ALGORITHMIC_LATENCY_MS` / `_SAMPLES` | 20 / 320 | Window length plus lookahead. |
| `LOOKAHEAD_FRAMES` | 0 | The model never sees the future. |
| `ERB_BANDS`, `ERB_MIN_BINS` | 32, 2 | ERB gains and log-power features. |
| `ERB_WIDTHS` | 20 x 2, then 3, 6, 7, 7, 8, 8, 10, 12, 12, 14, 16, 18 | Bins per band, low to high, contiguous, sum 161 (derived). |
| `DF_ORDER`, `DF_BINS`, `DF_LOOKAHEAD_FRAMES` | 3, 64, 0 | Deep filter on bins 0-63 (0 to 3150 Hz) with taps on frames `t`, `t-1`, `t-2`. |
| `NORM_KIND`, `NORM_TAU_S` | `causal_exponential_mean`, 1.0 | Causal feature normalisation. |
| `NORM_ALPHA` | `exp(-0.01)` = 0.99005 | Per-hop decay (derived). |
| `EMBEDDING_DIM` | 256 | WeSpeaker ResNet34-LM embedding (and the learned NULL vector). |
| `VAD_REFERENCE`, `VAD_THRESHOLD_DB` | `utterance_peak`, -40.0 | VAD label rule. |
| `VAD_HANGOVER_MS` / `_FRAMES` | 50 / 5 | Label hangover. |
| `BARGEIN_MIN_ACTIVE_MS` / `_FRAMES` | 200 / 20 | Barge-in event: minimum activity. |
| `BARGEIN_MIN_SILENCE_MS` / `_FRAMES` | 300 / 30 | Barge-in event: minimum preceding silence. |
| `CONTRACT_VERSION`, `CONTRACT_HASH` | 1, 16 hex digits | Schema version and a hash of every other constant. |

## Conventions behind the numbers

**WOLA.** The periodic sqrt-Hann window satisfies `w[n]^2 + w[n + 160]^2 = 1`, so analysis
window, forward FFT, (mask), inverse FFT, synthesis window and overlap-add reconstructs
the input exactly with no extra gain, with `FFT_NORM = backward`. An output sample is
final only after the last frame covering it has been processed, and that frame needs
input up to one full window past its start; that is the 20 ms (window-length)
algorithmic latency. `tests/test_contract.py` checks this reconstruction.

**ERB bands.** `ERB_WIDTHS` follows the DeepFilterNet `erb_fb` rule, computed in double
precision: band edges equally spaced on the ERB-rate scale, rounded to the nearest bin,
each band at least `ERB_MIN_BINS` wide, the last band absorbing the remainder. Band `b`
covers bins `[sum(ERB_WIDTHS[:b]), sum(ERB_WIDTHS[:b + 1]))`. Consumers read the widths;
they never recompute them.

**Deep filter.** For bin `k < DF_BINS`, the output is
`Y[t, k] = sum_{i=0}^{DF_ORDER-1} C_i[t, k] * X[t - i, k]` (complex), with zero lookahead.

**Normalisation.** Each normalised quantity keeps a running mean updated once per hop:
`m_t = NORM_ALPHA * m_{t-1} + (1 - NORM_ALPHA) * x_t`, a 1 s time constant. The initial
values of these running means are part of the model's streaming state and are recorded in
the export manifest, not in the contract.

**VAD labels.** Per contract frame (the STFT framing above), compute the direct-path
target energy. A frame is active when that energy exceeds the utterance's peak frame
energy plus `VAD_THRESHOLD_DB` (-40 dB). Activity is then held for `VAD_HANGOVER_FRAMES`
frames after the last active frame.

**Barge-in event.** One definition everywhere (eval, the web Gate markers): an onset is
at least `BARGEIN_MIN_ACTIVE_FRAMES` frames of activity following at least
`BARGEIN_MIN_SILENCE_FRAMES` frames of inactivity.

**Contract hash.** `CONTRACT_HASH` is the first 16 hex digits of the SHA-256 of every
other (name, value) pair. Weight manifests and golden files should record it, and the
engine should refuse a blob whose manifest hash differs from its compiled
`EARMARK_CONTRACT_HASH`.

## Changing the contract

1. Edit `contract/signal.yaml`. Only primary values go there; derived values are computed.
2. Run `make codegen`. Validation fails loudly on unknown keys, missing keys, wrong
   types, or inconsistencies (for example a hop that is not half the window, a latency
   that does not match window plus lookahead, or a hangover that is not a whole number
   of hops).
3. Commit the YAML and the three generated files together.
4. Re-export weights and goldens: a new `CONTRACT_HASH` invalidates old blobs.

`make contract-check` (also run in CI) and `tests/test_contract.py` fail when any
generated file is stale. The test also reads the constants back from all three
languages (by parsing each file, by compiling the header as C++17 and as C, and by
importing the JS module in Node when these tools are available) and requires every value
to match exactly.
