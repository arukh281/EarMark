# Earmark engine

A dependency-free C++17 streaming engine behind a C ABI (`include/earmark.h`), built
natively with CMake + Ninja and as a standalone WASM module (CI only). The only
vendored code is pocketfft (header only) and, for the tests, Catch2. See
`third_party/README.md` for their versions, hashes and licences.

## Build and test

```sh
.venv/bin/cmake -S engine -B engine/build -G Ninja -DCMAKE_MAKE_PROGRAM=$PWD/.venv/bin/ninja
.venv/bin/cmake --build engine/build
.venv/bin/ctest --test-dir engine/build --output-on-failure
```

`make engine` runs the same three steps. Sanitizers: add `-DEARMARK_ASAN=ON
-DEARMARK_UBSAN=ON` and use a separate build directory, for example
`engine/build-asan`. WASM: `engine/wasm/build.sh` (needs `emcc`; runs in GitHub
Actions).

## Layout

| Path | What it is |
| --- | --- |
| `include/earmark.h` | the C ABI: `em_create`, `em_set_embedding`, `em_process`, `em_process_hop_16k`, `em_reset`, and so on |
| `include/earmark_constants.h` | generated from `contract/signal.yaml`; never edit by hand |
| `src/arena.*` | fixed bump allocator; frozen after `em_create` |
| `src/ringbuf.*` | bounded FIFO with partial transfers |
| `src/resampler.*` | rational polyphase Kaiser resampler (device rate to and from 16 kHz) |
| `src/fft.*`, `src/stft.*` | pocketfft real FFT (allocation-free patch) and sqrt-Hann WOLA |
| `src/erb.*` | ERB band power and features, low-band unit norm, ERB gains |
| `src/gru.*` | stacked GRU step (PyTorch gate order r, z, n) |
| `src/matvec_f32.*` | fp32 dot/matvec; the WASM SIMD128 path and the scalar path share one summation order |
| `src/weights.*` | `.emwb` blob parser, CRC-32 and manifest cross-check |
| `src/state.*` | fixed-size per-stream state, the prefix of `earmark.model.stream.STATE_FIELDS` |
| `src/engine.cpp` | the C API over the signal path |
| `tests/` | Catch2 tests against the Python goldens (`tests/goldens/`, regenerate with `python -m earmark.export.golden`) plus the allocation counter |
| `wasm/build.sh` | Emscripten build: `-sSTANDALONE_WASM --no-entry`, fixed memory, SIMD128 |

## Status (week 1)

The signal path runs end to end at full cost: resampling, WOLA, ERB and low-band
normalisation, synthesis. The network (encoder, FiLM, GRU body, heads, deep filter) is
not wired into `em_process` yet: its gains are fixed at 1 and the VAD output is 0.
The GRU step and matvec kernels are implemented and golden-tested, ready to be wired
in.

## Ownership

Every file under `engine/` was agent-written for week 1. The components the plan marks
as author-owned (ringbuf, resampler, the state struct, the SIMD128 matvec and the
GRU step) are meant to be reviewed and owned line by line.
