// WOLA analysis and synthesis at the contract's framing (earmark.model.dsp).
//
// * Window: periodic sqrt-Hann, w[n] = sin(pi n / WINDOW_LENGTH).
// * Analysis (model framing): frame t = [previous hop | hop t] * w, then an unscaled
//   rfft of length N_FFT. The previous hop lives in StreamState::in_buf and starts at
//   zeros.
// * Synthesis: y = irfft(Y) * w (scaled 1/N_FFT); out = y[:HOP] + ola; ola = y[HOP:].
//
// With an identity mask, synthesis(analysis(x)) returns x delayed by one hop, exactly
// (w^2 at 50 % overlap sums to 1).
#pragma once

#include <cstddef>

#include "arena.h"
#include "earmark_constants.h"
#include "fft.h"

namespace earmark {

/// Interleaved spectrum length: N_BINS complex bins as (re, im).
inline constexpr int kSpecFloats = 2 * EARMARK_N_BINS;

class Stft {
 public:
  static std::size_t arena_bytes();

  /// Builds the FFT plan and the window. False on failure.
  bool init(Arena& arena);

  /// Analyses one hop: writes N_BINS interleaved bins to `spec_ri` and replaces
  /// `in_buf` [HOP] with `hop`. `hop` may alias `in_buf`'s storage only if identical.
  void analyze(const float* hop, float* in_buf, float* spec_ri);

  /// Synthesises one hop from `spec_ri` into `out` [HOP], updating `ola_buf` [HOP].
  void synthesize(const float* spec_ri, float* ola_buf, float* out);

  /// The float32 analysis/synthesis window [WINDOW_LENGTH].
  const float* window() const { return window_; }

 private:
  RealFft fft_;
  float* window_ = nullptr;  // [WINDOW_LENGTH]
  float* frame_ = nullptr;   // [N_FFT]
};

/// Fills `out` [WINDOW_LENGTH] with the contract window (computed in double).
void sqrt_hann_window(float* out);

}  // namespace earmark
