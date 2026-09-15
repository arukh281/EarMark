// Rational polyphase resampler (device rate <-> 16 kHz), streaming and allocation-free.
//
// Design (mirrors earmark.export.reference.design_resampler operation for operation):
// up/down = out/in reduced by their gcd; one Kaiser-windowed-sinc prototype at the
// upsampled rate in * up, with cutoff at the lower Nyquist, 120 dB stop band and
// passband up to 0.4375 x the lower rate (7 kHz when 16 kHz is the lower rate). The
// prototype is normalised so its taps sum to `up` (unit passband gain).
//
// Streaming: output m uses t = m * down, base = t / up, phase p = t % up, and
//   y[m] = sum_k h[p + k * up] * x[base - k],   k < taps_per_phase,
// with zeros before the first input. Output m is emitted as soon as x[base] arrives, so
// n inputs yield exactly ceil(n * up / down) outputs in total, however they are split
// into blocks. Linear phase: the group delay is (prototype_taps - 1) / (2 in up) seconds.
#pragma once

#include <cstddef>
#include <cstdint>

#include "arena.h"

namespace earmark {

inline constexpr double kResamplerAttenuationDb = 120.0;
inline constexpr double kResamplerPassbandFraction = 0.4375;
inline constexpr int32_t kResamplerMaxUp = 1024;
inline constexpr int32_t kResamplerMaxTaps = 1 << 16;

struct ResamplerDesign {
  int32_t in_rate = 0;
  int32_t out_rate = 0;
  int32_t up = 1;
  int32_t down = 1;
  int32_t taps_per_phase = 1;
  int32_t prototype_taps = 1;
  double cutoff = 1.0;  ///< lower rate / (in_rate * up)
  double beta = 0.0;    ///< Kaiser shape

  bool identity() const { return up == 1 && down == 1; }
  /// Group delay of the prototype in seconds (0 for the identity).
  double delay_seconds() const;
  /// Group delay in output samples (may be fractional).
  double delay_out_samples() const { return delay_seconds() * out_rate; }
};

/// Designs in_rate -> out_rate. False for non-positive rates, up > kResamplerMaxUp or
/// more than kResamplerMaxTaps prototype taps.
bool design_resampler(int32_t in_rate, int32_t out_rate, ResamplerDesign* out);

/// Modified Bessel function I0 by its power series (the same series as the reference).
double bessel_i0(double x);

/// Writes the float32 prototype h [prototype_taps], computed in double.
void resampler_prototype(const ResamplerDesign& design, float* taps);

class Resampler {
 public:
  /// Arena bytes for `design` (per-phase taps plus the doubled history).
  static std::size_t arena_bytes(const ResamplerDesign& design);

  /// Computes the taps and takes storage from `arena`. False on failure.
  bool init(const ResamplerDesign& design, Arena& arena);

  /// Clears the history (back to all zeros) and the output phase.
  void reset();

  /// Upper bound on the outputs one process() call makes from `n` inputs.
  static std::size_t max_output(const ResamplerDesign& design, std::size_t n);
  std::size_t max_output(std::size_t n) const { return max_output(design_, n); }

  /// Consumes `n` inputs and writes every output they complete to `out` (which must
  /// hold max_output(n) floats). Returns the number written. `out` must not alias `in`.
  std::size_t process(const float* in, std::size_t n, float* out);

  /// Prototype tap h[k] as stored (for tests).
  float prototype_tap(int32_t k) const;

  const ResamplerDesign& design() const { return design_; }

 private:
  ResamplerDesign design_{};
  float* phase_taps_ = nullptr;  // [up][L]: phase p, j -> h[p + (L - 1 - j) * up]
  float* history_ = nullptr;     // [2L]: history_[i] == history_[i + L]
  int32_t newest_ = 0;           // slot of the newest sample, 0..L-1
  int32_t phase_ = 0;            // p of the next output
  int64_t wait_ = 1;             // inputs still needed before the next output
};

}  // namespace earmark
