// Real FFT of one fixed length, backed by the vendored pocketfft, allocation-free
// after init().
//
// Conventions match numpy / torch.fft with norm="backward": forward() is unscaled with
// X[k] = sum_n x[n] exp(-2 pi i k n / N), and inverse() scales by 1/N. Spectra are
// interleaved (re, im) pairs for bins 0..N/2.
#pragma once

#include <cstddef>

#include "arena.h"

namespace earmark {

class RealFft {
 public:
  RealFft() = default;
  ~RealFft();
  RealFft(const RealFft&) = delete;
  RealFft& operator=(const RealFft&) = delete;

  /// Arena bytes for a transform of length `n` (two work buffers).
  static std::size_t arena_bytes(int n) { return 2 * Arena::aligned(static_cast<std::size_t>(n) * sizeof(float)); }

  /// Builds the plan (one heap allocation, at create time) and takes the work buffers
  /// from `arena`. Fails for odd or non-positive `n`, and for lengths whose plan would
  /// allocate on every call (pocketfft's Bluestein plans).
  bool init(int n, Arena& arena);

  /// `in` [n] -> `out_ri` [(n/2 + 1) * 2]. The imaginary parts of DC and Nyquist are 0.
  void forward(const float* in, float* out_ri);

  /// `in_ri` [(n/2 + 1) * 2] -> `out` [n], scaled by 1/n. The imaginary parts of DC and
  /// Nyquist are ignored, as in numpy's irfft.
  void inverse(const float* in_ri, float* out);

  int size() const { return n_; }

 private:
  void* plan_ = nullptr;      // pocketfft::detail::pocketfft_r<float>, owned
  float* work_ = nullptr;     // [n] packed half-complex data, transformed in place
  float* scratch_ = nullptr;  // [n] pocketfft's ping-pong buffer
  int n_ = 0;
};

}  // namespace earmark
