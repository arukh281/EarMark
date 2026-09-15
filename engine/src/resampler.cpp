#include "resampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "matvec_f32.h"

namespace earmark {

namespace {
constexpr double kPi = 3.14159265358979323846;

// One prototype value before normalisation; resampler_prototype calls this twice per
// tap (once for the sum, once for the value), so both passes see identical numbers.
double prototype_value(const ResamplerDesign& d, int32_t k, double center, double i0_beta) {
  const double x = d.cutoff * (static_cast<double>(k) - center);
  const double sinc = x == 0.0 ? 1.0 : std::sin(kPi * x) / (kPi * x);
  const double r = 2.0 * static_cast<double>(k) / static_cast<double>(d.prototype_taps - 1) - 1.0;
  const double window = bessel_i0(d.beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0_beta;
  return d.cutoff * sinc * window;
}
}  // namespace

double ResamplerDesign::delay_seconds() const {
  if (identity()) return 0.0;
  return static_cast<double>(prototype_taps - 1) / (2.0 * static_cast<double>(in_rate) * static_cast<double>(up));
}

bool design_resampler(int32_t in_rate, int32_t out_rate, ResamplerDesign* out) {
  if (out == nullptr || in_rate <= 0 || out_rate <= 0) return false;
  const int32_t g = std::gcd(in_rate, out_rate);
  ResamplerDesign d;
  d.in_rate = in_rate;
  d.out_rate = out_rate;
  d.up = out_rate / g;
  d.down = in_rate / g;
  if (d.identity()) {
    *out = d;
    return true;
  }
  if (d.up > kResamplerMaxUp) return false;
  const double high = static_cast<double>(in_rate) * static_cast<double>(d.up);
  d.cutoff = static_cast<double>(std::min(in_rate, out_rate)) / high;
  const double transition = (1.0 - 2.0 * kResamplerPassbandFraction) * d.cutoff;
  const double delta_omega = 2.0 * kPi * transition;
  const double order = std::ceil((kResamplerAttenuationDb - 7.95) / (2.285 * delta_omega));
  if (!(order < static_cast<double>(kResamplerMaxTaps))) return false;
  const int32_t order_i = static_cast<int32_t>(order);
  d.taps_per_phase = (order_i + 1 + d.up - 1) / d.up;
  const int64_t prototype = static_cast<int64_t>(d.taps_per_phase) * d.up;
  if (prototype > kResamplerMaxTaps) return false;
  d.prototype_taps = static_cast<int32_t>(prototype);
  d.beta = 0.1102 * (kResamplerAttenuationDb - 8.7);
  *out = d;
  return true;
}

double bessel_i0(double x) {
  double total = 1.0;
  double term = 1.0;
  const double quarter = x * x / 4.0;
  for (int k = 1; k < 500; ++k) {
    term *= quarter / (static_cast<double>(k) * static_cast<double>(k));
    total += term;
    if (term < total * 1e-17) break;
  }
  return total;
}

void resampler_prototype(const ResamplerDesign& d, float* taps) {
  if (d.identity()) {
    taps[0] = 1.0f;
    return;
  }
  const int32_t n = d.prototype_taps;
  const double center = static_cast<double>(n - 1) / 2.0;
  const double i0_beta = bessel_i0(d.beta);
  double total = 0.0;
  for (int32_t k = 0; k < n; ++k) total += prototype_value(d, k, center, i0_beta);
  const double scale = static_cast<double>(d.up) / total;
  for (int32_t k = 0; k < n; ++k) taps[k] = static_cast<float>(prototype_value(d, k, center, i0_beta) * scale);
}

std::size_t Resampler::arena_bytes(const ResamplerDesign& d) {
  if (d.identity()) return 0;
  const auto taps = static_cast<std::size_t>(d.prototype_taps);
  const auto history = 2 * static_cast<std::size_t>(d.taps_per_phase);
  return Arena::aligned(taps * sizeof(float)) + Arena::aligned(history * sizeof(float));
}

bool Resampler::init(const ResamplerDesign& d, Arena& arena) {
  design_ = d;
  if (d.identity()) return true;
  const int32_t length = d.taps_per_phase;
  phase_taps_ = arena.allocate_array<float>(static_cast<std::size_t>(d.prototype_taps));
  history_ = arena.allocate_array<float>(2 * static_cast<std::size_t>(length));
  if (phase_taps_ == nullptr || history_ == nullptr) return false;
  // Same numbers as resampler_prototype(), written straight into per-phase rows:
  // h[k] with k = p + j * up goes to row p, column L - 1 - j (reversed, so that each
  // output is a forward dot product with the history, oldest sample first).
  const int32_t up = d.up;
  const double center = static_cast<double>(d.prototype_taps - 1) / 2.0;
  const double i0_beta = bessel_i0(d.beta);
  double total = 0.0;
  for (int32_t k = 0; k < d.prototype_taps; ++k) total += prototype_value(d, k, center, i0_beta);
  const double scale = static_cast<double>(up) / total;
  for (int32_t p = 0; p < up; ++p) {
    float* row = phase_taps_ + static_cast<std::ptrdiff_t>(p) * length;
    for (int32_t j = 0; j < length; ++j) {
      const int32_t k = p + (length - 1 - j) * up;
      row[j] = static_cast<float>(prototype_value(d, k, center, i0_beta) * scale);
    }
  }
  reset();
  return true;
}

void Resampler::reset() {
  if (design_.identity()) return;
  std::memset(history_, 0, 2 * static_cast<std::size_t>(design_.taps_per_phase) * sizeof(float));
  newest_ = design_.taps_per_phase - 1;  // the first sample lands in slot 0
  phase_ = 0;
  wait_ = 1;
}

std::size_t Resampler::max_output(const ResamplerDesign& design, std::size_t n) {
  if (design.identity()) return n;
  const auto up = static_cast<std::size_t>(design.up);
  const auto down = static_cast<std::size_t>(design.down);
  return (n * up + down - 1) / down + 1;
}

std::size_t Resampler::process(const float* in, std::size_t n, float* out) {
  if (design_.identity()) {
    std::memcpy(out, in, n * sizeof(float));
    return n;
  }
  const int32_t length = design_.taps_per_phase;
  const int32_t up = design_.up;
  const int32_t down = design_.down;
  std::size_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    newest_ = newest_ + 1 == length ? 0 : newest_ + 1;
    history_[newest_] = in[i];
    history_[newest_ + length] = in[i];
    --wait_;
    // The last L inputs, oldest first, are history_[newest_ + 1 .. newest_ + L].
    while (wait_ == 0) {
      out[count++] = dot_f32(phase_taps_ + static_cast<std::ptrdiff_t>(phase_) * length, history_ + newest_ + 1, length);
      phase_ += down;
      wait_ += phase_ / up;
      phase_ %= up;
    }
  }
  return count;
}

float Resampler::prototype_tap(int32_t k) const {
  if (design_.identity()) return 1.0f;
  const int32_t length = design_.taps_per_phase;
  const int32_t p = k % design_.up;
  const int32_t j = k / design_.up;
  return phase_taps_[static_cast<std::ptrdiff_t>(p) * length + (length - 1 - j)];
}

}  // namespace earmark
