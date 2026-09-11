#include <cmath>
#include <random>
#include <vector>

#include "arena.h"
#include "catch_amalgamated.hpp"
#include "fft.h"
#include "stft.h"
#include "test_support.h"

using earmark::Arena;
using earmark::kSpecFloats;
using earmark::Stft;

namespace {
constexpr int kHop = EARMARK_HOP_LENGTH;
constexpr int kWindow = EARMARK_WINDOW_LENGTH;
constexpr int kBins = EARMARK_N_BINS;
constexpr double kPi = 3.14159265358979323846;

struct StftFixture {
  Arena arena;
  Stft stft;
  StftFixture() {
    REQUIRE(arena.reserve(Stft::arena_bytes()));
    REQUIRE(stft.init(arena));
  }
};
}  // namespace

TEST_CASE("sqrt-Hann window matches the golden", "[stft]") {
  const earmark_test::Golden golden("stft");
  float window[kWindow];
  earmark::sqrt_hann_window(window);
  const auto stats = earmark_test::require_close("stft.window", window, golden.f32("window", {kWindow}), kWindow, 1e-7);
  earmark_test::report("stft.window", stats, 1e-7);
  // Periodic sqrt-Hann is power-complementary at 50 % overlap: w[n]^2 + w[n + hop]^2 = 1.
  for (int n = 0; n < kHop; ++n) {
    const double sum = static_cast<double>(window[n]) * window[n] + static_cast<double>(window[n + kHop]) * window[n + kHop];
    REQUIRE(std::fabs(sum - 1.0) < 1e-6);
  }
}

TEST_CASE("WOLA analysis matches the golden", "[stft]") {
  const earmark_test::Golden golden("stft");
  const std::size_t samples = golden.numel("input");
  const std::size_t frames = samples / kHop;
  const float* input = golden.f32("input");
  const float* spec = golden.f32("spec", {static_cast<uint32_t>(frames), kBins, 2});
  StftFixture f;
  float in_buf[kHop] = {};
  float ours[kSpecFloats];
  earmark_test::ErrorStats total;
  for (std::size_t t = 0; t < frames; ++t) {
    f.stft.analyze(input + t * kHop, in_buf, ours);
    earmark_test::accumulate(total, earmark_test::require_close("stft.spec frame " + std::to_string(t), ours,
                                                                spec + t * kSpecFloats, kSpecFloats));
  }
  earmark_test::report("stft.spec (per frame)", total);
}

TEST_CASE("WOLA synthesis matches the golden", "[stft]") {
  const earmark_test::Golden golden("stft");
  const std::size_t frames = golden.numel("synth_spec") / kSpecFloats;
  const float* spec = golden.f32("synth_spec");
  StftFixture f;
  float ola[kHop] = {};
  std::vector<float> out(frames * kHop);
  for (std::size_t t = 0; t < frames; ++t) f.stft.synthesize(spec + t * kSpecFloats, ola, out.data() + t * kHop);
  earmark_test::report("stft.synth_output", earmark_test::require_close("stft.synth_output", out.data(),
                                                                       golden.f32("synth_output"), out.size()));
  earmark_test::report("stft.synth_tail", earmark_test::require_close("stft.synth_tail", ola, golden.f32("synth_tail"), kHop));
}

TEST_CASE("analysis then synthesis returns the input one hop late", "[stft]") {
  StftFixture f;
  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.0f, 0.3f);
  const int hops = 40;
  std::vector<float> x(static_cast<std::size_t>(hops) * kHop);
  for (float& v : x) v = noise(rng);
  float in_buf[kHop] = {};
  float ola[kHop] = {};
  float spec[kSpecFloats];
  std::vector<float> y(x.size());
  for (int t = 0; t < hops; ++t) {
    f.stft.analyze(x.data() + t * kHop, in_buf, spec);
    f.stft.synthesize(spec, ola, y.data() + t * kHop);
  }
  const auto stats = earmark_test::compare(y.data() + kHop, x.data(), x.size() - kHop);
  earmark_test::report("stft identity reconstruction", stats, 1e-6);
  REQUIRE(stats.max_abs < 1e-6);
}

TEST_CASE("RealFft matches a direct DFT and inverts exactly", "[stft]") {
  Arena arena;
  REQUIRE(arena.reserve(earmark::RealFft::arena_bytes(EARMARK_N_FFT)));
  earmark::RealFft fft;
  REQUIRE(fft.init(EARMARK_N_FFT, arena));
  const int n = EARMARK_N_FFT;
  std::vector<float> x(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i)] = static_cast<float>(std::sin(0.37 * i) + 0.5 * std::cos(1.9 * i + 0.2));
  std::vector<float> spec(static_cast<std::size_t>(n + 2));
  fft.forward(x.data(), spec.data());
  std::vector<float> direct(static_cast<std::size_t>(n + 2));
  for (int k = 0; k <= n / 2; ++k) {
    double re = 0.0;
    double im = 0.0;
    for (int i = 0; i < n; ++i) {
      const double angle = -2.0 * kPi * static_cast<double>(k) * i / n;
      re += x[static_cast<std::size_t>(i)] * std::cos(angle);
      im += x[static_cast<std::size_t>(i)] * std::sin(angle);
    }
    direct[static_cast<std::size_t>(2 * k)] = static_cast<float>(re);
    direct[static_cast<std::size_t>(2 * k + 1)] = static_cast<float>(im);
  }
  earmark_test::report("fft vs direct DFT", earmark_test::require_close("fft vs direct DFT", spec.data(), direct.data(), spec.size()));
  std::vector<float> back(static_cast<std::size_t>(n));
  fft.inverse(spec.data(), back.data());
  REQUIRE(earmark_test::compare(back.data(), x.data(), x.size()).max_abs < 1e-6);
  earmark::RealFft odd;
  REQUIRE_FALSE(odd.init(321, arena));
}
