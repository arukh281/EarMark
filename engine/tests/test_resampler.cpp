#include <cmath>
#include <string>
#include <vector>

#include "arena.h"
#include "catch_amalgamated.hpp"
#include "resampler.h"
#include "test_support.h"

using earmark::Arena;
using earmark::Resampler;
using earmark::ResamplerDesign;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Stage {
  Arena arena;
  Resampler resampler;
  explicit Stage(const ResamplerDesign& d) {
    REQUIRE(arena.reserve(Resampler::arena_bytes(d) + 64));
    REQUIRE(resampler.init(d, arena));
  }
};

ResamplerDesign design(int32_t in_rate, int32_t out_rate) {
  ResamplerDesign d;
  REQUIRE(earmark::design_resampler(in_rate, out_rate, &d));
  return d;
}

// Streams `x` through `stage` in blocks cycling through `blocks`.
std::vector<float> run(Resampler& stage, const std::vector<float>& x, const std::vector<std::size_t>& blocks) {
  std::vector<float> y(stage.max_output(x.size()) + x.size());
  std::size_t pos = 0;
  std::size_t written = 0;
  std::size_t index = 0;
  while (pos < x.size()) {
    const std::size_t n = std::min(blocks[index++ % blocks.size()], x.size() - pos);
    written += stage.process(x.data() + pos, n, y.data() + written);
    pos += n;
  }
  y.resize(written);
  return y;
}

}  // namespace

TEST_CASE("resampler designs, taps and streamed output match the golden", "[resampler]") {
  const earmark_test::Golden golden("resampler");
  const std::size_t cases = golden.numel("cases") / 2;
  REQUIRE(cases == 4);
  for (std::size_t c = 0; c < cases; ++c) {
    const std::string prefix = "case" + std::to_string(c) + ".";
    const int32_t* rates = golden.i32(prefix + "rates");
    const int32_t* want = golden.i32(prefix + "design");
    INFO(rates[0] << " -> " << rates[1]);
    const ResamplerDesign d = design(rates[0], rates[1]);
    REQUIRE(d.up == want[0]);
    REQUIRE(d.down == want[1]);
    REQUIRE(d.taps_per_phase == want[2]);
    REQUIRE(d.prototype_taps == want[3]);
    REQUIRE(std::fabs(d.delay_out_samples() - golden.f32(prefix + "delay_out_samples")[0]) < 1e-3);

    Stage stage(d);
    const int32_t stride = golden.i32(prefix + "taps_stride")[0];
    const float* taps = golden.f32(prefix + "taps");
    const std::size_t tap_count = golden.numel(prefix + "taps");
    std::vector<float> ours(tap_count);
    std::size_t exact = 0;
    for (std::size_t i = 0; i < tap_count; ++i) {
      ours[i] = stage.resampler.prototype_tap(static_cast<int32_t>(i) * stride);
      exact += ours[i] == taps[i] ? 1 : 0;
    }
    // Same double-precision recipe on both sides: only a last-ulp libm difference is allowed.
    const earmark_test::ErrorStats tap_stats = earmark_test::require_close(prefix + "taps", ours.data(), taps, tap_count, 1e-7);
    earmark_test::report(prefix + "taps (" + std::to_string(exact) + "/" + std::to_string(tap_count) + " bit-exact)",
                         tap_stats, 1e-7);

    const float* input = golden.f32(prefix + "input");
    const std::vector<float> x(input, input + golden.numel(prefix + "input"));
    const int32_t* block_sizes = golden.i32(prefix + "blocks");
    const std::vector<std::size_t> blocks(block_sizes, block_sizes + golden.numel(prefix + "blocks"));
    const std::vector<float> y = run(stage.resampler, x, blocks);
    REQUIRE(y.size() == golden.numel(prefix + "expected"));
    earmark_test::report(prefix + "expected",
                         earmark_test::require_close(prefix + "expected", y.data(), golden.f32(prefix + "expected"), y.size()));
  }
}

TEST_CASE("resampler output does not depend on the block split", "[resampler]") {
  const auto [in_rate, out_rate] = GENERATE(std::pair{48000, 16000}, std::pair{16000, 44100}, std::pair{44100, 16000},
                                            std::pair{16000, 48000}, std::pair{16000, 16000});
  INFO(in_rate << " -> " << out_rate);
  std::vector<float> x(3000);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(std::sin(0.013 * i) + 0.2 * std::cos(0.41 * i));
  Stage whole(design(in_rate, out_rate));
  Stage single(design(in_rate, out_rate));
  Stage irregular(design(in_rate, out_rate));
  const std::vector<float> a = run(whole.resampler, x, {x.size()});
  const std::vector<float> b = run(single.resampler, x, {1});
  const std::vector<float> c = run(irregular.resampler, x, {5, 128, 1, 441, 17});
  const auto d = whole.resampler.design();
  REQUIRE(a.size() == (x.size() * static_cast<std::size_t>(d.up) + d.down - 1) / d.down);
  REQUIRE(a == b);
  REQUIRE(a == c);
  // reset() returns to the initial state.
  whole.resampler.reset();
  REQUIRE(run(whole.resampler, x, {x.size()}) == a);
}

TEST_CASE("resampler round trip 16k -> device -> 16k keeps SNR above 90 dB", "[resampler]") {
  const int32_t device_rate = GENERATE(48000, 44100);
  INFO("device rate " << device_rate);
  const ResamplerDesign up = design(16000, device_rate);
  const ResamplerDesign down = design(device_rate, 16000);
  Stage to_device(up);
  Stage to_model(down);

  // Band-limited test signal: tones up to 6.1 kHz (the passband ends at 7 kHz).
  const double freqs[] = {173.0, 997.0, 2512.0, 4400.0, 6100.0};
  const double amps[] = {0.3, 0.25, 0.2, 0.1, 0.05};
  const double phases[] = {0.1, 1.3, 2.2, 0.7, 2.9};
  auto signal = [&](double t) {
    double v = 0.0;
    for (int k = 0; k < 5; ++k) v += amps[k] * std::sin(2.0 * kPi * freqs[k] * t + phases[k]);
    return v;
  };
  const std::size_t n = 16000;
  std::vector<float> x(n);
  for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<float>(signal(static_cast<double>(i) / 16000.0));

  const std::vector<float> mid = run(to_device.resampler, x, {128, 7, 480});
  const std::vector<float> y = run(to_model.resampler, mid, {441, 1, 1000});
  REQUIRE(y.size() >= n - 1);

  // Both prototypes are linear phase: y[i] = x(i / 16000 - D) with D the summed group delay.
  const double delay = up.delay_seconds() + down.delay_seconds();
  double signal_energy = 0.0;
  double error_energy = 0.0;
  for (std::size_t i = 400; i < y.size(); ++i) {
    const double want = signal(static_cast<double>(i) / 16000.0 - delay);
    signal_energy += want * want;
    error_energy += (static_cast<double>(y[i]) - want) * (static_cast<double>(y[i]) - want);
  }
  const double snr_db = 10.0 * std::log10(signal_energy / error_energy);
  std::printf("  round trip 16k -> %d -> 16k: SNR %.1f dB (delay %.3f samples)\n", device_rate, snr_db, delay * 16000.0);
  REQUIRE(snr_db > 90.0);
}

TEST_CASE("resampler rejects unsupported rates", "[resampler]") {
  ResamplerDesign d;
  REQUIRE_FALSE(earmark::design_resampler(0, 16000, &d));
  REQUIRE_FALSE(earmark::design_resampler(16000, -1, &d));
  REQUIRE_FALSE(earmark::design_resampler(44099, 16000, &d));  // up = 16000 > kResamplerMaxUp
  REQUIRE(earmark::design_resampler(16000, 16000, &d));
  REQUIRE(d.identity());
  REQUIRE(d.delay_seconds() == 0.0);
  REQUIRE(earmark::design_resampler(22050, 16000, &d));
  REQUIRE(d.up == 320);
  REQUIRE(d.down == 441);
}
