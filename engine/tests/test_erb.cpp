#include <cstring>
#include <vector>

#include "catch_amalgamated.hpp"
#include "erb.h"
#include "stft.h"
#include "test_support.h"

namespace {
constexpr int kBands = EARMARK_ERB_BANDS;
constexpr int kDfBins = EARMARK_DF_BINS;
constexpr int kBins = EARMARK_N_BINS;
constexpr int kSpec = earmark::kSpecFloats;
}  // namespace

TEST_CASE("ERB layout matches the contract and the golden", "[erb]") {
  const earmark_test::Golden golden("erb");
  const int32_t* widths = golden.i32("erb_widths");
  REQUIRE(golden.numel("erb_widths") == kBands);
  const earmark::ErbLayout& layout = earmark::erb_layout();
  int32_t start = 0;
  for (int b = 0; b < kBands; ++b) {
    REQUIRE(layout.width[static_cast<std::size_t>(b)] == widths[b]);
    REQUIRE(layout.start[static_cast<std::size_t>(b)] == start);
    start += widths[b];
  }
  REQUIRE(start == kBins);
}

TEST_CASE("ERB power, features and unit norm match the golden", "[erb]") {
  const earmark_test::Golden golden("erb");
  const std::size_t frames = golden.numel("spec") / kSpec;
  REQUIRE(frames == 32);
  const auto t32 = static_cast<uint32_t>(frames);
  const float* spec = golden.f32("spec", {t32, kBins, 2});
  const float* power = golden.f32("erb_power", {t32, kBands});
  const float* feat = golden.f32("erb_feat", {t32, kBands});
  const float* spec_feat = golden.f32("spec_feat", {t32, kDfBins, 2});
  const float* gains = golden.f32("gains", {t32, kBands});
  const float* gained = golden.f32("gained_spec", {t32, kBins, 2});

  float erb_norm[kBands];
  float spec_norm[kDfBins];
  std::memcpy(erb_norm, golden.f32("erb_norm_init", {kBands}), sizeof(erb_norm));
  std::memcpy(spec_norm, golden.f32("spec_norm_init", {kDfBins}), sizeof(spec_norm));

  earmark_test::ErrorStats power_total;
  earmark_test::ErrorStats feat_total;
  earmark_test::ErrorStats unit_total;
  earmark_test::ErrorStats gain_total;
  float ours_power[kBands];
  float ours_feat[kBands];
  float ours_unit[2 * kDfBins];
  float ours_gained[kSpec];
  for (std::size_t t = 0; t < frames; ++t) {
    const std::string frame = " frame " + std::to_string(t);
    const float* s = spec + t * kSpec;
    earmark::erb_band_power(s, ours_power);
    earmark_test::accumulate(power_total, earmark_test::require_close("erb.power" + frame, ours_power, power + t * kBands, kBands));
    earmark::erb_features_step(s, erb_norm, ours_feat);
    earmark_test::accumulate(feat_total, earmark_test::require_close("erb.feat" + frame, ours_feat, feat + t * kBands, kBands));
    earmark::unit_norm_step(s, spec_norm, ours_unit);
    earmark_test::accumulate(unit_total, earmark_test::require_close("erb.spec_feat" + frame, ours_unit,
                                                                     spec_feat + t * 2 * kDfBins, 2 * kDfBins));
    earmark::apply_erb_gains(gains + t * kBands, s, ours_gained);
    earmark_test::accumulate(gain_total, earmark_test::require_close("erb.gained_spec" + frame, ours_gained,
                                                                     gained + t * kSpec, kSpec));
  }
  earmark_test::report("erb.erb_power (per frame)", power_total);
  earmark_test::report("erb.erb_feat", feat_total);
  earmark_test::report("erb.spec_feat", unit_total);
  earmark_test::report("erb.gained_spec (per frame)", gain_total);
  earmark_test::report("erb.erb_norm_final", earmark_test::require_close("erb.erb_norm_final", erb_norm,
                                                                        golden.f32("erb_norm_final"), kBands));
  earmark_test::report("erb.spec_norm_final", earmark_test::require_close("erb.spec_norm_final", spec_norm,
                                                                         golden.f32("spec_norm_final"), kDfBins));
}

TEST_CASE("ERB gains may be applied in place", "[erb]") {
  std::vector<float> spec(kSpec);
  for (int i = 0; i < kSpec; ++i) spec[static_cast<std::size_t>(i)] = static_cast<float>(i % 7) - 3.0f;
  std::vector<float> separate(kSpec);
  float gains[kBands];
  for (int b = 0; b < kBands; ++b) gains[b] = 0.03f * static_cast<float>(b);
  earmark::apply_erb_gains(gains, spec.data(), separate.data());
  earmark::apply_erb_gains(gains, spec.data(), spec.data());
  REQUIRE(spec == separate);
}
