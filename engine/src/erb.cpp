#include "erb.h"

#include <cmath>

namespace earmark {

namespace {

constexpr int kBands = EARMARK_ERB_BANDS;
constexpr int kDfBins = EARMARK_DF_BINS;
constexpr float kAlpha = static_cast<float>(EARMARK_NORM_ALPHA);
constexpr float kOneMinusAlpha = static_cast<float>(1.0 - EARMARK_NORM_ALPHA);

constexpr ErbLayout make_layout() {
  constexpr int widths[] = EARMARK_ERB_WIDTHS_INIT;
  static_assert(sizeof(widths) / sizeof(widths[0]) == kBands, "ERB width count");
  ErbLayout layout{};
  int32_t start = 0;
  for (int b = 0; b < kBands; ++b) {
    layout.start[static_cast<std::size_t>(b)] = start;
    layout.width[static_cast<std::size_t>(b)] = widths[b];
    start += widths[b];
  }
  return layout;
}

constexpr ErbLayout kLayout = make_layout();
static_assert(kLayout.start[kBands - 1] + kLayout.width[kBands - 1] == EARMARK_N_BINS,
              "ERB bands must tile every bin");

}  // namespace

const ErbLayout& erb_layout() { return kLayout; }

void erb_band_power(const float* spec_ri, float* power) {
  for (int b = 0; b < kBands; ++b) {
    const int32_t start = kLayout.start[static_cast<std::size_t>(b)];
    const int32_t width = kLayout.width[static_cast<std::size_t>(b)];
    float sum = 0.0f;
    for (int32_t k = start; k < start + width; ++k) {
      const float re = spec_ri[2 * k];
      const float im = spec_ri[2 * k + 1];
      sum += re * re + im * im;
    }
    power[b] = sum / static_cast<float>(width);
  }
}

void erb_features_step(const float* spec_ri, float* norm, float* feat) {
  float power[kBands];
  erb_band_power(spec_ri, power);
  for (int b = 0; b < kBands; ++b) {
    const float db = 10.0f * std::log10(power[b] + kPowerEps);
    norm[b] = kAlpha * norm[b] + kOneMinusAlpha * db;
    feat[b] = (db - norm[b]) / kErbFeatureScaleDb;
  }
}

void unit_norm_step(const float* spec_ri, float* norm, float* feat_ri) {
  for (int k = 0; k < kDfBins; ++k) {
    const float re = spec_ri[2 * k];
    const float im = spec_ri[2 * k + 1];
    const float magnitude = std::sqrt(re * re + im * im);
    norm[k] = kAlpha * norm[k] + kOneMinusAlpha * magnitude;
    const float inv = 1.0f / std::sqrt(norm[k]);
    feat_ri[2 * k] = re * inv;
    feat_ri[2 * k + 1] = im * inv;
  }
}

void apply_erb_gains(const float* gains, const float* spec_ri, float* out_ri) {
  for (int b = 0; b < kBands; ++b) {
    const int32_t start = kLayout.start[static_cast<std::size_t>(b)];
    const int32_t end = start + kLayout.width[static_cast<std::size_t>(b)];
    const float g = gains[b];
    for (int32_t k = start; k < end; ++k) {
      out_ri[2 * k] = spec_ri[2 * k] * g;
      out_ri[2 * k + 1] = spec_ri[2 * k + 1] * g;
    }
  }
}

}  // namespace earmark
