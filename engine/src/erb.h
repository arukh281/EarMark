// ERB band features, low-band unit normalisation and ERB gain expansion, one frame at a
// time (the step versions in earmark.model.dsp).
//
// Spectra are interleaved (re, im) [N_BINS * 2]. Bands tile the bins with the contract's
// ERB_WIDTHS: band b covers [start[b], start[b] + width[b]).
//
// * ERB power: mean of |X|^2 over each band.
// * ERB features: db = 10 log10(power + kPowerEps); m = a m + (1 - a) db (update first,
//   a = NORM_ALPHA); feature = (db - m) / kErbFeatureScaleDb.
// * Unit norm: s = a s + (1 - a) |X| on bins [0, DF_BINS) (update first); feature =
//   X / sqrt(s), interleaved [DF_BINS * 2].
// * Gains: every bin of band b is multiplied by gains[b] (rectangular expansion).
//
// kPowerEps and kErbFeatureScaleDb are model conventions, recorded in every export
// manifest under conventions.power_eps and conventions.erb_feature_scale_db.
#pragma once

#include <array>
#include <cstdint>

#include "earmark_constants.h"

namespace earmark {

inline constexpr float kPowerEps = 1e-10f;
inline constexpr float kErbFeatureScaleDb = 40.0f;

/// First bin of each ERB band, from the contract widths.
struct ErbLayout {
  std::array<int32_t, EARMARK_ERB_BANDS> start{};
  std::array<int32_t, EARMARK_ERB_BANDS> width{};
};

/// The contract layout (computed at compile time).
const ErbLayout& erb_layout();

/// `spec_ri` -> mean band power `power` [ERB_BANDS].
void erb_band_power(const float* spec_ri, float* power);

/// Normalised ERB log-power for one frame; updates `norm` [ERB_BANDS] (dB) in place.
void erb_features_step(const float* spec_ri, float* norm, float* feat);

/// Unit-normalised low band for one frame; updates `norm` [DF_BINS] in place and writes
/// `feat_ri` [DF_BINS * 2].
void unit_norm_step(const float* spec_ri, float* norm, float* feat_ri);

/// out_ri = spec_ri * gains expanded over the bands. `out_ri` may alias `spec_ri`.
void apply_erb_gains(const float* gains, const float* spec_ri, float* out_ri);

}  // namespace earmark
