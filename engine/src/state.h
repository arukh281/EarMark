// Fixed-size per-stream state (the C++ side of earmark.model.stream.StreamState).
//
// Fields follow earmark.model.stream.STATE_FIELDS in order, float32, one stream. The last
// field, body (GRU [layers, hidden]), depends on the model, so it lives in the network's
// arena (Network::body_state) rather than in this struct.
#pragma once

#include <cstddef>

#include "earmark_constants.h"

namespace earmark {

/// Floats in StreamState::df_hist: [DF_ORDER - 1, DF_BINS, 2].
inline constexpr int kDfHistFloats = (EARMARK_DF_ORDER - 1) * EARMARK_DF_BINS * 2;

struct StreamState {
  float in_buf[EARMARK_HOP_LENGTH];        ///< previous input hop (first half of the next frame)
  float ola_buf[EARMARK_HOP_LENGTH];       ///< synthesis overlap tail
  float erb_norm[EARMARK_ERB_BANDS];       ///< running mean of ERB log-power, dB
  float spec_norm[EARMARK_DF_BINS];        ///< running mean of |X| on bins [0, DF_BINS)
  float enc_erb_prev[EARMARK_ERB_BANDS];   ///< previous ERB feature frame [1, ERB_BANDS]
  float enc_df_prev[2 * EARMARK_DF_BINS];  ///< previous low-band features [2, DF_BINS]: re, then im
  float df_hist[kDfHistFloats];            ///< gained low band at t-2, t-1: [DF_ORDER - 1, DF_BINS, (re, im)]
};

static_assert(sizeof(StreamState) == sizeof(float) * (2 * EARMARK_HOP_LENGTH + 2 * EARMARK_ERB_BANDS +
                                                      3 * EARMARK_DF_BINS + kDfHistFloats),
              "StreamState must be a packed array of floats");

/// Zeros the buffers and loads the normalisation means from the blob's const tensors.
void reset_state(StreamState& state, const float* erb_norm_init, const float* spec_norm_init);

}  // namespace earmark
