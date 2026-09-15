// Fixed-size per-stream state (the C++ side of earmark.model.stream.StreamState).
//
// Fields follow earmark.model.stream.STATE_FIELDS in order, float32, one stream. The
// week-1 skeleton holds the signal-path prefix of that list. The model fields come next,
// in this order, when the network is wired in: enc_erb_prev [1, 32], enc_df_prev [2, 64],
// df_hist [2, 64, 2] (t-2 then t-1, re/im) and body (GRU [layers, hidden]).
#pragma once

#include <cstddef>

#include "earmark_constants.h"

namespace earmark {

struct StreamState {
  float in_buf[EARMARK_HOP_LENGTH];   ///< previous input hop (first half of the next frame)
  float ola_buf[EARMARK_HOP_LENGTH];  ///< synthesis overlap tail
  float erb_norm[EARMARK_ERB_BANDS];  ///< running mean of ERB log-power, dB
  float spec_norm[EARMARK_DF_BINS];   ///< running mean of |X| on bins [0, DF_BINS)
};

static_assert(sizeof(StreamState) ==
                  sizeof(float) * (2 * EARMARK_HOP_LENGTH + EARMARK_ERB_BANDS + EARMARK_DF_BINS),
              "StreamState must be a packed array of floats");

/// Zeros the buffers and loads the normalisation means from the blob's const tensors.
void reset_state(StreamState& state, const float* erb_norm_init, const float* spec_norm_init);

}  // namespace earmark
