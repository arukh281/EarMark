// The Earmark network for one hop: the C++ side of EarmarkNet.step (earmark_net.py).
//
// Per hop, from the features the signal path already computes:
//
//   erb_feat [32] ──► erb_enc: conv (2 x 3, frames t-1 and t), then stride-2 convs ─┐
//   spec_feat [64 re/im] ──► df_enc: the same, 64 -> 8 bins ────────────────────────┤
//     concat (ERB branch first, each channel-major) ► enc_proj + ReLU ► FiLM pre ► GRU stack
//     GRU output ► vad_head (the VAD reads the body before post-FiLM)
//     GRU output ► FiLM post ► gain_head + sigmoid (32 ERB gains)
//                            ► df_head, tanh, +1 on the real part of tap 0 (deep-filter taps)
//
// Weights are views into the blob under their PyTorch state_dict names. Every dimension
// (encoder channels and groups, hidden size, GRU layers, FiLM rank, head groups) is read
// from the tensor shapes and cross-checked, so one binary serves M, M-256 and S-GRU. The
// S-SSM body is not implemented: its blobs bind as kInvalid.
//
// The engine owns the per-stream state that has a fixed size (enc_erb_prev, enc_df_prev,
// df_hist in StreamState); the GRU state, whose size depends on the model, lives here.
#pragma once

#include <cstddef>
#include <cstdint>

#include "arena.h"
#include "earmark_constants.h"
#include "gru.h"
#include "weights.h"

namespace earmark {

/// Deep-filter head outputs: DF_BINS x DF_ORDER x (re, im).
inline constexpr int32_t kDfOutputs = EARMARK_DF_BINS * EARMARK_DF_ORDER * 2;
/// Most GRU layers a blob may have.
inline constexpr int32_t kMaxGruLayers = 4;
/// Most stride-2 convs per encoder branch.
inline constexpr int32_t kMaxDownsamples = 4;

/// Whether a blob holds a network, and whether it is one this engine can run.
enum class NetworkBind : int32_t {
  kAbsent = 0,  ///< no network tensors at all (a signal-path-only blob)
  kOk,
  kInvalid,     ///< some network tensors, but missing, mis-shaped or an unsupported body
};

/// One Conv2d over [channels, bins] feature maps with a 3-bin frequency kernel.
struct FreqConv {
  const float* weight = nullptr;  ///< [out, in / groups, frames, 3]
  const float* bias = nullptr;    ///< [out]
  int32_t in = 0;
  int32_t out = 0;
  int32_t groups = 1;
  int32_t frames = 1;  ///< 2 for the first conv (t-1, t), 1 for the stride-2 convs
};

/// One encoder branch: the first conv, then `downsamples` stride-2 convs, ReLU after each.
struct FreqBranch {
  FreqConv first;
  FreqConv down[kMaxDownsamples];
  int32_t downsamples = 0;
  int32_t in_bins = 0;
  int32_t channels = 0;

  int32_t out_bins() const { return in_bins >> downsamples; }
  int32_t out_features() const { return channels * out_bins(); }
};

/// A GroupedLinear (groups == 1 is a plain nn.Linear stored as [1, out, in]) or nn.Linear.
struct Dense {
  const float* weight = nullptr;
  const float* bias = nullptr;
  int32_t in = 0;
  int32_t out = 0;
  int32_t groups = 1;
};

class Network {
 public:
  /// Binds the network tensors of a parsed blob (views; the blob must outlive this).
  NetworkBind bind(const Blob& blob);

  /// Arena bytes for the conditioning, the GRU state and all scratch (after bind).
  std::size_t arena_bytes() const;
  /// Carves every buffer from `arena`. False when it runs out.
  bool init(Arena& arena);

  /// FiLM parameters for `embedding` [EMBEDDING_DIM], which is L2-normalised first;
  /// `embedding` == nullptr selects conditioner.null_embedding (Denoise mode).
  void condition(const float* embedding);
  /// Zeros the GRU state.
  void reset();

  /// One hop. `erb_feat` [ERB_BANDS] and `spec_feat_ri` [DF_BINS * 2] are this hop's
  /// features; `enc_erb_prev` [ERB_BANDS] and `enc_df_prev` [2 * DF_BINS] hold the previous
  /// hop's and are updated. Writes `gains` [ERB_BANDS] and the effective deep-filter taps
  /// `df_taps` [DF_BINS, DF_ORDER, 2]; returns the VAD logit.
  float step(const float* erb_feat, const float* spec_feat_ri, float* enc_erb_prev, float* enc_df_prev,
             float* gains, float* df_taps);

  int32_t hidden() const { return hidden_; }
  int32_t layers() const { return layers_; }
  std::size_t body_state_floats() const { return static_cast<std::size_t>(layers_) * hidden_; }

  // Intermediates of the last step (and the current conditioning), for the tests.
  const float* enc_erb() const { return enc_erb_; }
  const float* enc_df() const { return enc_df_; }
  const float* enc() const { return enc_; }
  const float* film_pre() const { return film_pre_; }
  const float* body_out() const { return body_ + static_cast<std::ptrdiff_t>(layers_ - 1) * hidden_; }
  const float* film_post() const { return film_post_; }
  const float* df_raw() const { return df_raw_; }
  const float* pre_scale() const { return pre_scale_; }
  const float* pre_shift() const { return pre_shift_; }
  const float* post_scale() const { return post_scale_; }
  const float* post_shift() const { return post_shift_; }

 private:
  FreqBranch erb_enc_;
  FreqBranch df_enc_;
  Dense enc_proj_;
  const float* null_embedding_ = nullptr;
  Dense cond_proj_;
  Dense cond_pre_;
  Dense cond_post_;
  GruLayer gru_[kMaxGruLayers];
  Dense vad_head_;
  Dense gain_head_;
  Dense df_head_;
  int32_t hidden_ = 0;
  int32_t layers_ = 0;
  int32_t rank_ = 0;

  // Arena buffers.
  float* unit_embedding_ = nullptr;  // [EMBEDDING_DIM]
  float* code_ = nullptr;            // [rank]
  float* film_tmp_ = nullptr;        // [2 * hidden]
  float* pre_scale_ = nullptr;       // [hidden] each
  float* pre_shift_ = nullptr;
  float* post_scale_ = nullptr;
  float* post_shift_ = nullptr;
  float* body_ = nullptr;            // [layers, hidden] GRU state
  float* gru_scratch_ = nullptr;     // [gru_scratch_floats(hidden)]
  float* df_in_ = nullptr;           // [2 * DF_BINS] this hop's low band, re then im
  float* conv_a_ = nullptr;          // [max channels * max bins] ping-pong conv buffers
  float* conv_b_ = nullptr;
  float* enc_cat_ = nullptr;         // [enc_proj.in]: enc_erb then enc_df
  float* enc_erb_ = nullptr;         // views into enc_cat_
  float* enc_df_ = nullptr;
  float* enc_ = nullptr;             // [hidden]
  float* film_pre_ = nullptr;        // [hidden]
  float* film_post_ = nullptr;       // [hidden]
  float* df_raw_ = nullptr;          // [kDfOutputs]
  std::size_t conv_floats_ = 0;
};

/// Order-DF_ORDER complex deep filter on bins [0, DF_BINS) of `spec_ri` (interleaved,
/// updated in place): Y[k] = sum_i taps[k, i] * X[t - i, k] with X[t] the current value and
/// X[t - i] from `df_hist` ([DF_ORDER - 1, DF_BINS, 2], oldest first). `df_hist` then shifts
/// to hold the unfiltered input up to t. Bins >= DF_BINS are left untouched.
void deep_filter_step(const float* taps, float* spec_ri, float* df_hist);

}  // namespace earmark
