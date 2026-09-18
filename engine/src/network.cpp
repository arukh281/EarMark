#include "network.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "matvec_f32.h"

namespace earmark {

namespace {

constexpr int32_t kEmbeddingDim = EARMARK_EMBEDDING_DIM;
constexpr int32_t kErbBands = EARMARK_ERB_BANDS;
constexpr int32_t kDfBins = EARMARK_DF_BINS;
constexpr int32_t kDfOrder = EARMARK_DF_ORDER;
/// F.normalize's eps in Conditioner.resolve.
constexpr float kNormalizeEps = 1e-8f;

/// A tensor name. Blob names are at most 71 bytes, so 96 is ample; a name that would not
/// fit is left empty, which matches no tensor.
struct Name {
  char text[96] = {};
};

/// `prefix` followed by `suffix`.
Name name_of(const char* prefix, const char* suffix) {
  Name name;
  const std::size_t head = std::strlen(prefix);
  const std::size_t tail = std::strlen(suffix);
  if (head + tail < sizeof(name.text)) {
    std::memcpy(name.text, prefix, head);
    std::memcpy(name.text + head, suffix, tail + 1);
  }
  return name;
}

/// `pattern` (one %d) formatted with `index`.
Name name_of(const char* pattern, int32_t index) {
  Name name;
  const int written = std::snprintf(name.text, sizeof(name.text), pattern, static_cast<int>(index));
  if (written < 0 || static_cast<std::size_t>(written) >= sizeof(name.text)) name.text[0] = '\0';
  return name;
}

/// A float32 tensor of rank `ndim`; false when absent, another dtype or another rank.
bool find_rank(const Blob& blob, const char* name, uint32_t ndim, TensorView* view) {
  return blob.find(name, view) && view->f32() != nullptr && view->ndim == ndim;
}

/// A float32 vector of exactly `length` values.
bool find_vector(const Blob& blob, const char* name, int32_t length, const float** out) {
  TensorView view;
  if (length <= 0 || !blob.find(name, &view) || view.f32() == nullptr ||
      !view.has_shape({static_cast<uint32_t>(length)})) {
    return false;
  }
  *out = view.f32();
  return true;
}

/// nn.Linear `prefix`.weight [out, in] and .bias [out]; `in` must match, `out` is read
/// from the shape when it is 0 and checked otherwise.
bool bind_linear(const Blob& blob, const char* prefix, int32_t in, int32_t out, Dense* dense) {
  TensorView w;
  if (!find_rank(blob, name_of(prefix, ".weight").text, 2, &w)) return false;
  const auto rows = static_cast<int32_t>(w.shape[0]);
  if (rows <= 0 || static_cast<int32_t>(w.shape[1]) != in || (out != 0 && rows != out)) return false;
  *dense = Dense{w.f32(), nullptr, in, rows, 1};
  return find_vector(blob, name_of(prefix, ".bias").text, rows, &dense->bias);
}

/// GroupedLinear `prefix`.weight [groups, out / groups, in / groups] and .bias [out];
/// `in` must match, `out` is read from the shape when it is 0 and checked otherwise.
bool bind_grouped(const Blob& blob, const char* prefix, int32_t in, int32_t out, Dense* dense) {
  TensorView w;
  if (!find_rank(blob, name_of(prefix, ".weight").text, 3, &w)) return false;
  const auto groups = static_cast<int32_t>(w.shape[0]);
  if (groups <= 0 || in % groups != 0 || static_cast<int32_t>(w.shape[2]) != in / groups) return false;
  const int32_t rows = groups * static_cast<int32_t>(w.shape[1]);
  if (rows <= 0 || (out != 0 && rows != out)) return false;
  *dense = Dense{w.f32(), nullptr, in, rows, groups};
  return find_vector(blob, name_of(prefix, ".bias").text, rows, &dense->bias);
}

/// One FreqConvBranch: `prefix`.first then `prefix`.down.0, .down.1, ... (as many as exist).
bool bind_branch(const Blob& blob, const char* prefix, int32_t in_channels, int32_t in_bins, FreqBranch* branch) {
  TensorView w;
  if (!find_rank(blob, name_of(prefix, ".first.weight").text, 4, &w)) return false;
  const auto channels = static_cast<int32_t>(w.shape[0]);
  if (channels <= 0 || static_cast<int32_t>(w.shape[1]) != in_channels || w.shape[2] != 2 || w.shape[3] != 3) {
    return false;
  }
  *branch = FreqBranch{};
  branch->first = FreqConv{w.f32(), nullptr, in_channels, channels, 1, 2};
  branch->in_bins = in_bins;
  branch->channels = channels;
  if (!find_vector(blob, name_of(prefix, ".first.bias").text, channels, &branch->first.bias)) return false;

  int32_t bins = in_bins;
  const Name down_pattern = name_of(prefix, ".down.%d");  // prefixes are literals without '%'
  for (int32_t i = 0; i <= kMaxDownsamples; ++i) {
    const Name base = name_of(down_pattern.text, i);
    TensorView probe;
    if (!blob.find(name_of(base.text, ".weight").text, &probe)) break;
    if (i == kMaxDownsamples) return false;  // more stride-2 convs than the engine supports
    if (!find_rank(blob, name_of(base.text, ".weight").text, 4, &w)) return false;
    const auto per_group = static_cast<int32_t>(w.shape[1]);
    if (static_cast<int32_t>(w.shape[0]) != channels || per_group <= 0 || channels % per_group != 0 ||
        w.shape[2] != 1 || w.shape[3] != 3 || bins < 2 || bins % 2 != 0) {
      return false;
    }
    FreqConv& conv = branch->down[i];
    conv = FreqConv{w.f32(), nullptr, channels, channels, channels / per_group, 1};
    if (!find_vector(blob, name_of(base.text, ".bias").text, channels, &conv.bias)) return false;
    bins /= 2;
    branch->downsamples = i + 1;
  }
  return true;
}

void apply(const Dense& dense, const float* x, float* y) {
  if (dense.groups == 1) {
    matvec_f32(dense.weight, dense.out, dense.in, x, dense.bias, y);
  } else {
    grouped_matvec_f32(dense.weight, dense.in, dense.out, dense.groups, x, dense.bias, y);
  }
}

inline float relu(float v) { return v > 0.0f ? v : 0.0f; }
inline float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

/// The first conv: kernel (2, 3) over frames (t-1, t), frequency padding 1, ungrouped.
/// `prev`, `cur` [in, bins] -> `y` [out, bins], ReLU applied.
void conv_first(const FreqConv& conv, const float* prev, const float* cur, int32_t bins, float* y) {
  for (int32_t o = 0; o < conv.out; ++o) {
    for (int32_t f = 0; f < bins; ++f) {
      float acc = 0.0f;
      for (int32_t c = 0; c < conv.in; ++c) {
        const float* w = conv.weight + (static_cast<std::ptrdiff_t>(o) * conv.in + c) * 6;  // [2, 3]
        const float* frames[2] = {prev + static_cast<std::ptrdiff_t>(c) * bins,
                                  cur + static_cast<std::ptrdiff_t>(c) * bins};
        for (int32_t t = 0; t < 2; ++t) {
          for (int32_t k = 0; k < 3; ++k) {
            const int32_t bin = f + k - 1;
            if (bin >= 0 && bin < bins) acc += w[t * 3 + k] * frames[t][bin];
          }
        }
      }
      y[static_cast<std::ptrdiff_t>(o) * bins + f] = relu(acc + conv.bias[o]);
    }
  }
}

/// A stride-2 conv: kernel (1, 3), frequency padding 1, `groups` channel groups.
/// `x` [in, bins] -> `y` [out, bins / 2], ReLU applied.
void conv_down(const FreqConv& conv, const float* x, int32_t bins, float* y) {
  const int32_t in_per_group = conv.in / conv.groups;
  const int32_t out_per_group = conv.out / conv.groups;
  const int32_t out_bins = bins / 2;
  for (int32_t o = 0; o < conv.out; ++o) {
    const int32_t first_in = (o / out_per_group) * in_per_group;
    for (int32_t j = 0; j < out_bins; ++j) {
      float acc = 0.0f;
      for (int32_t c = 0; c < in_per_group; ++c) {
        const float* w = conv.weight + (static_cast<std::ptrdiff_t>(o) * in_per_group + c) * 3;
        const float* row = x + static_cast<std::ptrdiff_t>(first_in + c) * bins;
        for (int32_t k = 0; k < 3; ++k) {
          const int32_t bin = 2 * j + k - 1;
          if (bin >= 0 && bin < bins) acc += w[k] * row[bin];
        }
      }
      y[static_cast<std::ptrdiff_t>(o) * out_bins + j] = relu(acc + conv.bias[o]);
    }
  }
}

/// A whole branch into `out` [out_features], using `a` and `b` as scratch.
void run_branch(const FreqBranch& branch, const float* prev, const float* cur, float* a, float* b, float* out) {
  int32_t bins = branch.in_bins;
  float* src = branch.downsamples == 0 ? out : a;
  conv_first(branch.first, prev, cur, bins, src);
  for (int32_t i = 0; i < branch.downsamples; ++i) {
    float* dst = i == branch.downsamples - 1 ? out : (src == a ? b : a);
    conv_down(branch.down[i], src, bins, dst);
    bins /= 2;
    src = dst;
  }
}

std::size_t floats_bytes(std::size_t count) { return Arena::aligned(count * sizeof(float)); }

}  // namespace

NetworkBind Network::bind(const Blob& blob) {
  *this = Network();
  static constexpr const char* kMarkers[] = {
      "erb_enc.first.weight", "df_enc.first.weight", "enc_proj.weight", "conditioner.proj.weight",
      "body.gru.weight_ih_l0", "vad_head.weight",    "gain_head.weight", "df_head.weight",
  };
  bool any = false;
  for (const char* marker : kMarkers) {
    TensorView probe;
    any = any || blob.find(marker, &probe);
  }
  if (!any) return NetworkBind::kAbsent;

  const bool ok = [&]() {
    if (!bind_branch(blob, "erb_enc", 1, kErbBands, &erb_enc_) || !bind_branch(blob, "df_enc", 2, kDfBins, &df_enc_)) {
      return false;
    }
    if (!bind_grouped(blob, "enc_proj", erb_enc_.out_features() + df_enc_.out_features(), 0, &enc_proj_)) {
      return false;
    }
    hidden_ = enc_proj_.out;
    if (!find_vector(blob, "conditioner.null_embedding", kEmbeddingDim, &null_embedding_) ||
        !bind_linear(blob, "conditioner.proj", kEmbeddingDim, 0, &cond_proj_)) {
      return false;
    }
    rank_ = cond_proj_.out;
    if (!bind_linear(blob, "conditioner.pre", rank_, 2 * hidden_, &cond_pre_) ||
        !bind_linear(blob, "conditioner.post", rank_, 2 * hidden_, &cond_post_)) {
      return false;
    }
    const int32_t gates = 3 * hidden_;
    for (int32_t k = 0; k <= kMaxGruLayers; ++k) {
      TensorView probe;
      if (!blob.find(name_of("body.gru.weight_ih_l%d", k).text, &probe)) break;
      if (k == kMaxGruLayers) return false;
      GruLayer& layer = gru_[k];
      TensorView w_ih;
      TensorView w_hh;
      if (!find_rank(blob, name_of("body.gru.weight_ih_l%d", k).text, 2, &w_ih) ||
          !w_ih.has_shape({static_cast<uint32_t>(gates), static_cast<uint32_t>(hidden_)}) ||
          !find_rank(blob, name_of("body.gru.weight_hh_l%d", k).text, 2, &w_hh) ||
          !w_hh.has_shape({static_cast<uint32_t>(gates), static_cast<uint32_t>(hidden_)}) ||
          !find_vector(blob, name_of("body.gru.bias_ih_l%d", k).text, gates, &layer.bias_ih) ||
          !find_vector(blob, name_of("body.gru.bias_hh_l%d", k).text, gates, &layer.bias_hh)) {
        return false;
      }
      layer.weight_ih = w_ih.f32();
      layer.weight_hh = w_hh.f32();
      layer.input = hidden_;
      layer.hidden = hidden_;
      layers_ = k + 1;
    }
    if (layers_ == 0) return false;  // no GRU body (S-SSM is not implemented)
    return bind_linear(blob, "vad_head", hidden_, 1, &vad_head_) &&
           bind_linear(blob, "gain_head", hidden_, kErbBands, &gain_head_) &&
           bind_grouped(blob, "df_head", hidden_, kDfOutputs, &df_head_);
  }();
  if (!ok) {
    *this = Network();
    return NetworkBind::kInvalid;
  }
  conv_floats_ = static_cast<std::size_t>(erb_enc_.channels) * erb_enc_.in_bins;
  const std::size_t df_floats = static_cast<std::size_t>(df_enc_.channels) * df_enc_.in_bins;
  if (df_floats > conv_floats_) conv_floats_ = df_floats;
  return NetworkBind::kOk;
}

std::size_t Network::arena_bytes() const {
  const auto hidden = static_cast<std::size_t>(hidden_);
  return floats_bytes(kEmbeddingDim) + floats_bytes(static_cast<std::size_t>(rank_)) + floats_bytes(2 * hidden) +
         4 * floats_bytes(hidden) + floats_bytes(body_state_floats()) + floats_bytes(gru_scratch_floats(hidden_)) +
         floats_bytes(2 * kDfBins) + 2 * floats_bytes(conv_floats_) + floats_bytes(static_cast<std::size_t>(enc_proj_.in)) +
         3 * floats_bytes(hidden) + floats_bytes(kDfOutputs);
}

bool Network::init(Arena& arena) {
  const auto hidden = static_cast<std::size_t>(hidden_);
  unit_embedding_ = arena.allocate_array<float>(kEmbeddingDim);
  code_ = arena.allocate_array<float>(static_cast<std::size_t>(rank_));
  film_tmp_ = arena.allocate_array<float>(2 * hidden);
  pre_scale_ = arena.allocate_array<float>(hidden);
  pre_shift_ = arena.allocate_array<float>(hidden);
  post_scale_ = arena.allocate_array<float>(hidden);
  post_shift_ = arena.allocate_array<float>(hidden);
  body_ = arena.allocate_array<float>(body_state_floats());
  gru_scratch_ = arena.allocate_array<float>(gru_scratch_floats(hidden_));
  df_in_ = arena.allocate_array<float>(2 * kDfBins);
  conv_a_ = arena.allocate_array<float>(conv_floats_);
  conv_b_ = arena.allocate_array<float>(conv_floats_);
  enc_cat_ = arena.allocate_array<float>(static_cast<std::size_t>(enc_proj_.in));
  enc_ = arena.allocate_array<float>(hidden);
  film_pre_ = arena.allocate_array<float>(hidden);
  film_post_ = arena.allocate_array<float>(hidden);
  df_raw_ = arena.allocate_array<float>(kDfOutputs);
  const float* buffers[] = {unit_embedding_, code_,  film_tmp_, pre_scale_, pre_shift_, post_scale_,
                            post_shift_,     body_,  gru_scratch_, df_in_, conv_a_,   conv_b_,
                            enc_cat_,        enc_,   film_pre_, film_post_, df_raw_};
  for (const float* buffer : buffers) {
    if (buffer == nullptr) return false;
  }
  enc_erb_ = enc_cat_;
  enc_df_ = enc_cat_ + erb_enc_.out_features();
  return true;
}

void Network::condition(const float* embedding) {
  const float* e = embedding != nullptr ? embedding : null_embedding_;
  const float norm = std::sqrt(dot_f32(e, e, kEmbeddingDim));
  const float denom = norm > kNormalizeEps ? norm : kNormalizeEps;
  for (int32_t i = 0; i < kEmbeddingDim; ++i) unit_embedding_[i] = e[i] / denom;
  apply(cond_proj_, unit_embedding_, code_);
  for (int32_t r = 0; r < rank_; ++r) code_[r] = std::tanh(code_[r]);
  apply(cond_pre_, code_, film_tmp_);
  for (int32_t j = 0; j < hidden_; ++j) {
    pre_scale_[j] = 1.0f + film_tmp_[j];
    pre_shift_[j] = film_tmp_[hidden_ + j];
  }
  apply(cond_post_, code_, film_tmp_);
  for (int32_t j = 0; j < hidden_; ++j) {
    post_scale_[j] = 1.0f + film_tmp_[j];
    post_shift_[j] = film_tmp_[hidden_ + j];
  }
}

void Network::reset() { std::memset(body_, 0, body_state_floats() * sizeof(float)); }

float Network::step(const float* erb_feat, const float* spec_feat_ri, float* enc_erb_prev, float* enc_df_prev,
                    float* gains, float* df_taps) {
  for (int32_t k = 0; k < kDfBins; ++k) {  // (re, im) interleaved -> re channel, im channel
    df_in_[k] = spec_feat_ri[2 * k];
    df_in_[kDfBins + k] = spec_feat_ri[2 * k + 1];
  }
  run_branch(erb_enc_, enc_erb_prev, erb_feat, conv_a_, conv_b_, enc_erb_);
  run_branch(df_enc_, enc_df_prev, df_in_, conv_a_, conv_b_, enc_df_);
  std::memcpy(enc_erb_prev, erb_feat, kErbBands * sizeof(float));
  std::memcpy(enc_df_prev, df_in_, 2 * kDfBins * sizeof(float));

  apply(enc_proj_, enc_cat_, enc_);
  for (int32_t j = 0; j < hidden_; ++j) {
    enc_[j] = relu(enc_[j]);
    film_pre_[j] = enc_[j] * pre_scale_[j] + pre_shift_[j];
  }
  gru_stack_step(gru_, layers_, film_pre_, body_, gru_scratch_);
  const float* out = body_out();

  float vad_logit = 0.0f;
  apply(vad_head_, out, &vad_logit);
  for (int32_t j = 0; j < hidden_; ++j) film_post_[j] = out[j] * post_scale_[j] + post_shift_[j];
  apply(gain_head_, film_post_, gains);
  for (int32_t b = 0; b < kErbBands; ++b) gains[b] = sigmoid(gains[b]);
  apply(df_head_, film_post_, df_raw_);
  for (int32_t i = 0; i < kDfOutputs; ++i) df_taps[i] = std::tanh(df_raw_[i]);
  for (int32_t k = 0; k < kDfBins; ++k) df_taps[k * kDfOrder * 2] += 1.0f;  // identity on tap 0 (re)
  return vad_logit;
}

void deep_filter_step(const float* taps, float* spec_ri, float* df_hist) {
  for (int32_t k = 0; k < kDfBins; ++k) {
    const float xr = spec_ri[2 * k];
    const float xi = spec_ri[2 * k + 1];
    const float* c = taps + k * kDfOrder * 2;
    float yr = c[0] * xr - c[1] * xi;
    float yi = c[0] * xi + c[1] * xr;
    for (int32_t i = 1; i < kDfOrder; ++i) {  // tap i multiplies frame t - i = df_hist[DF_ORDER - 1 - i]
      const float* h = df_hist + ((kDfOrder - 1 - i) * kDfBins + k) * 2;
      const float cr = c[2 * i];
      const float ci = c[2 * i + 1];
      yr = yr + (cr * h[0] - ci * h[1]);
      yi = yi + (cr * h[1] + ci * h[0]);
    }
    for (int32_t i = 0; i + 1 < kDfOrder - 1; ++i) {  // shift: oldest first, then the unfiltered input
      float* dst = df_hist + (i * kDfBins + k) * 2;
      const float* src = df_hist + ((i + 1) * kDfBins + k) * 2;
      dst[0] = src[0];
      dst[1] = src[1];
    }
    float* newest = df_hist + ((kDfOrder - 2) * kDfBins + k) * 2;
    newest[0] = xr;
    newest[1] = xi;
    spec_ri[2 * k] = yr;
    spec_ri[2 * k + 1] = yi;
  }
}

}  // namespace earmark
