// The C API (include/earmark.h) over the signal path and the network.
//
// Signal flow of em_process at device rate R:
//
//   in (R) -> Resampler R->16k -> model_in FIFO -> hops of 160 -> run_hop()
//          -> Resampler 16k->R -> device_out FIFO (primed with P zeros) -> out (R)
//
// run_hop() is EarmarkNet.step: analysis, features, the network, ERB gains, deep filter,
// synthesis. The network's buffers come from a second arena (net_arena), reserved once
// the blob has been parsed and its shapes are known.
//
// P = ceil(159 R / 16000) zeros guarantee that device_out never runs dry: after I input
// samples at least I - 159 R / 16000 output samples have been produced (at most 159
// model-rate samples wait in model_in for a full hop), so with the priming every read
// of I samples succeeds.
#include "earmark.h"

#include <cmath>
#include <cstring>
#include <new>

#include "arena.h"
#include "erb.h"
#include "matvec_f32.h"
#include "network.h"
#include "resampler.h"
#include "ringbuf.h"
#include "state.h"
#include "stft.h"
#include "weights.h"

namespace {

using earmark::Arena;

constexpr int kHop = EARMARK_HOP_LENGTH;
constexpr int32_t kModelRate = EARMARK_SAMPLE_RATE;
constexpr int kEmbeddingDim = EARMARK_EMBEDDING_DIM;
/// Device samples handled per internal iteration of em_process (bounds every buffer).
constexpr std::size_t kChunk = 512;

constexpr const char* kErbNormInit = "const.erb_norm_init";
constexpr const char* kSpecNormInit = "const.spec_norm_init";
constexpr const char* kNullEmbedding = "conditioner.null_embedding";

std::size_t floats_bytes(std::size_t count) { return Arena::aligned(count * sizeof(float)); }

}  // namespace

struct em_engine {
  Arena arena;
  Arena net_arena;  // network buffers and GRU state (empty without a network)
  earmark::Blob blob;
  earmark::Network net;
  bool has_network = false;
  int32_t device_rate = 0;
  earmark::ResamplerDesign in_design{};   // device -> 16 kHz
  earmark::ResamplerDesign out_design{};  // 16 kHz -> device
  earmark::Resampler to_model;
  earmark::Resampler from_model;
  earmark::RingBuffer model_in;    // 16 kHz samples waiting for a full hop
  earmark::RingBuffer device_out;  // device-rate output FIFO
  earmark::Stft stft;
  earmark::StreamState state{};

  const float* erb_norm_init = nullptr;   // blob views
  const float* spec_norm_init = nullptr;
  const float* null_embedding = nullptr;

  float* embedding = nullptr;   // [EMBEDDING_DIM] current conditioning
  bool personal = false;
  float* chunk_model = nullptr; // [to_model max output for kChunk]
  float* hop_in = nullptr;      // [HOP]
  float* hop_out = nullptr;     // [HOP]
  float* hop_device = nullptr;  // [from_model max output for HOP]
  float* spec = nullptr;        // [N_BINS * 2]
  float* spec_out = nullptr;    // [N_BINS * 2]
  float* erb_feat = nullptr;    // [ERB_BANDS]
  float* spec_feat = nullptr;   // [DF_BINS * 2]
  float* gains = nullptr;       // [ERB_BANDS]
  float* df_taps = nullptr;     // [DF_BINS, DF_ORDER, 2] (network only)

  std::size_t prime = 0;
  float vad = 0.0f;
  uint64_t xruns = 0;
  std::size_t state_bytes = 0;
};

namespace {

// One 16 kHz hop: EarmarkNet.step. Without a network the gains stay at 1 and there is
// no deep filter, so out is the input delayed by one hop and the VAD is 0.
void run_hop(em_engine& e, const float* in, float* out) {
  e.stft.analyze(in, e.state.in_buf, e.spec);
  earmark::erb_features_step(e.spec, e.state.erb_norm, e.erb_feat);
  earmark::unit_norm_step(e.spec, e.state.spec_norm, e.spec_feat);
  if (!e.has_network) {
    earmark::apply_erb_gains(e.gains, e.spec, e.spec_out);
    e.stft.synthesize(e.spec_out, e.state.ola_buf, out);
    e.vad = 0.0f;
    return;
  }
  const float vad_logit = e.net.step(e.erb_feat, e.spec_feat, e.state.enc_erb_prev, e.state.enc_df_prev,
                                     e.gains, e.df_taps);
  earmark::apply_erb_gains(e.gains, e.spec, e.spec_out);
  earmark::deep_filter_step(e.df_taps, e.spec_out, e.state.df_hist);
  e.stft.synthesize(e.spec_out, e.state.ola_buf, out);
  e.vad = 1.0f / (1.0f + std::exp(-vad_logit));
}

bool find_f32(const earmark::Blob& blob, const char* name, uint32_t length, const float** out) {
  earmark::TensorView view;
  if (!blob.find(name, &view) || view.f32() == nullptr || !view.has_shape({length})) return false;
  *out = view.f32();
  return true;
}

bool all_finite(const float* values, int count) {
  for (int i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) return false;
  }
  return true;
}

}  // namespace

extern "C" {

uint32_t em_abi_version(void) { return EM_ABI_VERSION; }

const char* em_contract_hash(void) { return EARMARK_CONTRACT_HASH; }

const char* em_status_string(em_status status) {
  switch (status) {
    case EM_OK: return "ok";
    case EM_ERR_ARGUMENT: return "invalid argument";
    case EM_ERR_BLOB: return "weight blob is malformed or corrupt";
    case EM_ERR_CONTRACT: return "weights were exported under a different signal contract";
    case EM_ERR_MANIFEST: return "manifest is malformed or does not match the blob";
    case EM_ERR_MISSING_TENSOR: return "a required tensor is missing or has the wrong shape";
    case EM_ERR_RATE: return "unsupported device sample rate";
    case EM_ERR_NO_MEMORY: return "out of memory";
    case EM_ERR_FFT: return "FFT plan could not be built";
    default: return "unknown status";
  }
}

const char* em_build_info(void) {
#if defined(__wasm_simd128__)
  return "earmark-engine abi=2 contract=" EARMARK_CONTRACT_HASH " matvec=wasm-simd128 network=gru";
#else
  return "earmark-engine abi=2 contract=" EARMARK_CONTRACT_HASH " matvec=scalar network=gru";
#endif
}

em_engine* em_create(const uint8_t* blob, size_t blob_bytes, const char* manifest, size_t manifest_bytes,
                     int32_t device_rate, em_status* status) {
  em_status local_status = EM_OK;
  em_status* result = status != nullptr ? status : &local_status;
  *result = EM_OK;
  if (blob == nullptr || blob_bytes == 0) {
    *result = EM_ERR_ARGUMENT;
    return nullptr;
  }
  if (device_rate < EM_MIN_DEVICE_RATE || device_rate > EM_MAX_DEVICE_RATE) {
    *result = EM_ERR_RATE;
    return nullptr;
  }
  earmark::ResamplerDesign in_design;
  earmark::ResamplerDesign out_design;
  if (!earmark::design_resampler(device_rate, kModelRate, &in_design) ||
      !earmark::design_resampler(kModelRate, device_rate, &out_design)) {
    *result = EM_ERR_RATE;
    return nullptr;
  }

  em_engine* e = new (std::nothrow) em_engine();
  if (e == nullptr) {
    *result = EM_ERR_NO_MEMORY;
    return nullptr;
  }
  auto fail = [&](em_status code) -> em_engine* {
    delete e;
    *result = code;
    return nullptr;
  };

  e->device_rate = device_rate;
  e->in_design = in_design;
  e->out_design = out_design;
  e->prime = (static_cast<std::size_t>(kHop - 1) * static_cast<std::size_t>(device_rate) + kModelRate - 1) /
             static_cast<std::size_t>(kModelRate);
  const std::size_t chunk_model = earmark::Resampler::max_output(in_design, kChunk);
  const std::size_t hop_device = earmark::Resampler::max_output(out_design, kHop);
  const std::size_t model_in_capacity = static_cast<std::size_t>(kHop) + chunk_model;
  const std::size_t device_out_capacity = e->prime + kChunk + 2 * hop_device + 64;

  const std::size_t arena_bytes =
      Arena::aligned(blob_bytes) + earmark::Stft::arena_bytes() + earmark::Resampler::arena_bytes(in_design) +
      earmark::Resampler::arena_bytes(out_design) + earmark::RingBuffer::arena_bytes(model_in_capacity) +
      earmark::RingBuffer::arena_bytes(device_out_capacity) + floats_bytes(kEmbeddingDim) +
      floats_bytes(chunk_model) + 2 * floats_bytes(kHop) + floats_bytes(hop_device) +
      2 * floats_bytes(earmark::kSpecFloats) + floats_bytes(EARMARK_ERB_BANDS) + floats_bytes(2 * EARMARK_DF_BINS) +
      floats_bytes(EARMARK_ERB_BANDS);
  if (!e->arena.reserve(arena_bytes)) return fail(EM_ERR_NO_MEMORY);

  // Weights: copy into the arena (64-byte aligned), then validate the copy.
  auto* copy = e->arena.allocate_array<uint8_t>(blob_bytes);
  if (copy == nullptr) return fail(EM_ERR_NO_MEMORY);
  std::memcpy(copy, blob, blob_bytes);
  const earmark::BlobStatus blob_status = e->blob.parse(copy, blob_bytes, true);
  if (blob_status != earmark::BlobStatus::kOk) return fail(EM_ERR_BLOB);
  if (std::strcmp(e->blob.contract_hash(), EARMARK_CONTRACT_HASH) != 0) return fail(EM_ERR_CONTRACT);
  if (manifest != nullptr) {
    const std::size_t length = manifest_bytes != 0 ? manifest_bytes : std::strlen(manifest);
    earmark::ManifestFields fields;
    if (earmark::scan_manifest(manifest, length, &fields) != earmark::ManifestStatus::kOk) {
      return fail(EM_ERR_MANIFEST);
    }
    const earmark::ManifestStatus check = earmark::check_manifest(fields, e->blob);
    if (check == earmark::ManifestStatus::kContractMismatch) return fail(EM_ERR_CONTRACT);
    if (check != earmark::ManifestStatus::kOk) return fail(EM_ERR_MANIFEST);
  }
  if (!find_f32(e->blob, kErbNormInit, EARMARK_ERB_BANDS, &e->erb_norm_init) ||
      !find_f32(e->blob, kSpecNormInit, EARMARK_DF_BINS, &e->spec_norm_init) ||
      !find_f32(e->blob, kNullEmbedding, kEmbeddingDim, &e->null_embedding)) {
    return fail(EM_ERR_MISSING_TENSOR);
  }
  switch (e->net.bind(e->blob)) {
    case earmark::NetworkBind::kAbsent:
      break;
    case earmark::NetworkBind::kInvalid:
      return fail(EM_ERR_MISSING_TENSOR);
    case earmark::NetworkBind::kOk:
      if (!e->net_arena.reserve(e->net.arena_bytes() + floats_bytes(earmark::kDfOutputs)) ||
          !e->net.init(e->net_arena)) {
        return fail(EM_ERR_NO_MEMORY);
      }
      e->df_taps = e->net_arena.allocate_array<float>(earmark::kDfOutputs);
      if (e->df_taps == nullptr) return fail(EM_ERR_NO_MEMORY);
      e->has_network = true;
      break;
  }
  e->net_arena.freeze();

  if (!e->stft.init(e->arena)) return fail(EM_ERR_FFT);
  if (!e->to_model.init(in_design, e->arena) || !e->from_model.init(out_design, e->arena) ||
      !e->model_in.init(e->arena, model_in_capacity) || !e->device_out.init(e->arena, device_out_capacity)) {
    return fail(EM_ERR_NO_MEMORY);
  }
  e->embedding = e->arena.allocate_array<float>(kEmbeddingDim);
  e->chunk_model = e->arena.allocate_array<float>(chunk_model);
  e->hop_in = e->arena.allocate_array<float>(kHop);
  e->hop_out = e->arena.allocate_array<float>(kHop);
  e->hop_device = e->arena.allocate_array<float>(hop_device);
  e->spec = e->arena.allocate_array<float>(earmark::kSpecFloats);
  e->spec_out = e->arena.allocate_array<float>(earmark::kSpecFloats);
  e->erb_feat = e->arena.allocate_array<float>(EARMARK_ERB_BANDS);
  e->spec_feat = e->arena.allocate_array<float>(2 * EARMARK_DF_BINS);
  e->gains = e->arena.allocate_array<float>(EARMARK_ERB_BANDS);
  if (e->embedding == nullptr || e->chunk_model == nullptr || e->hop_in == nullptr || e->hop_out == nullptr ||
      e->hop_device == nullptr || e->spec == nullptr || e->spec_out == nullptr || e->erb_feat == nullptr ||
      e->spec_feat == nullptr || e->gains == nullptr) {
    return fail(EM_ERR_NO_MEMORY);
  }
  for (int b = 0; b < EARMARK_ERB_BANDS; ++b) e->gains[b] = 1.0f;
  e->arena.freeze();

  const std::size_t history_floats =
      (in_design.identity() ? 0 : 2 * static_cast<std::size_t>(in_design.taps_per_phase)) +
      (out_design.identity() ? 0 : 2 * static_cast<std::size_t>(out_design.taps_per_phase));
  e->state_bytes = sizeof(earmark::StreamState) + (history_floats + model_in_capacity + device_out_capacity +
                                                   kEmbeddingDim + e->net.body_state_floats()) *
                                                      sizeof(float);

  em_set_embedding(e, nullptr);
  em_reset(e);
  return e;
}

void em_destroy(em_engine* engine) { delete engine; }

em_status em_set_embedding(em_engine* engine, const float* embedding) {
  if (engine == nullptr) return EM_ERR_ARGUMENT;
  if (embedding != nullptr && !all_finite(embedding, kEmbeddingDim)) return EM_ERR_ARGUMENT;
  std::memcpy(engine->embedding, embedding != nullptr ? embedding : engine->null_embedding,
              kEmbeddingDim * sizeof(float));
  engine->personal = embedding != nullptr;
  if (engine->has_network) engine->net.condition(engine->embedding);
  return EM_OK;
}

em_status em_reset(em_engine* engine) {
  if (engine == nullptr) return EM_ERR_ARGUMENT;
  earmark::reset_state(engine->state, engine->erb_norm_init, engine->spec_norm_init);
  engine->to_model.reset();
  engine->from_model.reset();
  engine->model_in.clear();
  engine->device_out.clear();
  engine->device_out.write_zeros(engine->prime);
  if (engine->has_network) engine->net.reset();
  engine->vad = 0.0f;
  return EM_OK;
}

em_status em_process_hop_16k(em_engine* engine, const float* in, float* out, float* vad_out) {
  if (engine == nullptr || in == nullptr || out == nullptr) return EM_ERR_ARGUMENT;
  run_hop(*engine, in, out);
  if (vad_out != nullptr) *vad_out = engine->vad;
  return EM_OK;
}

em_status em_process(em_engine* engine, const float* in, size_t n, float* out, float* vad_out) {
  if (engine == nullptr || (n > 0 && (in == nullptr || out == nullptr))) return EM_ERR_ARGUMENT;
  em_engine& e = *engine;
  std::size_t done = 0;
  while (done < n) {
    const std::size_t block = n - done < kChunk ? n - done : kChunk;
    // Consume this block's input before writing its output, so in == out is safe.
    const std::size_t produced = e.to_model.process(in + done, block, e.chunk_model);
    if (e.model_in.write(e.chunk_model, produced) != produced) ++e.xruns;
    while (e.model_in.size() >= static_cast<std::size_t>(kHop)) {
      e.model_in.read(e.hop_in, kHop);
      run_hop(e, e.hop_in, e.hop_out);
      const std::size_t back = e.from_model.process(e.hop_out, kHop, e.hop_device);
      if (e.device_out.write(e.hop_device, back) != back) ++e.xruns;
    }
    const std::size_t got = e.device_out.read(out + done, block);
    if (got < block) {
      std::memset(out + done + got, 0, (block - got) * sizeof(float));
      ++e.xruns;
    }
    done += block;
  }
  if (vad_out != nullptr) *vad_out = e.vad;
  return EM_OK;
}

int32_t em_device_rate(const em_engine* engine) { return engine != nullptr ? engine->device_rate : 0; }

int32_t em_has_network(const em_engine* engine) { return engine != nullptr && engine->has_network ? 1 : 0; }

double em_latency_seconds(const em_engine* engine) {
  if (engine == nullptr) return 0.0;
  return static_cast<double>(engine->prime) / static_cast<double>(engine->device_rate) +
         static_cast<double>(kHop) / kModelRate + engine->in_design.delay_seconds() +
         engine->out_design.delay_seconds();
}

int32_t em_latency_samples(const em_engine* engine) {
  if (engine == nullptr) return 0;
  return static_cast<int32_t>(std::lround(em_latency_seconds(engine) * static_cast<double>(engine->device_rate)));
}

size_t em_arena_bytes(const em_engine* engine) {
  return engine != nullptr ? engine->arena.capacity() + engine->net_arena.capacity() : 0;
}

size_t em_state_bytes(const em_engine* engine) { return engine != nullptr ? engine->state_bytes : 0; }

uint64_t em_xruns(const em_engine* engine) { return engine != nullptr ? engine->xruns : 0; }

}  // extern "C"
