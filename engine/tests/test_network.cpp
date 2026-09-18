// The network against PyTorch. The goldens (earmark.export.golden: network_model,
// network_partial, network) stream 1 s through a tiny random-weight GRU EarmarkNet in
// float64, hop by hop, with per-layer traces of the first frames.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "arena.h"
#include "catch_amalgamated.hpp"
#include "earmark.h"
#include "erb.h"
#include "network.h"
#include "state.h"
#include "stft.h"
#include "test_support.h"

namespace {

using earmark_test::ErrorStats;
using earmark_test::Golden;

constexpr int kHop = EARMARK_HOP_LENGTH;
constexpr int kEmbeddingDim = EARMARK_EMBEDDING_DIM;

struct Files {
  std::vector<uint8_t> blob = earmark_test::read_file(earmark_test::golden_path("network_model.emwb"));
  std::vector<uint8_t> manifest = earmark_test::read_file(earmark_test::golden_path("network_model.json"));
};

struct Engine {
  em_engine* handle = nullptr;
  Engine(const Files& f, int32_t rate) {
    em_status status = 1;
    handle = em_create(f.blob.data(), f.blob.size(), reinterpret_cast<const char*>(f.manifest.data()),
                       f.manifest.size(), rate, &status);
    INFO("em_create: " << em_status_string(status));
    REQUIRE(status == EM_OK);
    REQUIRE(handle != nullptr);
  }
  ~Engine() { em_destroy(handle); }
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
};

void require_within(const std::string& what, const ErrorStats& stats) {
  earmark_test::report(what, stats);
  INFO(what << ": worst element " << stats.worst);
  REQUIRE(stats.max_abs <= earmark_test::kGoldenTol * std::max(1.0, stats.peak));
}

/// Streams the golden input through em_process_hop_16k; returns (wav, vad).
std::pair<std::vector<float>, std::vector<float>> stream_hops(em_engine* engine, const Golden& g, int32_t hops) {
  std::vector<float> wav(static_cast<std::size_t>(hops) * kHop);
  std::vector<float> vad(static_cast<std::size_t>(hops));
  const float* x = g.f32("x");
  for (int32_t t = 0; t < hops; ++t) {
    REQUIRE(em_process_hop_16k(engine, x + t * kHop, wav.data() + t * kHop, &vad[static_cast<std::size_t>(t)]) ==
            EM_OK);
  }
  return {wav, vad};
}

}  // namespace

TEST_CASE("the network binds from tensor shapes and refuses a partial network", "[network]") {
  const Golden model("network_model");
  earmark::Network net;
  REQUIRE(net.bind(model.blob()) == earmark::NetworkBind::kOk);
  CHECK(net.hidden() == 24);  // golden.NETWORK_CONFIG
  CHECK(net.layers() == 2);

  earmark::Network none;
  CHECK(none.bind(Golden("weights_small").blob()) == earmark::NetworkBind::kAbsent);
  earmark::Network partial;
  CHECK(partial.bind(Golden("network_partial").blob()) == earmark::NetworkBind::kInvalid);

  const std::vector<uint8_t> blob = earmark_test::read_file(earmark_test::golden_path("network_partial.emwb"));
  em_status status = 1;
  em_engine* engine = em_create(blob.data(), blob.size(), nullptr, 0, 16000, &status);
  CHECK(engine == nullptr);
  CHECK(status == EM_ERR_MISSING_TENSOR);
  em_destroy(engine);
}

TEST_CASE("each layer of the network matches PyTorch hop by hop", "[network]") {
  const Golden model("network_model");
  const Golden g("network");
  const int32_t traced = g.i32("dims")[1];
  earmark::Network net;
  REQUIRE(net.bind(model.blob()) == earmark::NetworkBind::kOk);
  earmark::Arena arena;
  REQUIRE(arena.reserve(net.arena_bytes() + earmark::Stft::arena_bytes()));
  REQUIRE(net.init(arena));
  earmark::Stft stft;
  REQUIRE(stft.init(arena));

  const int32_t hidden = net.hidden();
  net.condition(g.f32("embedding", {kEmbeddingDim}));
  require_within("FiLM pre scale", earmark_test::compare(net.pre_scale(), g.f32("cond.pre_scale"), hidden));
  require_within("FiLM pre shift", earmark_test::compare(net.pre_shift(), g.f32("cond.pre_shift"), hidden));
  require_within("FiLM post scale", earmark_test::compare(net.post_scale(), g.f32("cond.post_scale"), hidden));
  require_within("FiLM post shift", earmark_test::compare(net.post_shift(), g.f32("cond.post_shift"), hidden));

  earmark::StreamState state{};
  earmark::reset_state(state, model.f32("const.erb_norm_init"), model.f32("const.spec_norm_init"));
  net.reset();
  std::vector<float> spec(earmark::kSpecFloats);
  std::vector<float> spec_out(earmark::kSpecFloats);
  std::vector<float> erb_feat(EARMARK_ERB_BANDS);
  std::vector<float> spec_feat(2 * EARMARK_DF_BINS);
  std::vector<float> gains(EARMARK_ERB_BANDS);
  std::vector<float> taps(earmark::kDfOutputs);
  std::vector<float> wav(kHop);

  std::vector<std::pair<std::string, ErrorStats>> totals;
  auto check = [&](const std::string& name, const float* got, int32_t frame) {
    const std::string tensor = "trace." + name;
    const std::size_t width = g.numel(tensor) / static_cast<std::size_t>(traced);
    const ErrorStats stats = earmark_test::compare(got, g.f32(tensor) + frame * width, width);
    auto it = std::find_if(totals.begin(), totals.end(), [&](const auto& entry) { return entry.first == name; });
    if (it == totals.end()) {
      totals.emplace_back(name, stats);
    } else {
      earmark_test::accumulate(it->second, stats);
    }
  };
  const float* x = g.f32("x");
  for (int32_t t = 0; t < traced; ++t) {
    stft.analyze(x + t * kHop, state.in_buf, spec.data());
    earmark::erb_features_step(spec.data(), state.erb_norm, erb_feat.data());
    earmark::unit_norm_step(spec.data(), state.spec_norm, spec_feat.data());
    const float vad_logit = net.step(erb_feat.data(), spec_feat.data(), state.enc_erb_prev, state.enc_df_prev,
                                     gains.data(), taps.data());
    earmark::apply_erb_gains(gains.data(), spec.data(), spec_out.data());
    earmark::deep_filter_step(taps.data(), spec_out.data(), state.df_hist);
    stft.synthesize(spec_out.data(), state.ola_buf, wav.data());

    check("erb_feat", erb_feat.data(), t);
    check("spec_feat", spec_feat.data(), t);
    check("enc_erb", net.enc_erb(), t);
    check("enc_df", net.enc_df(), t);
    check("enc", net.enc(), t);
    check("film_pre", net.film_pre(), t);
    check("body_out", net.body_out(), t);
    check("vad_logit", &vad_logit, t);
    check("film_post", net.film_post(), t);
    check("gains", gains.data(), t);
    check("df_raw", net.df_raw(), t);
    check("spec_out", spec_out.data(), t);
    check("wav", wav.data(), t);
  }
  for (const auto& [name, stats] : totals) require_within("layer " + name, stats);
}

TEST_CASE("em_process_hop_16k reproduces PyTorch's audio and VAD in both modes", "[network]") {
  const Files files;
  const Golden g("network");
  const int32_t hops = g.i32("dims")[0];
  const std::size_t n = static_cast<std::size_t>(hops) * kHop;
  Engine engine(files, 16000);
  REQUIRE(em_has_network(engine.handle) == 1);

  REQUIRE(em_set_embedding(engine.handle, g.f32("embedding", {kEmbeddingDim})) == EM_OK);
  REQUIRE(em_reset(engine.handle) == EM_OK);
  const auto personal = stream_hops(engine.handle, g, hops);
  require_within("personal wav", earmark_test::compare(personal.first.data(), g.f32("personal.wav"), n));
  require_within("personal vad", earmark_test::compare(personal.second.data(), g.f32("personal.vad"), hops));

  REQUIRE(em_set_embedding(engine.handle, nullptr) == EM_OK);
  REQUIRE(em_reset(engine.handle) == EM_OK);
  const auto null = stream_hops(engine.handle, g, hops);
  require_within("NULL-embedding wav", earmark_test::compare(null.first.data(), g.f32("null.wav"), n));
  require_within("NULL-embedding vad", earmark_test::compare(null.second.data(), g.f32("null.vad"), hops));

  // The embedding reaches the output: the two modes differ far beyond the tolerance.
  const ErrorStats modes = earmark_test::compare(personal.first.data(), null.first.data(), n);
  std::printf("  personal vs NULL wav differ by up to %.3e\n", modes.max_abs);
  REQUIRE(modes.max_abs > 100.0 * earmark_test::kGoldenTol);

  // em_reset restores the initial state exactly.
  REQUIRE(em_reset(engine.handle) == EM_OK);
  const auto again = stream_hops(engine.handle, g, hops);
  REQUIRE(std::memcmp(again.first.data(), null.first.data(), n * sizeof(float)) == 0);
}

TEST_CASE("em_process at 16 kHz is the hop path delayed by the FIFO priming", "[network]") {
  const Files files;
  const Golden g("network");
  const int32_t hops = g.i32("dims")[0];
  const std::size_t n = static_cast<std::size_t>(hops) * kHop;
  Engine engine(files, 16000);
  REQUIRE(em_set_embedding(engine.handle, g.f32("embedding", {kEmbeddingDim})) == EM_OK);

  const std::size_t blocks[] = {1, 7, 128, 3, 441, 480, 64, 1000, 2, 250};
  std::vector<float> out(n);
  const float* x = g.f32("x");
  std::size_t done = 0;
  for (std::size_t i = 0; done < n; ++i) {
    const std::size_t block = std::min(blocks[i % std::size(blocks)], n - done);
    float vad = -1.0f;
    REQUIRE(em_process(engine.handle, x + done, block, out.data() + done, &vad) == EM_OK);
    REQUIRE(vad >= 0.0f);
    REQUIRE(vad <= 1.0f);
    done += block;
  }
  const auto prime = static_cast<std::size_t>(em_latency_samples(engine.handle) - kHop);
  for (std::size_t i = 0; i < prime; ++i) REQUIRE(out[i] == 0.0f);
  require_within("em_process 16k wav (delayed)",
                 earmark_test::compare(out.data() + prime, g.f32("personal.wav"), n - prime));
  REQUIRE(em_xruns(engine.handle) == 0);

  // The only state the network adds beyond the signal path is the GRU state.
  Files small;
  small.blob = earmark_test::read_file(earmark_test::golden_path("weights_small.emwb"));
  small.manifest = earmark_test::read_file(earmark_test::golden_path("weights_small.json"));
  Engine plain(small, 16000);
  REQUIRE(em_has_network(plain.handle) == 0);
  CHECK(em_state_bytes(engine.handle) - em_state_bytes(plain.handle) == 2 * 24 * sizeof(float));
}
