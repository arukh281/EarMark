// The C API over the week-1 signal path, plus the arena.
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "arena.h"
#include "catch_amalgamated.hpp"
#include "earmark.h"
#include "test_support.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kHop = EARMARK_HOP_LENGTH;

struct Weights {
  std::vector<uint8_t> blob = earmark_test::read_file(earmark_test::golden_path("weights_small.emwb"));
  std::vector<uint8_t> manifest = earmark_test::read_file(earmark_test::golden_path("weights_small.json"));
};

struct Engine {
  em_engine* handle = nullptr;
  Engine(const Weights& w, int32_t rate) {
    em_status status = 1;
    handle = em_create(w.blob.data(), w.blob.size(), reinterpret_cast<const char*>(w.manifest.data()),
                       w.manifest.size(), rate, &status);
    INFO("em_create: " << em_status_string(status));
    REQUIRE(status == EM_OK);
    REQUIRE(handle != nullptr);
  }
  ~Engine() { em_destroy(handle); }
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
};

em_status create_status(const std::vector<uint8_t>& blob, const char* manifest, std::size_t manifest_bytes,
                        int32_t rate) {
  em_status status = 1;
  em_engine* e = em_create(blob.empty() ? nullptr : blob.data(), blob.size(), manifest, manifest_bytes, rate, &status);
  if (e != nullptr) REQUIRE(status == EM_OK);
  if (status != EM_OK) REQUIRE(e == nullptr);
  em_destroy(e);
  return status;
}

// Processes x in blocks cycling through `blocks`; returns the output.
std::vector<float> process(em_engine* e, const std::vector<float>& x, const std::vector<std::size_t>& blocks) {
  std::vector<float> y(x.size());
  std::size_t pos = 0;
  std::size_t index = 0;
  while (pos < x.size()) {
    const std::size_t n = std::min(blocks[index++ % blocks.size()], x.size() - pos);
    float vad = -1.0f;
    REQUIRE(em_process(e, x.data() + pos, n, y.data() + pos, &vad) == EM_OK);
    REQUIRE(vad == 0.0f);  // network not wired yet
    pos += n;
  }
  return y;
}

std::vector<float> noise(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.25f);
  std::vector<float> x(n);
  for (float& v : x) v = dist(rng);
  return x;
}

}  // namespace

TEST_CASE("arena hands out aligned zeroed blocks and refuses after freeze", "[arena]") {
  earmark::Arena arena;
  REQUIRE(arena.allocate(8) == nullptr);  // no block yet
  REQUIRE(arena.reserve(256));
  REQUIRE_FALSE(arena.reserve(256));
  auto* a = static_cast<unsigned char*>(arena.allocate(3));
  auto* b = arena.allocate_array<float>(10);
  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);
  REQUIRE(reinterpret_cast<std::uintptr_t>(a) % 64 == 0);
  REQUIRE(reinterpret_cast<std::uintptr_t>(b) % 64 == 0);
  REQUIRE(a[0] == 0);
  REQUIRE(b[9] == 0.0f);
  REQUIRE(arena.used() == 128);
  REQUIRE(arena.allocate(200) == nullptr);  // exhausted
  REQUIRE(arena.allocate_array<double>(std::numeric_limits<std::size_t>::max() / 4) == nullptr);  // overflow
  arena.freeze();
  REQUIRE(arena.allocate(1) == nullptr);

  alignas(64) static unsigned char buffer[192];
  earmark::Arena external;
  REQUIRE_FALSE(external.attach(buffer + 1, 64));  // misaligned
  REQUIRE(external.attach(buffer, sizeof(buffer)));
  REQUIRE(external.allocate(64) == buffer);
}

TEST_CASE("library identity", "[capi]") {
  REQUIRE(em_abi_version() == EM_ABI_VERSION);
  REQUIRE(std::string(em_contract_hash()) == EARMARK_CONTRACT_HASH);
  REQUIRE(std::string(em_build_info()).find(EARMARK_CONTRACT_HASH) != std::string::npos);
  REQUIRE(std::string(em_status_string(EM_ERR_CONTRACT)).find("contract") != std::string::npos);
  REQUIRE(std::string(em_status_string(12345)) == "unknown status");
}

TEST_CASE("em_create validates its inputs", "[capi]") {
  const Weights w;
  REQUIRE(!w.blob.empty());
  const char* manifest = reinterpret_cast<const char*>(w.manifest.data());
  const std::string manifest_text(w.manifest.begin(), w.manifest.end());

  REQUIRE(create_status(w.blob, manifest, w.manifest.size(), 48000) == EM_OK);
  REQUIRE(create_status(w.blob, nullptr, 0, 48000) == EM_OK);                    // header check only
  REQUIRE(create_status(w.blob, manifest_text.c_str(), 0, 16000) == EM_OK);      // NUL-terminated manifest
  REQUIRE(create_status({}, nullptr, 0, 48000) == EM_ERR_ARGUMENT);
  REQUIRE(create_status(w.blob, nullptr, 0, 7999) == EM_ERR_RATE);
  REQUIRE(create_status(w.blob, nullptr, 0, 192001) == EM_ERR_RATE);
  REQUIRE(create_status(w.blob, nullptr, 0, 44099) == EM_ERR_RATE);              // needs up = 16000

  std::vector<uint8_t> corrupt = w.blob;
  corrupt[corrupt.size() - 1] ^= 0x40;
  REQUIRE(create_status(corrupt, nullptr, 0, 48000) == EM_ERR_BLOB);

  std::string tampered = manifest_text;
  tampered.replace(tampered.find(EARMARK_CONTRACT_HASH), 16, "0123456789abcdef");
  REQUIRE(create_status(w.blob, tampered.c_str(), 0, 48000) == EM_ERR_CONTRACT);
  REQUIRE(create_status(w.blob, "{\"format\": 3}", 0, 48000) == EM_ERR_MANIFEST);
  REQUIRE(create_status(w.blob, "not json", 0, 48000) == EM_ERR_MANIFEST);

  // A valid blob without the model's const tensors (the matvec golden).
  const std::vector<uint8_t> other = earmark_test::read_file(earmark_test::golden_path("matvec.emwb"));
  REQUIRE(create_status(other, nullptr, 0, 16000) == EM_ERR_MISSING_TENSOR);

  em_status status = EM_OK;
  REQUIRE(em_create(nullptr, 10, nullptr, 0, 16000, &status) == nullptr);
  REQUIRE(status == EM_ERR_ARGUMENT);
  em_engine* no_status = em_create(w.blob.data(), w.blob.size(), nullptr, 0, 16000, nullptr);  // status is optional
  REQUIRE(no_status != nullptr);
  em_destroy(no_status);
  em_destroy(nullptr);
}

TEST_CASE("em_create does not keep the caller's blob", "[capi]") {
  Weights w;
  Engine engine(w, 16000);
  std::fill(w.blob.begin(), w.blob.end(), uint8_t{0xAB});  // the engine copied it
  const std::vector<float> x = noise(4 * kHop, 3);
  std::vector<float> y(x.size());
  REQUIRE(em_process(engine.handle, x.data(), x.size(), y.data(), nullptr) == EM_OK);
  for (float v : y) REQUIRE(std::isfinite(v));
}

TEST_CASE("at 16 kHz em_process returns the input delayed by em_latency_samples", "[capi]") {
  const Weights w;
  Engine engine(w, 16000);
  const int32_t latency = em_latency_samples(engine.handle);
  REQUIRE(latency == (kHop - 1) + kHop);  // FIFO priming + one WOLA hop
  const std::vector<float> x = noise(40 * kHop + 77, 11);
  const std::vector<float> y = process(engine.handle, x, {128, 1, 441, 480, 7, 2048});
  const auto stats = earmark_test::compare(y.data() + latency, x.data(), x.size() - static_cast<std::size_t>(latency));
  earmark_test::report("em_process 16k identity (delayed)", stats, 1e-6);
  REQUIRE(stats.max_abs < 1e-6);
  // Before the latency there is only FIFO priming (exact zeros) and the first WOLA hop,
  // whose zero half-frame comes back as FFT round-off.
  for (int32_t i = 0; i < latency; ++i) REQUIRE(std::fabs(y[static_cast<std::size_t>(i)]) < 1e-6f);
  REQUIRE(em_xruns(engine.handle) == 0);
  REQUIRE(em_arena_bytes(engine.handle) > w.blob.size());
  REQUIRE(em_state_bytes(engine.handle) > sizeof(float) * (2 * kHop + EARMARK_ERB_BANDS + EARMARK_DF_BINS));
  std::printf("  16 kHz engine: latency %d samples, arena %zu B, state %zu B\n", latency, em_arena_bytes(engine.handle),
              em_state_bytes(engine.handle));
}

TEST_CASE("em_process_hop_16k delays by exactly one hop", "[capi]") {
  const Weights w;
  Engine engine(w, 48000);
  const std::vector<float> x = noise(20 * kHop, 5);
  std::vector<float> y(x.size());
  float vad = -1.0f;
  for (std::size_t t = 0; t < x.size() / kHop; ++t) {
    REQUIRE(em_process_hop_16k(engine.handle, x.data() + t * kHop, y.data() + t * kHop, &vad) == EM_OK);
  }
  REQUIRE(vad == 0.0f);
  REQUIRE(earmark_test::compare(y.data() + kHop, x.data(), x.size() - kHop).max_abs < 1e-6);
  REQUIRE(em_process_hop_16k(engine.handle, nullptr, y.data(), nullptr) == EM_ERR_ARGUMENT);
}

TEST_CASE("resampled em_process reproduces a tone after the reported latency", "[capi]") {
  const int32_t rate = GENERATE(48000, 44100, 22050, 96000, 8000);
  INFO("device rate " << rate);
  const Weights w;
  Engine engine(w, rate);
  const double f = rate >= 16000 ? 997.0 : 613.0;
  const std::size_t n = static_cast<std::size_t>(rate) / 2;
  std::vector<float> x(n);
  for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<float>(0.5 * std::sin(2.0 * kPi * f * static_cast<double>(i) / rate));
  const std::vector<float> y = process(engine.handle, x, {128, 480, 1, 441, 1024});
  // The pipeline is linear phase end to end, so y(t) = x(t - D) with the exact
  // (fractional) delay D = em_latency_seconds(); em_latency_samples() is D rounded.
  const int32_t latency = em_latency_samples(engine.handle);
  const double delay = em_latency_seconds(engine.handle) * rate;
  double sig = 0.0;
  double err = 0.0;
  for (std::size_t i = n / 4; i < n; ++i) {
    const double want = 0.5 * std::sin(2.0 * kPi * f * (static_cast<double>(i) - delay) / rate);
    sig += want * want;
    err += (y[i] - want) * (y[i] - want);
  }
  const double snr = 10.0 * std::log10(sig / err);
  std::printf("  %6d Hz: latency %d samples (exact %.3f), tone SNR %.1f dB\n", rate, latency, delay, snr);
  REQUIRE(snr > 90.0);
  REQUIRE(std::fabs(delay - latency) <= 0.5);
  REQUIRE(em_xruns(engine.handle) == 0);
}

TEST_CASE("em_reset restores the initial state and in-place processing is safe", "[capi]") {
  const Weights w;
  Engine engine(w, 44100);
  const std::vector<float> x = noise(9000, 21);
  const std::vector<float> first = process(engine.handle, x, {441});
  REQUIRE(em_reset(engine.handle) == EM_OK);
  std::vector<float> in_place = x;
  std::size_t pos = 0;
  while (pos < in_place.size()) {
    const std::size_t n = std::min<std::size_t>(300, in_place.size() - pos);
    REQUIRE(em_process(engine.handle, in_place.data() + pos, n, in_place.data() + pos, nullptr) == EM_OK);
    pos += n;
  }
  REQUIRE(in_place == first);
  REQUIRE(em_process(engine.handle, x.data(), 0, nullptr, nullptr) == EM_OK);
  REQUIRE(em_process(engine.handle, nullptr, 5, in_place.data(), nullptr) == EM_ERR_ARGUMENT);
  REQUIRE(em_process(nullptr, x.data(), 5, in_place.data(), nullptr) == EM_ERR_ARGUMENT);
  REQUIRE(em_reset(nullptr) == EM_ERR_ARGUMENT);
}

TEST_CASE("em_set_embedding accepts finite embeddings and NULL", "[capi]") {
  const Weights w;
  Engine engine(w, 16000);
  std::vector<float> emb(EARMARK_EMBEDDING_DIM, 0.0625f);
  REQUIRE(em_set_embedding(engine.handle, emb.data()) == EM_OK);
  REQUIRE(em_set_embedding(engine.handle, nullptr) == EM_OK);
  emb[17] = std::numeric_limits<float>::quiet_NaN();
  REQUIRE(em_set_embedding(engine.handle, emb.data()) == EM_ERR_ARGUMENT);
  REQUIRE(em_set_embedding(nullptr, nullptr) == EM_ERR_ARGUMENT);
  REQUIRE(em_device_rate(engine.handle) == 16000);
  REQUIRE(em_device_rate(nullptr) == 0);
  REQUIRE(em_latency_samples(nullptr) == 0);
}
