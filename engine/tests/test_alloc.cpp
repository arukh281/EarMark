// Proves that nothing allocates after em_create. Built as its own executable so the
// allocation hooks in alloc_counter.cpp never affect the other tests.
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "alloc_counter.h"
#include "arena.h"
#include "catch_amalgamated.hpp"
#include "earmark.h"
#include "gru.h"
#include "matvec_f32.h"
#include "test_support.h"

namespace {
void* volatile g_sink = nullptr;
constexpr double kPi = 3.14159265358979323846;
}  // namespace

TEST_CASE("the allocation counter sees malloc and operator new", "[alloc]") {
  INFO("backend " << earmark_test::alloc_counter_backend());
  std::printf("  allocation counter backend: %s\n", earmark_test::alloc_counter_backend());
  earmark_test::alloc_counter_start();
  void* block = std::malloc(48);
  g_sink = block;
  int* array = new int[9];
  g_sink = array;
  const uint64_t count = earmark_test::alloc_counter_stop();
  std::free(block);
  delete[] array;
  REQUIRE(count >= (earmark_test::alloc_counter_sees_malloc() ? 2u : 1u));

  earmark_test::alloc_counter_start();
  const uint64_t idle = earmark_test::alloc_counter_stop();
  REQUIRE(idle == 0);
}

TEST_CASE("no allocation after em_create", "[alloc]") {
  const int32_t rate = GENERATE(16000, 48000, 44100, 22050, 8000, 96000);
  INFO("device rate " << rate);
  const std::vector<uint8_t> blob = earmark_test::read_file(earmark_test::golden_path("weights_small.emwb"));
  const std::vector<uint8_t> manifest = earmark_test::read_file(earmark_test::golden_path("weights_small.json"));
  REQUIRE(!blob.empty());
  em_status status = 1;
  em_engine* engine = em_create(blob.data(), blob.size(), reinterpret_cast<const char*>(manifest.data()),
                                manifest.size(), rate, &status);
  REQUIRE(status == EM_OK);
  REQUIRE(engine != nullptr);

  std::vector<float> in(4096);
  std::vector<float> out(4096);
  for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(0.3 * std::sin(2.0 * kPi * 440.0 * static_cast<double>(i) / rate));
  std::vector<float> embedding(EARMARK_EMBEDDING_DIM, 0.05f);
  const std::size_t blocks[] = {128, 1, 441, 480, 7, 2048, 4096, 0, 160, 1023};
  float vad = 0.0f;
  em_status worst = EM_OK;

  earmark_test::alloc_counter_start();
  for (int repeat = 0; repeat < 12; ++repeat) {
    for (std::size_t n : blocks) worst |= em_process(engine, in.data(), n, out.data(), &vad);
  }
  worst |= em_set_embedding(engine, embedding.data());
  worst |= em_process(engine, in.data(), 1000, in.data(), &vad);  // in place
  worst |= em_set_embedding(engine, nullptr);
  worst |= em_reset(engine);
  for (int hop = 0; hop < 50; ++hop) worst |= em_process_hop_16k(engine, in.data(), out.data(), &vad);
  const int32_t latency = em_latency_samples(engine);
  const std::size_t state = em_state_bytes(engine);
  const uint64_t xruns = em_xruns(engine);
  const uint64_t count = earmark_test::alloc_counter_stop();

  em_destroy(engine);
  REQUIRE(worst == EM_OK);
  REQUIRE(latency > 0);
  REQUIRE(state > 0);
  REQUIRE(xruns == 0);
  INFO("allocations after em_create: " << count);
  REQUIRE(count == 0);
}

TEST_CASE("GRU and matvec kernels do not allocate", "[alloc]") {
  constexpr int32_t kIn = 37;
  constexpr int32_t kHidden = 48;
  std::vector<float> w_ih(3 * kHidden * kIn, 0.01f);
  std::vector<float> w_hh(3 * kHidden * kHidden, -0.02f);
  std::vector<float> b(3 * kHidden, 0.1f);
  std::vector<float> x(kIn, 0.5f);
  std::vector<float> h(2 * kHidden, 0.0f);
  std::vector<float> scratch(earmark::gru_scratch_floats(kHidden));
  std::vector<float> y(3 * kHidden);
  const earmark::GruLayer layers[2] = {{w_ih.data(), w_hh.data(), b.data(), b.data(), kIn, kHidden},
                                       {w_hh.data(), w_hh.data(), b.data(), b.data(), kHidden, kHidden}};
  earmark_test::alloc_counter_start();
  for (int t = 0; t < 100; ++t) earmark::gru_stack_step(layers, 2, x.data(), h.data(), scratch.data());
  earmark::matvec_f32(w_ih.data(), 3 * kHidden, kIn, x.data(), b.data(), y.data());
  earmark::grouped_matvec_f32(w_hh.data(), kHidden, kHidden, 4, h.data(), nullptr, y.data());
  const uint64_t count = earmark_test::alloc_counter_stop();
  REQUIRE(std::isfinite(h[0]));
  REQUIRE(count == 0);
}
