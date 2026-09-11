#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gru.h"
#include "test_support.h"

TEST_CASE("stacked GRU steps match the golden", "[gru]") {
  const earmark_test::Golden golden("gru");
  for (const std::string name : {"a", "b"}) {
    const std::string p = name + ".";
    const int32_t* dims = golden.i32(p + "dims");
    const int32_t input = dims[0];
    const int32_t hidden = dims[1];
    const int32_t layers = dims[2];
    const int32_t frames = dims[3];
    INFO("case " << name << ": " << input << " -> " << hidden << " x " << layers << ", " << frames << " frames");
    const auto u_in = static_cast<uint32_t>(input);
    const auto u_h = static_cast<uint32_t>(hidden);
    std::vector<earmark::GruLayer> stack(static_cast<std::size_t>(layers));
    for (int32_t l = 0; l < layers; ++l) {
      const std::string s = "_l" + std::to_string(l);
      earmark::GruLayer& layer = stack[static_cast<std::size_t>(l)];
      layer.input = l == 0 ? input : hidden;
      layer.hidden = hidden;
      layer.weight_ih = golden.f32(p + "weight_ih" + s, {3 * u_h, static_cast<uint32_t>(layer.input)});
      layer.weight_hh = golden.f32(p + "weight_hh" + s, {3 * u_h, u_h});
      layer.bias_ih = golden.f32(p + "bias_ih" + s, {3 * u_h});
      layer.bias_hh = golden.f32(p + "bias_hh" + s, {3 * u_h});
    }
    const float* x = golden.f32(p + "x", {static_cast<uint32_t>(frames), u_in});
    const float* h0 = golden.f32(p + "h0", {static_cast<uint32_t>(layers), u_h});
    const float* y = golden.f32(p + "y", {static_cast<uint32_t>(frames), u_h});
    std::vector<float> h(h0, h0 + static_cast<std::ptrdiff_t>(layers) * hidden);
    std::vector<float> scratch(earmark::gru_scratch_floats(hidden));
    earmark_test::ErrorStats total;
    for (int32_t t = 0; t < frames; ++t) {
      earmark::gru_stack_step(stack.data(), layers, x + static_cast<std::ptrdiff_t>(t) * input, h.data(), scratch.data());
      const float* top = h.data() + static_cast<std::ptrdiff_t>(layers - 1) * hidden;
      earmark_test::accumulate(total, earmark_test::require_close("gru." + p + "y frame " + std::to_string(t), top,
                                                                  y + static_cast<std::ptrdiff_t>(t) * hidden,
                                                                  static_cast<std::size_t>(hidden)));
    }
    earmark_test::report("gru." + p + "y", total);
    earmark_test::report("gru." + p + "h_final", earmark_test::require_close("gru." + p + "h_final", h.data(),
                                                                          golden.f32(p + "h_final"), h.size()));
  }
}
