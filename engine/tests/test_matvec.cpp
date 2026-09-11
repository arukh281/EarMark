#include <cstring>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "matvec_f32.h"
#include "test_support.h"

namespace {

// The documented summation order (matvec_f32.h), written out independently. Both
// backends must equal it bit for bit.
float documented_dot(const float* a, const float* b, int32_t n) {
  float lane[8] = {};
  int32_t i = 0;
  for (; i + 8 <= n; i += 8) {
    for (int32_t l = 0; l < 8; ++l) {
      const float product = a[i + l] * b[i + l];
      lane[l] = lane[l] + product;
    }
  }
  const float head = ((lane[0] + lane[4]) + (lane[1] + lane[5])) + ((lane[2] + lane[6]) + (lane[3] + lane[7]));
  float tail = 0.0f;
  for (; i < n; ++i) {
    const float product = a[i] * b[i];
    tail = tail + product;
  }
  return head + tail;
}

}  // namespace

TEST_CASE("dense and grouped matvec match the golden", "[matvec]") {
  const earmark_test::Golden golden("matvec");
  const int32_t* shapes = golden.i32("shapes");
  const std::size_t cases = golden.numel("shapes") / 2;
  std::printf("  matvec backend: %s\n", earmark::matvec_backend());
  earmark_test::ErrorStats total;
  for (std::size_t c = 0; c < cases; ++c) {
    const int32_t rows = shapes[2 * c];
    const int32_t cols = shapes[2 * c + 1];
    const std::string p = "case" + std::to_string(c) + ".";
    INFO(rows << " x " << cols);
    const float* w = golden.f32(p + "w", {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)});
    const float* x = golden.f32(p + "x", {static_cast<uint32_t>(cols)});
    const float* b = golden.f32(p + "b", {static_cast<uint32_t>(rows)});
    std::vector<float> y(static_cast<std::size_t>(rows));
    earmark::matvec_f32(w, rows, cols, x, b, y.data());
    earmark_test::accumulate(total, earmark_test::require_close("matvec." + p + "y", y.data(), golden.f32(p + "y"), y.size()));
    // Without bias it is the plain product.
    std::vector<float> no_bias(static_cast<std::size_t>(rows));
    earmark::matvec_f32(w, rows, cols, x, nullptr, no_bias.data());
    for (int32_t r = 0; r < rows; ++r) REQUIRE(no_bias[static_cast<std::size_t>(r)] + b[r] == y[static_cast<std::size_t>(r)]);
  }
  earmark_test::report("matvec dense (all shapes)", total);

  const int32_t* dims = golden.i32("grouped.dims");
  const int32_t in = dims[0];
  const int32_t out = dims[1];
  const int32_t groups = dims[2];
  std::vector<float> y(static_cast<std::size_t>(out));
  earmark::grouped_matvec_f32(golden.f32("grouped.w", {static_cast<uint32_t>(groups), static_cast<uint32_t>(out / groups),
                                                       static_cast<uint32_t>(in / groups)}),
                              in, out, groups, golden.f32("grouped.x"), golden.f32("grouped.b"), y.data());
  earmark_test::report("matvec grouped", earmark_test::require_close("matvec.grouped.y", y.data(), golden.f32("grouped.y"), y.size()));
}

TEST_CASE("dot product follows the documented summation order bit for bit", "[matvec]") {
  std::vector<float> a(300);
  std::vector<float> b(300);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(std::sin(0.7 * static_cast<double>(i)) * 3.0);
    b[i] = static_cast<float>(std::cos(1.3 * static_cast<double>(i)) / 7.0);
  }
  for (int32_t n : {0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 131, 300}) {
    const float got = earmark::dot_f32(a.data(), b.data(), n);
    const float want = documented_dot(a.data(), b.data(), n);
    INFO("n " << n);
    REQUIRE(std::memcmp(&got, &want, sizeof(float)) == 0);
  }
}
