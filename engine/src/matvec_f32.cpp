// See matvec_f32.h for the shared summation order. Keep both backends in step: any
// change to the order in one must be made in the other, or native/WASM parity breaks.
#include "matvec_f32.h"

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

namespace earmark {

#if defined(__wasm_simd128__)

float dot_f32(const float* a, const float* b, int32_t n) {
  v128_t low = wasm_f32x4_splat(0.0f);   // lanes 0..3
  v128_t high = wasm_f32x4_splat(0.0f);  // lanes 4..7
  int32_t i = 0;
  for (; i + 8 <= n; i += 8) {
    low = wasm_f32x4_add(low, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
    high = wasm_f32x4_add(high, wasm_f32x4_mul(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4)));
  }
  const v128_t pairs = wasm_f32x4_add(low, high);  // (l0+l4, l1+l5, l2+l6, l3+l7)
  const float head = (wasm_f32x4_extract_lane(pairs, 0) + wasm_f32x4_extract_lane(pairs, 1)) +
                     (wasm_f32x4_extract_lane(pairs, 2) + wasm_f32x4_extract_lane(pairs, 3));
  float tail = 0.0f;
  for (; i < n; ++i) tail += a[i] * b[i];
  return head + tail;
}

const char* matvec_backend() { return "wasm-simd128"; }

#else

float dot_f32(const float* a, const float* b, int32_t n) {
  float lane[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int32_t i = 0;
  for (; i + 8 <= n; i += 8) {
    for (int32_t l = 0; l < 8; ++l) lane[l] += a[i + l] * b[i + l];
  }
  const float head = ((lane[0] + lane[4]) + (lane[1] + lane[5])) + ((lane[2] + lane[6]) + (lane[3] + lane[7]));
  float tail = 0.0f;
  for (; i < n; ++i) tail += a[i] * b[i];
  return head + tail;
}

const char* matvec_backend() { return "scalar"; }

#endif

void matvec_f32(const float* w, int32_t rows, int32_t cols, const float* x, const float* bias, float* y) {
  for (int32_t r = 0; r < rows; ++r) {
    const float sum = dot_f32(w + static_cast<int64_t>(r) * cols, x, cols);
    y[r] = bias != nullptr ? sum + bias[r] : sum;
  }
}

void grouped_matvec_f32(const float* w, int32_t in, int32_t out, int32_t groups, const float* x,
                        const float* bias, float* y) {
  const int32_t in_g = in / groups;
  const int32_t out_g = out / groups;
  for (int32_t g = 0; g < groups; ++g) {
    matvec_f32(w + static_cast<int64_t>(g) * out_g * in_g, out_g, in_g, x + g * in_g,
               bias != nullptr ? bias + g * out_g : nullptr, y + g * out_g);
  }
}

}  // namespace earmark
