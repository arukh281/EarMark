// fp32 dot product and matrix-vector kernels.
//
// Two backends share one summation order, so native and WASM builds produce
// bit-identical results (the engine is compiled with -ffp-contract=off, so no FMA):
//
// * WASM SIMD128 (compiled only when __wasm_simd128__ is defined): two f32x4
//   accumulators hold eight lane sums.
// * Scalar (every other target): the same eight lane sums in a plain array, written so
//   the compiler's SLP vectoriser turns them into NEON / SSE without reassociating.
//
// Lane l accumulates a[i + l] * b[i + l] for i = 0, 8, 16, ... in order. The lanes are
// then reduced as ((l0+l4) + (l1+l5)) + ((l2+l6) + (l3+l7)), and the n % 8 tail is
// summed serially and added last.
//
// Only this header is included by the WASM build of the kernel, so it depends on
// nothing but <stdint.h>.
#pragma once

#include <stdint.h>

namespace earmark {

/// sum_i a[i] * b[i] for i < n.
float dot_f32(const float* a, const float* b, int32_t n);

/// y = W x (+ bias): W is row-major [rows, cols]; `bias` may be nullptr.
/// `y` must not alias `x`.
void matvec_f32(const float* w, int32_t rows, int32_t cols, const float* x, const float* bias, float* y);

/// GroupedLinear: W is [groups, out/groups, in/groups]; input g * in/groups + i feeds
/// group g and output g * out/groups + o comes from it. `bias` [out] may be nullptr.
void grouped_matvec_f32(const float* w, int32_t in, int32_t out, int32_t groups, const float* x,
                        const float* bias, float* y);

/// "wasm-simd128" or "scalar".
const char* matvec_backend();

}  // namespace earmark
