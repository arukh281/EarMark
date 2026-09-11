// One step of a (stacked) GRU with PyTorch's nn.GRU convention.
//
// Weights keep the state_dict layout: weight_ih [3H, in], weight_hh [3H, H], bias_ih and
// bias_hh [3H], with gate rows stacked r, z, n:
//
//   r  = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
//   z  = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
//   n  = tanh(W_in x + b_in + r * (W_hn h + b_hn))
//   h' = (1 - z) * n + z * h
#pragma once

#include <cstddef>
#include <cstdint>

namespace earmark {

/// Non-owning views of one layer's weights (usually into the weight blob).
struct GruLayer {
  const float* weight_ih = nullptr;  ///< [3H, input]
  const float* weight_hh = nullptr;  ///< [3H, hidden]
  const float* bias_ih = nullptr;    ///< [3H]
  const float* bias_hh = nullptr;    ///< [3H]
  int32_t input = 0;
  int32_t hidden = 0;
};

/// Scratch floats needed by gru_step() for a layer of `hidden` units.
inline constexpr std::size_t gru_scratch_floats(int32_t hidden) { return 6 * static_cast<std::size_t>(hidden); }

/// Advances `h` [hidden] by one step with input `x` [input]. `x` must not alias `h`.
/// `scratch` holds gru_scratch_floats(hidden) floats.
void gru_step(const GruLayer& layer, const float* x, float* h, float* scratch);

/// Runs `count` stacked layers: layer 0 reads `x`, layer k reads layer k-1's new state.
/// `h` is [count, hidden] (all layers share `hidden`); the output is the last row.
void gru_stack_step(const GruLayer* layers, int32_t count, const float* x, float* h, float* scratch);

}  // namespace earmark
