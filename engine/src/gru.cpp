#include "gru.h"

#include <cmath>

#include "matvec_f32.h"

namespace earmark {

namespace {
inline float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }
}  // namespace

void gru_step(const GruLayer& layer, const float* x, float* h, float* scratch) {
  const int32_t hidden = layer.hidden;
  float* gi = scratch;                // [3H] W_ih x + b_ih
  float* gh = scratch + 3 * hidden;   // [3H] W_hh h + b_hh
  matvec_f32(layer.weight_ih, 3 * hidden, layer.input, x, layer.bias_ih, gi);
  matvec_f32(layer.weight_hh, 3 * hidden, hidden, h, layer.bias_hh, gh);
  for (int32_t j = 0; j < hidden; ++j) {
    const float r = sigmoid(gi[j] + gh[j]);
    const float z = sigmoid(gi[hidden + j] + gh[hidden + j]);
    const float n = std::tanh(gi[2 * hidden + j] + r * gh[2 * hidden + j]);
    h[j] = (1.0f - z) * n + z * h[j];
  }
}

void gru_stack_step(const GruLayer* layers, int32_t count, const float* x, float* h, float* scratch) {
  const float* input = x;
  for (int32_t k = 0; k < count; ++k) {
    float* state = h + static_cast<std::ptrdiff_t>(k) * layers[k].hidden;
    gru_step(layers[k], input, state, scratch);
    input = state;
  }
}

}  // namespace earmark
