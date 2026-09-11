#include "state.h"

#include <cstring>

namespace earmark {

void reset_state(StreamState& state, const float* erb_norm_init, const float* spec_norm_init) {
  std::memset(&state, 0, sizeof(state));
  std::memcpy(state.erb_norm, erb_norm_init, sizeof(state.erb_norm));
  std::memcpy(state.spec_norm, spec_norm_init, sizeof(state.spec_norm));
}

}  // namespace earmark
