#include "stft.h"

#include <cmath>
#include <cstring>

namespace earmark {

namespace {
constexpr int kHop = EARMARK_HOP_LENGTH;
constexpr int kWindow = EARMARK_WINDOW_LENGTH;
constexpr int kFft = EARMARK_N_FFT;
static_assert(kWindow == 2 * kHop, "WOLA assumes 50 % overlap");
static_assert(kFft == kWindow, "the engine assumes n_fft == window length (no zero padding)");
constexpr double kPi = 3.14159265358979323846;
}  // namespace

void sqrt_hann_window(float* out) {
  for (int n = 0; n < kWindow; ++n) {
    out[n] = static_cast<float>(std::sin(kPi * static_cast<double>(n) / static_cast<double>(kWindow)));
  }
}

std::size_t Stft::arena_bytes() {
  return RealFft::arena_bytes(kFft) + Arena::aligned(kWindow * sizeof(float)) +
         Arena::aligned(kFft * sizeof(float));
}

bool Stft::init(Arena& arena) {
  if (!fft_.init(kFft, arena)) return false;
  window_ = arena.allocate_array<float>(kWindow);
  frame_ = arena.allocate_array<float>(kFft);
  if (window_ == nullptr || frame_ == nullptr) return false;
  sqrt_hann_window(window_);
  return true;
}

void Stft::analyze(const float* hop, float* in_buf, float* spec_ri) {
  for (int n = 0; n < kHop; ++n) frame_[n] = in_buf[n] * window_[n];
  for (int n = 0; n < kHop; ++n) frame_[kHop + n] = hop[n] * window_[kHop + n];
  if (hop != in_buf) std::memmove(in_buf, hop, kHop * sizeof(float));
  fft_.forward(frame_, spec_ri);
}

void Stft::synthesize(const float* spec_ri, float* ola_buf, float* out) {
  fft_.inverse(spec_ri, frame_);
  for (int n = 0; n < kHop; ++n) out[n] = frame_[n] * window_[n] + ola_buf[n];
  for (int n = 0; n < kHop; ++n) ola_buf[n] = frame_[kHop + n] * window_[kHop + n];
}

}  // namespace earmark
