// The only engine translation unit compiled with C++ exceptions enabled: pocketfft
// throws on bad arguments and on allocation failure. Both can happen only inside init(),
// which catches them. forward() and inverse() call the allocation-free path added by
// third_party/patches/pocketfft_noalloc.patch, which cannot throw.
#include "fft.h"

#include <cstring>

#include "pocketfft_hdronly.hpp"

namespace earmark {

namespace {
using Plan = pocketfft::detail::pocketfft_r<float>;
}  // namespace

RealFft::~RealFft() { delete static_cast<Plan*>(plan_); }

bool RealFft::init(int n, Arena& arena) {
  if (plan_ != nullptr || n <= 0 || n % 2 != 0) return false;
  Plan* plan = nullptr;
  try {
    plan = new Plan(static_cast<std::size_t>(n));
  } catch (...) {
    return false;
  }
  if (!plan->supports_noalloc()) {
    delete plan;
    return false;
  }
  work_ = arena.allocate_array<float>(static_cast<std::size_t>(n));
  scratch_ = arena.allocate_array<float>(static_cast<std::size_t>(n));
  if (work_ == nullptr || scratch_ == nullptr) {
    delete plan;
    return false;
  }
  plan_ = plan;
  n_ = n;
  return true;
}

void RealFft::forward(const float* in, float* out_ri) {
  const Plan& plan = *static_cast<const Plan*>(plan_);
  std::memcpy(work_, in, static_cast<std::size_t>(n_) * sizeof(float));
  plan.exec_noalloc(work_, 1.0f, true, scratch_);
  // pocketfft's packed layout: r0, r1, i1, r2, i2, ..., r(n/2).
  const int half = n_ / 2;
  out_ri[0] = work_[0];
  out_ri[1] = 0.0f;
  for (int k = 1; k < half; ++k) {
    out_ri[2 * k] = work_[2 * k - 1];
    out_ri[2 * k + 1] = work_[2 * k];
  }
  out_ri[2 * half] = work_[n_ - 1];
  out_ri[2 * half + 1] = 0.0f;
}

void RealFft::inverse(const float* in_ri, float* out) {
  const Plan& plan = *static_cast<const Plan*>(plan_);
  const int half = n_ / 2;
  work_[0] = in_ri[0];
  for (int k = 1; k < half; ++k) {
    work_[2 * k - 1] = in_ri[2 * k];
    work_[2 * k] = in_ri[2 * k + 1];
  }
  work_[n_ - 1] = in_ri[2 * half];
  plan.exec_noalloc(work_, 1.0f / static_cast<float>(n_), false, scratch_);
  std::memcpy(out, work_, static_cast<std::size_t>(n_) * sizeof(float));
}

}  // namespace earmark
