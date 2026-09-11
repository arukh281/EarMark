#include "test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>

#include "catch_amalgamated.hpp"

#ifndef EARMARK_GOLDEN_DIR
#error "EARMARK_GOLDEN_DIR must point at engine/tests/goldens"
#endif

namespace earmark_test {

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::string golden_path(const std::string& file) { return std::string(EARMARK_GOLDEN_DIR) + "/" + file; }

Golden::Golden(const std::string& name) : name_(name), bytes_(read_file(golden_path(name + ".emwb"))) {
  INFO("golden " << golden_path(name + ".emwb") << " (regenerate: python -m earmark.export.golden)");
  REQUIRE(!bytes_.empty());
  const earmark::BlobStatus status = blob_.parse(bytes_.data(), bytes_.size());
  INFO("parse: " << earmark::blob_status_string(status));
  REQUIRE(status == earmark::BlobStatus::kOk);
  REQUIRE(std::string(blob_.contract_hash()) == EARMARK_CONTRACT_HASH);
}

earmark::TensorView Golden::view(const std::string& tensor) const {
  earmark::TensorView v;
  INFO("tensor " << name_ << ":" << tensor);
  REQUIRE(blob_.find(tensor.c_str(), &v));
  return v;
}

const float* Golden::f32(const std::string& tensor, std::initializer_list<uint32_t> shape) const {
  const earmark::TensorView v = view(tensor);
  INFO("tensor " << name_ << ":" << tensor);
  REQUIRE(v.f32() != nullptr);
  if (shape.size() != 0) REQUIRE(v.has_shape(shape));
  return v.f32();
}

const int32_t* Golden::i32(const std::string& tensor) const {
  const earmark::TensorView v = view(tensor);
  INFO("tensor " << name_ << ":" << tensor);
  REQUIRE(v.i32() != nullptr);
  return v.i32();
}

std::size_t Golden::numel(const std::string& tensor) const { return static_cast<std::size_t>(view(tensor).numel); }

ErrorStats compare(const float* got, const float* want, std::size_t n) {
  ErrorStats stats;
  for (std::size_t i = 0; i < n; ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    if (err > stats.max_abs || std::isnan(err)) {
      stats.max_abs = std::isnan(err) ? INFINITY : err;
      stats.worst = i;
    }
    stats.peak = std::max(stats.peak, std::fabs(static_cast<double>(want[i])));
  }
  return stats;
}

void report(const std::string& what, const ErrorStats& stats, double rel_tol) {
  const double tol = rel_tol * std::max(1.0, stats.peak);
  std::printf("  %-34s max err %.3e  (peak %.3g, tol %.3e)\n", what.c_str(), stats.max_abs, stats.peak, tol);
}

void accumulate(ErrorStats& total, const ErrorStats& next) {
  if (next.max_abs > total.max_abs) {
    total.max_abs = next.max_abs;
    total.worst = next.worst;
  }
  total.peak = std::max(total.peak, next.peak);
}

ErrorStats require_close(const std::string& what, const float* got, const float* want, std::size_t n,
                         double rel_tol) {
  const ErrorStats stats = compare(got, want, n);
  const double tol = rel_tol * std::max(1.0, stats.peak);
  INFO(what << ": worst index " << stats.worst << " got " << (n ? got[stats.worst] : 0.0f) << " want "
            << (n ? want[stats.worst] : 0.0f));
  CHECK(stats.max_abs <= tol);
  return stats;
}

}  // namespace earmark_test
