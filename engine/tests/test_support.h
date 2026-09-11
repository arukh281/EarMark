// Shared helpers for the engine tests: golden loading and tolerance checks.
//
// Goldens are .emwb blobs written by `python -m earmark.export.golden` into
// engine/tests/goldens (EARMARK_GOLDEN_DIR). They are parsed with the engine's own
// Blob, so every golden test also exercises the loader.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "earmark_constants.h"
#include "weights.h"

namespace earmark_test {

/// Per-layer golden tolerance: max |got - want| <= kGoldenTol * max(1, peak |want|).
inline constexpr double kGoldenTol = 1e-5;

/// Whole file as bytes (empty on failure).
std::vector<uint8_t> read_file(const std::string& path);

/// EARMARK_GOLDEN_DIR / file.
std::string golden_path(const std::string& file);

/// A golden blob held in memory.
class Golden {
 public:
  explicit Golden(const std::string& name);

  /// Float tensor; REQUIREs that it exists and, if `shape` is non-empty, its shape.
  const float* f32(const std::string& tensor, std::initializer_list<uint32_t> shape = {}) const;
  /// Int tensor; REQUIREs that it exists.
  const int32_t* i32(const std::string& tensor) const;
  /// Element count of a tensor (REQUIREs that it exists).
  std::size_t numel(const std::string& tensor) const;
  earmark::TensorView view(const std::string& tensor) const;
  const std::vector<uint8_t>& bytes() const { return bytes_; }
  const earmark::Blob& blob() const { return blob_; }

 private:
  std::string name_;
  std::vector<uint8_t> bytes_;
  earmark::Blob blob_;
};

struct ErrorStats {
  double max_abs = 0.0;
  double peak = 0.0;
  std::size_t worst = 0;
};

/// Max absolute error and peak |want| over n values.
ErrorStats compare(const float* got, const float* want, std::size_t n);

/// REQUIREs max_abs <= rel_tol * max(1, peak), and prints one line with the result.
ErrorStats require_close(const std::string& what, const float* got, const float* want, std::size_t n,
                         double rel_tol = kGoldenTol);

/// Prints an accumulated result line (for checks made frame by frame).
void report(const std::string& what, const ErrorStats& stats, double rel_tol = kGoldenTol);

/// Folds `next` into `total` (max error, max peak).
void accumulate(ErrorStats& total, const ErrorStats& next);

}  // namespace earmark_test
