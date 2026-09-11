// Weight-blob (.emwb) parser and JSON-manifest cross-check, allocation-free.
//
// The format is specified in python/earmark/export/blob.py (the writer and the Python
// reader); Blob::parse makes the same checks as earmark.export.blob.unpack_blob. In
// short: a 64-byte header (magic "EMWBLOB\0", format version, tensor count, 16-hex-digit
// contract hash, table/data offsets, file size, CRC-32 of header+table and of the data),
// a table of 128-byte entries (NUL-padded name, dtype, ndim, shape[6], offset, numel,
// CRC-32), then 64-byte-aligned little-endian float32 / int32 data.
//
// Blob never copies: TensorView::data points into the parsed buffer, which must outlive
// the Blob. The engine copies the caller's blob into its arena first, so every tensor is
// 64-byte aligned.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace earmark {

inline constexpr uint32_t kBlobFormatVersion = 1;
inline constexpr std::size_t kBlobHeaderBytes = 64;
inline constexpr std::size_t kBlobEntryBytes = 128;
inline constexpr std::size_t kBlobNameBytes = 72;
inline constexpr uint32_t kBlobMaxNdim = 6;
inline constexpr std::size_t kBlobAlignment = 64;
inline constexpr uint32_t kBlobMaxTensors = 65535;
inline constexpr uint32_t kDtypeFloat32 = 1;
inline constexpr uint32_t kDtypeInt32 = 2;
inline constexpr const char* kManifestFormat = "earmark-weights-blob";

/// CRC-32/ISO-HDLC (zlib.crc32). Pass a previous result as `crc` to continue it.
uint32_t crc32(const void* data, std::size_t size, uint32_t crc = 0);

enum class BlobStatus : int32_t {
  kOk = 0,
  kTooSmall,          ///< shorter than the header
  kBadMagic,
  kBadVersion,
  kBadContractHash,   ///< hash field is not 16 lowercase hex digits
  kBadLayout,         ///< offsets, sizes or tensor count inconsistent
  kSizeMismatch,      ///< header file size differs from the buffer size
  kTableCrc,
  kDataCrc,
  kBadEntry,          ///< malformed tensor entry (name, dtype, shape, offset, bounds)
  kDuplicateName,
  kTensorCrc,
  kMisaligned,        ///< tensor data would not be 4-byte aligned in memory
};

const char* blob_status_string(BlobStatus status);

/// A tensor inside a parsed blob.
struct TensorView {
  const char* name = nullptr;  ///< NUL-terminated, inside the blob
  uint32_t dtype = 0;
  uint32_t ndim = 0;
  uint32_t shape[kBlobMaxNdim] = {};
  uint64_t numel = 0;
  const void* data = nullptr;

  const float* f32() const { return dtype == kDtypeFloat32 ? static_cast<const float*>(data) : nullptr; }
  const int32_t* i32() const { return dtype == kDtypeInt32 ? static_cast<const int32_t*>(data) : nullptr; }
  /// True when ndim and every dimension equal `dims`.
  bool has_shape(std::initializer_list<uint32_t> dims) const;
};

class Blob {
 public:
  /// Validates `data` [size] in place. No allocation and no copy.
  BlobStatus parse(const uint8_t* data, std::size_t size, bool verify_crc = true);

  uint32_t count() const { return count_; }
  /// Tensor `index` in file order; false if out of range or nothing is parsed.
  bool tensor(uint32_t index, TensorView* out) const;
  /// Tensor by exact name; false when absent.
  bool find(const char* name, TensorView* out) const;

  /// The header's contract hash (16 hex digits, NUL-terminated).
  const char* contract_hash() const { return hash_; }
  uint32_t table_crc32() const { return table_crc_; }
  uint32_t data_crc32() const { return data_crc_; }
  uint64_t file_bytes() const { return size_; }

 private:
  const uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
  uint32_t count_ = 0;
  uint64_t data_offset_ = 0;
  uint32_t table_crc_ = 0;
  uint32_t data_crc_ = 0;
  char hash_[17] = {};
};

/// The flat top-level manifest keys the engine cross-checks. Missing numbers are -1,
/// missing strings are empty.
struct ManifestFields {
  char format[32] = {};
  int64_t format_version = -1;
  char contract_hash[17] = {};
  int64_t blob_bytes = -1;
  int64_t blob_table_crc32 = -1;
  int64_t blob_data_crc32 = -1;
};

enum class ManifestStatus : int32_t {
  kOk = 0,
  kSyntax,            ///< not a JSON object, or a field has the wrong type
  kMissingField,
  kWrongFormat,       ///< format or format_version is not the one this engine reads
  kContractMismatch,  ///< manifest or blob hash differs from EARMARK_CONTRACT_HASH
  kBlobMismatch,      ///< size, CRCs or hash disagree with the blob
};

const char* manifest_status_string(ManifestStatus status);

/// Scans the top level of a JSON object for the ManifestFields keys (nested values are
/// skipped). `json` need not be NUL-terminated. Returns kOk or kSyntax.
ManifestStatus scan_manifest(const char* json, std::size_t size, ManifestFields* out);

/// Checks scanned fields against a parsed blob and the compiled-in contract hash.
ManifestStatus check_manifest(const ManifestFields& fields, const Blob& blob);

}  // namespace earmark
