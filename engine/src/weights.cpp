#include "weights.h"

#include <array>
#include <cstring>

#include "earmark_constants.h"

namespace earmark {

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the weight blob is little-endian; big-endian hosts are not supported"
#endif

namespace {

constexpr std::array<uint32_t, 256> make_crc_table() {
  std::array<uint32_t, 256> table{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int bit = 0; bit < 8; ++bit) c = (c & 1u) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    table[i] = c;
  }
  return table;
}

constexpr std::array<uint32_t, 256> kCrcTable = make_crc_table();
constexpr char kMagic[8] = {'E', 'M', 'W', 'B', 'L', 'O', 'B', '\0'};
constexpr std::size_t kHeaderCrcSpan = 56;

// Header field offsets.
constexpr std::size_t kOffVersion = 8;
constexpr std::size_t kOffCount = 12;
constexpr std::size_t kOffHash = 16;
constexpr std::size_t kOffTable = 32;
constexpr std::size_t kOffData = 40;
constexpr std::size_t kOffFileBytes = 48;
constexpr std::size_t kOffTableCrc = 56;
constexpr std::size_t kOffDataCrc = 60;

// Entry field offsets.
constexpr std::size_t kEntDtype = 72;
constexpr std::size_t kEntNdim = 76;
constexpr std::size_t kEntShape = 80;
constexpr std::size_t kEntOffset = 104;
constexpr std::size_t kEntNumel = 112;
constexpr std::size_t kEntCrc = 120;

uint32_t load_u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

uint64_t load_u64(const uint8_t* p) {
  uint64_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

bool is_lower_hex(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }

// Length of the NUL-terminated name in a 72-byte field, or kBlobNameBytes if unterminated.
std::size_t name_length(const uint8_t* field) {
  for (std::size_t i = 0; i < kBlobNameBytes; ++i) {
    if (field[i] == 0) return i;
  }
  return kBlobNameBytes;
}

}  // namespace

uint32_t crc32(const void* data, std::size_t size, uint32_t crc) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  uint32_t c = crc ^ 0xFFFFFFFFu;
  for (std::size_t i = 0; i < size; ++i) c = kCrcTable[(c ^ bytes[i]) & 0xFFu] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

const char* blob_status_string(BlobStatus status) {
  switch (status) {
    case BlobStatus::kOk: return "ok";
    case BlobStatus::kTooSmall: return "blob is shorter than its header";
    case BlobStatus::kBadMagic: return "bad blob magic";
    case BlobStatus::kBadVersion: return "unsupported blob format version";
    case BlobStatus::kBadContractHash: return "malformed contract hash field";
    case BlobStatus::kBadLayout: return "inconsistent blob layout";
    case BlobStatus::kSizeMismatch: return "blob size differs from its header";
    case BlobStatus::kTableCrc: return "header/table CRC mismatch";
    case BlobStatus::kDataCrc: return "data CRC mismatch";
    case BlobStatus::kBadEntry: return "malformed tensor entry";
    case BlobStatus::kDuplicateName: return "duplicate tensor name";
    case BlobStatus::kTensorCrc: return "tensor CRC mismatch";
    case BlobStatus::kMisaligned: return "tensor data is not 4-byte aligned";
  }
  return "unknown blob status";
}

bool TensorView::has_shape(std::initializer_list<uint32_t> dims) const {
  if (dims.size() != ndim) return false;
  uint32_t i = 0;
  for (uint32_t d : dims) {
    if (shape[i++] != d) return false;
  }
  return true;
}

BlobStatus Blob::parse(const uint8_t* data, std::size_t size, bool verify_crc) {
  *this = Blob{};
  if (data == nullptr || size < kBlobHeaderBytes) return BlobStatus::kTooSmall;
  if (std::memcmp(data, kMagic, sizeof(kMagic)) != 0) return BlobStatus::kBadMagic;
  if (load_u32(data + kOffVersion) != kBlobFormatVersion) return BlobStatus::kBadVersion;
  const uint32_t count = load_u32(data + kOffCount);
  if (count > kBlobMaxTensors) return BlobStatus::kBadLayout;
  char hash[17] = {};
  for (std::size_t i = 0; i < 16; ++i) {
    hash[i] = static_cast<char>(data[kOffHash + i]);
    if (!is_lower_hex(hash[i])) return BlobStatus::kBadContractHash;
  }
  const uint64_t table_offset = load_u64(data + kOffTable);
  const uint64_t data_offset = load_u64(data + kOffData);
  const uint64_t file_bytes = load_u64(data + kOffFileBytes);
  const uint32_t table_crc = load_u32(data + kOffTableCrc);
  const uint32_t data_crc = load_u32(data + kOffDataCrc);
  if (table_offset != kBlobHeaderBytes) return BlobStatus::kBadLayout;
  const uint64_t table_end = table_offset + static_cast<uint64_t>(count) * kBlobEntryBytes;
  if (data_offset % kBlobAlignment != 0 || data_offset < table_end) return BlobStatus::kBadLayout;
  if (file_bytes != size) return BlobStatus::kSizeMismatch;
  if (data_offset > file_bytes) return BlobStatus::kBadLayout;
  if (verify_crc) {
    const uint32_t head = crc32(data, kHeaderCrcSpan);
    if (crc32(data + table_offset, static_cast<std::size_t>(table_end - table_offset), head) != table_crc) {
      return BlobStatus::kTableCrc;
    }
    if (crc32(data + data_offset, static_cast<std::size_t>(file_bytes - data_offset)) != data_crc) {
      return BlobStatus::kDataCrc;
    }
  }

  for (uint32_t index = 0; index < count; ++index) {
    const uint8_t* entry = data + table_offset + static_cast<uint64_t>(index) * kBlobEntryBytes;
    const std::size_t length = name_length(entry);
    if (length == 0 || length >= kBlobNameBytes) return BlobStatus::kBadEntry;
    for (uint32_t other = 0; other < index; ++other) {
      const uint8_t* previous = data + table_offset + static_cast<uint64_t>(other) * kBlobEntryBytes;
      if (std::memcmp(previous, entry, length + 1) == 0) return BlobStatus::kDuplicateName;
    }
    const uint32_t dtype = load_u32(entry + kEntDtype);
    const uint32_t ndim = load_u32(entry + kEntNdim);
    if (dtype != kDtypeFloat32 && dtype != kDtypeInt32) return BlobStatus::kBadEntry;
    if (ndim > kBlobMaxNdim) return BlobStatus::kBadEntry;
    uint64_t product = 1;
    for (uint32_t d = 0; d < kBlobMaxNdim; ++d) {
      const uint64_t dim = load_u32(entry + kEntShape + 4 * d);
      if (d >= ndim) {
        if (dim != 0) return BlobStatus::kBadEntry;
        continue;
      }
      if (dim != 0 && product > UINT64_MAX / dim) return BlobStatus::kBadEntry;
      product *= dim;
    }
    const uint64_t offset = load_u64(entry + kEntOffset);
    const uint64_t numel = load_u64(entry + kEntNumel);
    if (numel != product || offset % kBlobAlignment != 0) return BlobStatus::kBadEntry;
    if (offset > file_bytes - data_offset || numel > (file_bytes - data_offset - offset) / 4) {
      return BlobStatus::kBadEntry;
    }
    const uint8_t* payload = data + data_offset + offset;
    if (verify_crc && crc32(payload, static_cast<std::size_t>(numel * 4)) != load_u32(entry + kEntCrc)) {
      return BlobStatus::kTensorCrc;
    }
    if (reinterpret_cast<std::uintptr_t>(payload) % 4 != 0) return BlobStatus::kMisaligned;
  }

  data_ = data;
  size_ = size;
  count_ = count;
  data_offset_ = data_offset;
  table_crc_ = table_crc;
  data_crc_ = data_crc;
  std::memcpy(hash_, hash, sizeof(hash_));
  return BlobStatus::kOk;
}

bool Blob::tensor(uint32_t index, TensorView* out) const {
  if (data_ == nullptr || index >= count_ || out == nullptr) return false;
  const uint8_t* entry = data_ + kBlobHeaderBytes + static_cast<uint64_t>(index) * kBlobEntryBytes;
  TensorView view;
  view.name = reinterpret_cast<const char*>(entry);
  view.dtype = load_u32(entry + kEntDtype);
  view.ndim = load_u32(entry + kEntNdim);
  for (uint32_t d = 0; d < kBlobMaxNdim; ++d) view.shape[d] = load_u32(entry + kEntShape + 4 * d);
  view.numel = load_u64(entry + kEntNumel);
  view.data = data_ + data_offset_ + load_u64(entry + kEntOffset);
  *out = view;
  return true;
}

bool Blob::find(const char* name, TensorView* out) const {
  if (name == nullptr) return false;
  const std::size_t length = std::strlen(name);
  if (length == 0 || length >= kBlobNameBytes) return false;
  for (uint32_t index = 0; index < count_; ++index) {
    const uint8_t* entry = data_ + kBlobHeaderBytes + static_cast<uint64_t>(index) * kBlobEntryBytes;
    if (std::memcmp(entry, name, length + 1) == 0) return tensor(index, out);
  }
  return false;
}

// ----------------------------------------------------------------------------- manifest

const char* manifest_status_string(ManifestStatus status) {
  switch (status) {
    case ManifestStatus::kOk: return "ok";
    case ManifestStatus::kSyntax: return "manifest is not a well-formed JSON object";
    case ManifestStatus::kMissingField: return "manifest lacks a required field";
    case ManifestStatus::kWrongFormat: return "manifest format or version is not supported";
    case ManifestStatus::kContractMismatch: return "contract hash differs from this engine's contract";
    case ManifestStatus::kBlobMismatch: return "manifest does not describe this blob";
  }
  return "unknown manifest status";
}

namespace {

// A tiny forward-only JSON reader: enough to walk one object's top-level members and
// skip every other value. It never allocates and never recurses.
class JsonCursor {
 public:
  JsonCursor(const char* text, std::size_t size) : p_(text), end_(text + size) {}

  void skip_ws() {
    while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
  }
  bool at_end() const { return p_ >= end_; }
  bool consume(char c) {
    skip_ws();
    if (p_ < end_ && *p_ == c) {
      ++p_;
      return true;
    }
    return false;
  }
  char peek() {
    skip_ws();
    return p_ < end_ ? *p_ : '\0';
  }

  // Reads a string into `out` (capacity bytes including the NUL). Escapes are kept
  // verbatim; `truncated` reports overflow. False on a syntax error.
  bool read_string(char* out, std::size_t capacity, bool* truncated) {
    if (!consume('"')) return false;
    std::size_t n = 0;
    *truncated = false;
    while (p_ < end_ && *p_ != '"') {
      char c = *p_++;
      if (static_cast<unsigned char>(c) < 0x20) return false;
      if (c == '\\') {
        if (p_ >= end_) return false;
        if (n + 1 < capacity) out[n++] = c;
        else *truncated = true;
        c = *p_++;
      }
      if (n + 1 < capacity) out[n++] = c;
      else *truncated = true;
    }
    if (p_ >= end_) return false;
    ++p_;  // closing quote
    if (capacity > 0) out[n] = '\0';
    return true;
  }

  // Reads an integer (no fraction or exponent). False if the value is not one.
  bool read_int(int64_t* value) {
    skip_ws();
    bool negative = false;
    if (p_ < end_ && *p_ == '-') {
      negative = true;
      ++p_;
    }
    if (p_ >= end_ || *p_ < '0' || *p_ > '9') return false;
    int64_t v = 0;
    while (p_ < end_ && *p_ >= '0' && *p_ <= '9') {
      const int digit = *p_++ - '0';
      if (v > (INT64_MAX - digit) / 10) return false;
      v = v * 10 + digit;
    }
    if (p_ < end_ && (*p_ == '.' || *p_ == 'e' || *p_ == 'E')) return false;
    *value = negative ? -v : v;
    return true;
  }

  // Skips any JSON value, including nested objects and arrays.
  bool skip_value() {
    skip_ws();
    if (p_ >= end_) return false;
    if (*p_ == '"') return skip_string();
    if (*p_ == '{' || *p_ == '[') {
      int depth = 0;
      while (p_ < end_) {
        const char c = *p_;
        if (c == '"') {
          if (!skip_string()) return false;
          continue;
        }
        ++p_;
        if (c == '{' || c == '[') ++depth;
        if (c == '}' || c == ']') {
          if (--depth == 0) return true;
        }
      }
      return false;
    }
    const char* start = p_;
    while (p_ < end_ && *p_ != ',' && *p_ != '}' && *p_ != ']' && *p_ != ' ' && *p_ != '\t' && *p_ != '\n' &&
           *p_ != '\r') {
      ++p_;
    }
    return p_ > start;
  }

 private:
  bool skip_string() {
    ++p_;  // opening quote
    while (p_ < end_ && *p_ != '"') {
      if (*p_ == '\\') ++p_;
      ++p_;
    }
    if (p_ >= end_) return false;
    ++p_;
    return true;
  }

  const char* p_;
  const char* end_;
};

}  // namespace

ManifestStatus scan_manifest(const char* json, std::size_t size, ManifestFields* out) {
  if (json == nullptr || out == nullptr) return ManifestStatus::kSyntax;
  *out = ManifestFields{};
  JsonCursor cursor(json, size);
  if (!cursor.consume('{')) return ManifestStatus::kSyntax;
  bool seen[6] = {};
  if (!cursor.consume('}')) {
    for (;;) {
      char key[64];
      bool truncated = false;
      if (!cursor.read_string(key, sizeof(key), &truncated)) return ManifestStatus::kSyntax;
      if (!cursor.consume(':')) return ManifestStatus::kSyntax;
      int field = -1;
      if (!truncated) {
        const char* names[6] = {"format", "format_version", "contract_hash", "blob_bytes", "blob_table_crc32",
                                "blob_data_crc32"};
        for (int i = 0; i < 6; ++i) {
          if (std::strcmp(key, names[i]) == 0) field = i;
        }
      }
      bool ok = true;
      if (field >= 0 && seen[field]) return ManifestStatus::kSyntax;  // duplicate key
      switch (field) {
        case 0: ok = cursor.peek() == '"' && cursor.read_string(out->format, sizeof(out->format), &truncated) && !truncated; break;
        case 1: ok = cursor.read_int(&out->format_version); break;
        case 2:
          ok = cursor.peek() == '"' &&
               cursor.read_string(out->contract_hash, sizeof(out->contract_hash), &truncated) && !truncated;
          break;
        case 3: ok = cursor.read_int(&out->blob_bytes); break;
        case 4: ok = cursor.read_int(&out->blob_table_crc32); break;
        case 5: ok = cursor.read_int(&out->blob_data_crc32); break;
        default: ok = cursor.skip_value(); break;
      }
      if (!ok) return ManifestStatus::kSyntax;
      if (field >= 0) seen[field] = true;
      if (cursor.consume(',')) continue;
      if (cursor.consume('}')) break;
      return ManifestStatus::kSyntax;
    }
  }
  cursor.skip_ws();
  return cursor.at_end() ? ManifestStatus::kOk : ManifestStatus::kSyntax;
}

ManifestStatus check_manifest(const ManifestFields& fields, const Blob& blob) {
  if (fields.format[0] == '\0' || fields.format_version < 0 || fields.contract_hash[0] == '\0' ||
      fields.blob_bytes < 0 || fields.blob_table_crc32 < 0 || fields.blob_data_crc32 < 0) {
    return ManifestStatus::kMissingField;
  }
  if (std::strcmp(fields.format, kManifestFormat) != 0 || fields.format_version != kBlobFormatVersion) {
    return ManifestStatus::kWrongFormat;
  }
  if (std::strcmp(fields.contract_hash, EARMARK_CONTRACT_HASH) != 0 ||
      std::strcmp(blob.contract_hash(), EARMARK_CONTRACT_HASH) != 0) {
    return ManifestStatus::kContractMismatch;
  }
  if (std::strcmp(fields.contract_hash, blob.contract_hash()) != 0 ||
      static_cast<uint64_t>(fields.blob_bytes) != blob.file_bytes() ||
      static_cast<uint64_t>(fields.blob_table_crc32) != blob.table_crc32() ||
      static_cast<uint64_t>(fields.blob_data_crc32) != blob.data_crc32()) {
    return ManifestStatus::kBlobMismatch;
  }
  return ManifestStatus::kOk;
}

}  // namespace earmark
