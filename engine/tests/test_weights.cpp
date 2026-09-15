#include <cstring>
#include <string>
#include <vector>

#include "arena.h"
#include "catch_amalgamated.hpp"
#include "test_support.h"
#include "weights.h"

using earmark::Blob;
using earmark::BlobStatus;
using earmark::ManifestFields;
using earmark::ManifestStatus;
using earmark::TensorView;

namespace {

std::vector<uint8_t> small_blob() {
  std::vector<uint8_t> bytes = earmark_test::read_file(earmark_test::golden_path("weights_small.emwb"));
  REQUIRE(!bytes.empty());
  return bytes;
}

std::string small_manifest() {
  const std::vector<uint8_t> bytes = earmark_test::read_file(earmark_test::golden_path("weights_small.json"));
  REQUIRE(!bytes.empty());
  return std::string(bytes.begin(), bytes.end());
}

void store_u32(std::vector<uint8_t>& bytes, std::size_t offset, uint32_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

// Recomputes the header/table CRC after an intentional header edit.
void fix_table_crc(std::vector<uint8_t>& bytes) {
  uint32_t count;
  std::memcpy(&count, bytes.data() + 12, sizeof(count));
  const uint32_t head = earmark::crc32(bytes.data(), 56);
  store_u32(bytes, 56, earmark::crc32(bytes.data() + 64, count * earmark::kBlobEntryBytes, head));
}

std::string replace(std::string text, const std::string& from, const std::string& to) {
  const std::size_t at = text.find(from);
  REQUIRE(at != std::string::npos);
  return text.replace(at, from.size(), to);
}

}  // namespace

TEST_CASE("crc32 matches zlib", "[weights]") {
  const char check[] = "123456789";
  REQUIRE(earmark::crc32(check, 9) == 0xCBF43926u);
  // Chaining equals one pass, as with zlib.crc32(b, zlib.crc32(a)).
  REQUIRE(earmark::crc32(check + 4, 5, earmark::crc32(check, 4)) == 0xCBF43926u);
  REQUIRE(earmark::crc32(check, 0) == 0u);
}

TEST_CASE("the small blob parses with every dtype and rank", "[weights]") {
  const std::vector<uint8_t> bytes = small_blob();
  Blob blob;
  REQUIRE(blob.parse(bytes.data(), bytes.size()) == BlobStatus::kOk);
  REQUIRE(std::string(blob.contract_hash()) == EARMARK_CONTRACT_HASH);
  REQUIRE(blob.count() == 8);
  REQUIRE(blob.file_bytes() == bytes.size());

  TensorView v;
  REQUIRE(blob.find("test.arange", &v));
  REQUIRE(v.has_shape({2, 3, 4}));
  for (uint32_t i = 0; i < 24; ++i) REQUIRE(v.f32()[i] == 0.5f * static_cast<float>(i));
  REQUIRE(v.i32() == nullptr);

  REQUIRE(blob.find("test.int", &v));
  REQUIRE(v.has_shape({2, 3}));
  const int32_t ints[] = {1, -2, 3, 4, 5, -6};
  REQUIRE(std::memcmp(v.i32(), ints, sizeof(ints)) == 0);
  REQUIRE(v.f32() == nullptr);

  REQUIRE(blob.find("test.scalar", &v));
  REQUIRE(v.ndim == 0);
  REQUIRE(v.numel == 1);
  REQUIRE(v.f32()[0] == 3.5f);

  REQUIRE(blob.find("test.empty", &v));
  REQUIRE(v.has_shape({0}));
  REQUIRE(v.numel == 0);

  REQUIRE(blob.find("test.rank6", &v));
  REQUIRE(v.has_shape({1, 2, 1, 3, 1, 2}));
  for (uint32_t i = 0; i < 12; ++i) REQUIRE(v.f32()[i] == static_cast<float>(i) - 5.0f);

  REQUIRE(blob.find("conditioner.null_embedding", &v));
  REQUIRE(v.has_shape({EARMARK_EMBEDDING_DIM}));
  for (uint32_t i = 0; i < EARMARK_EMBEDDING_DIM; ++i) {
    REQUIRE(v.f32()[i] == static_cast<float>(static_cast<int>((i * 37) % 17) - 8) / 16.0f);
  }
  REQUIRE(blob.find("const.erb_norm_init", &v));
  REQUIRE(v.has_shape({EARMARK_ERB_BANDS}));
  REQUIRE(blob.find("const.spec_norm_init", &v));
  REQUIRE(v.has_shape({EARMARK_DF_BINS}));

  REQUIRE_FALSE(blob.find("missing", &v));
  REQUIRE_FALSE(blob.find("", &v));
  REQUIRE(blob.tensor(0, &v));
  REQUIRE(std::string(v.name) == "test.arange");
  REQUIRE_FALSE(blob.tensor(8, &v));
  // Tensor data starts on 64-byte boundaries relative to the blob start.
  for (uint32_t i = 0; i < blob.count(); ++i) {
    REQUIRE(blob.tensor(i, &v));
    REQUIRE((static_cast<const uint8_t*>(v.data) - bytes.data()) % 64 == 0);
  }
}

TEST_CASE("corrupt blobs are rejected", "[weights]") {
  const std::vector<uint8_t> good = small_blob();
  Blob blob;
  REQUIRE(blob.parse(good.data(), 10) == BlobStatus::kTooSmall);
  REQUIRE(blob.parse(nullptr, 100) == BlobStatus::kTooSmall);
  REQUIRE(blob.parse(good.data(), good.size() - 64) == BlobStatus::kSizeMismatch);

  std::vector<uint8_t> bad = good;
  bad[0] = 'X';
  REQUIRE(blob.parse(bad.data(), bad.size()) == BlobStatus::kBadMagic);

  bad = good;
  store_u32(bad, 8, 2);
  REQUIRE(blob.parse(bad.data(), bad.size()) == BlobStatus::kBadVersion);

  bad = good;
  bad[16] = 'Z';
  REQUIRE(blob.parse(bad.data(), bad.size()) == BlobStatus::kBadContractHash);

  bad = good;
  bad.back() ^= 0x01;  // inside the (padded) data section
  REQUIRE(blob.parse(bad.data(), bad.size()) == BlobStatus::kDataCrc);
  REQUIRE(blob.parse(bad.data(), bad.size(), false) == BlobStatus::kOk);

  bad = good;
  bad[64 + 80] ^= 0x01;  // first entry's shape[0]
  REQUIRE(blob.parse(bad.data(), bad.size()) == BlobStatus::kTableCrc);
  REQUIRE(blob.parse(bad.data(), bad.size(), false) == BlobStatus::kBadEntry);  // numel no longer matches

  bad = good;
  std::memcpy(bad.data() + 64 + earmark::kBlobEntryBytes, bad.data() + 64, earmark::kBlobNameBytes);  // clone name 0
  fix_table_crc(bad);
  REQUIRE(blob.parse(bad.data(), bad.size()) == BlobStatus::kDuplicateName);

  bad = good;
  bad[64 + 3 * earmark::kBlobEntryBytes + 76] = 7;  // ndim 7 on entry 3
  fix_table_crc(bad);
  REQUIRE(blob.parse(bad.data(), bad.size()) == BlobStatus::kBadEntry);

  // A failed parse leaves nothing behind.
  TensorView v;
  REQUIRE_FALSE(blob.find("test.arange", &v));
  REQUIRE(blob.count() == 0);
}

TEST_CASE("the manifest scanner reads the flat keys and checks them against the blob", "[weights]") {
  const std::vector<uint8_t> bytes = small_blob();
  Blob blob;
  REQUIRE(blob.parse(bytes.data(), bytes.size()) == BlobStatus::kOk);
  const std::string manifest = small_manifest();
  ManifestFields fields;
  REQUIRE(earmark::scan_manifest(manifest.data(), manifest.size(), &fields) == ManifestStatus::kOk);
  REQUIRE(std::string(fields.format) == earmark::kManifestFormat);
  REQUIRE(fields.format_version == 1);
  REQUIRE(std::string(fields.contract_hash) == EARMARK_CONTRACT_HASH);
  REQUIRE(fields.blob_bytes == static_cast<int64_t>(bytes.size()));
  REQUIRE(earmark::check_manifest(fields, blob) == ManifestStatus::kOk);

  auto check_text = [&](const std::string& text) {
    ManifestFields f;
    const ManifestStatus scanned = earmark::scan_manifest(text.data(), text.size(), &f);
    return scanned == ManifestStatus::kOk ? earmark::check_manifest(f, blob) : scanned;
  };
  const std::string size_key = "\"blob_bytes\": " + std::to_string(bytes.size());
  REQUIRE(check_text(replace(manifest, size_key, "\"blob_bytes\": 12")) == ManifestStatus::kBlobMismatch);
  REQUIRE(check_text(replace(manifest, EARMARK_CONTRACT_HASH, "0123456789abcdef")) == ManifestStatus::kContractMismatch);
  REQUIRE(check_text(replace(manifest, "\"format_version\": 1", "\"format_version\": 2")) == ManifestStatus::kWrongFormat);
  REQUIRE(check_text(replace(manifest, "\"blob_bytes\"", "\"blob_size\"")) == ManifestStatus::kMissingField);
  REQUIRE(check_text(replace(manifest, "{", "[")) == ManifestStatus::kSyntax);
  REQUIRE(check_text(manifest.substr(0, manifest.size() / 2)) == ManifestStatus::kSyntax);
  REQUIRE(check_text(replace(manifest, size_key, size_key + ", \"blob_bytes\": 1")) == ManifestStatus::kSyntax);
  REQUIRE(check_text(replace(manifest, size_key, "\"blob_bytes\": 1.5")) == ManifestStatus::kSyntax);
  // Keys nested below the top level are ignored.
  REQUIRE(check_text(replace(manifest, "{", "{\"model\": {\"blob_bytes\": 3, \"x\": [\"}\", {\"y\": null}]}, ")) ==
          ManifestStatus::kOk);

  // A blob exported under another contract is refused even with a consistent manifest.
  std::vector<uint8_t> other = bytes;
  std::memcpy(other.data() + 16, "0123456789abcdef", 16);
  fix_table_crc(other);
  Blob other_blob;
  REQUIRE(other_blob.parse(other.data(), other.size()) == BlobStatus::kOk);
  ManifestFields f;
  REQUIRE(earmark::scan_manifest(manifest.data(), manifest.size(), &f) == ManifestStatus::kOk);
  REQUIRE(earmark::check_manifest(f, other_blob) == ManifestStatus::kContractMismatch);
}
