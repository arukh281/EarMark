#include <cstring>
#include <vector>

#include "arena.h"
#include "catch_amalgamated.hpp"
#include "ringbuf.h"
#include "test_support.h"

using earmark::Arena;
using earmark::RingBuffer;

namespace {
// Op codes of the golden schedule (earmark.export.golden.RING_*).
enum : int32_t { kWrite = 0, kRead = 1, kWriteZeros = 2, kDiscard = 3, kPeek = 4 };
}  // namespace

TEST_CASE("ring buffer replays the golden operation schedule", "[ringbuf]") {
  const earmark_test::Golden golden("ringbuf");
  const int32_t capacity = golden.i32("capacity")[0];
  const int32_t* ops = golden.i32("ops");
  const std::size_t op_count = golden.numel("ops") / 2;
  const float* input = golden.f32("input");
  const std::size_t input_count = golden.numel("input");
  const int32_t* counts = golden.i32("expected_counts");
  const int32_t* sizes = golden.i32("expected_sizes");
  const float* expected = golden.f32("expected_output");
  const std::size_t expected_count = golden.numel("expected_output");
  REQUIRE(golden.numel("expected_counts") == op_count);

  Arena arena;
  REQUIRE(arena.reserve(RingBuffer::arena_bytes(static_cast<std::size_t>(capacity))));
  RingBuffer ring;
  REQUIRE(ring.init(arena, static_cast<std::size_t>(capacity)));

  std::vector<float> scratch(256);
  std::size_t in_pos = 0;
  std::size_t out_pos = 0;
  for (std::size_t i = 0; i < op_count; ++i) {
    const int32_t kind = ops[2 * i];
    const auto n = static_cast<std::size_t>(ops[2 * i + 1]);
    REQUIRE(n <= scratch.size());
    std::size_t count = 0;
    switch (kind) {
      case kWrite:
        REQUIRE(in_pos + n <= input_count);
        count = ring.write(input + in_pos, n);
        in_pos += n;
        break;
      case kWriteZeros: count = ring.write_zeros(n); break;
      case kDiscard: count = ring.discard(n); break;
      case kRead:
      case kPeek:
        count = kind == kRead ? ring.read(scratch.data(), n) : ring.peek(scratch.data(), n);
        REQUIRE(out_pos + count <= expected_count);
        REQUIRE(std::memcmp(scratch.data(), expected + out_pos, count * sizeof(float)) == 0);
        out_pos += count;
        break;
      default: FAIL("unknown op " << kind);
    }
    INFO("op " << i << " kind " << kind << " n " << n);
    REQUIRE(count == static_cast<std::size_t>(counts[i]));
    REQUIRE(ring.size() == static_cast<std::size_t>(sizes[i]));
    REQUIRE(ring.space() == static_cast<std::size_t>(capacity) - ring.size());
  }
  REQUIRE(in_pos == input_count);
  REQUIRE(out_pos == expected_count);
  std::printf("  ringbuf: %zu ops replayed bit-exactly\n", op_count);
}

TEST_CASE("ring buffer wraps, fills and empties", "[ringbuf]") {
  Arena arena;
  REQUIRE(arena.reserve(RingBuffer::arena_bytes(5)));
  RingBuffer ring;
  REQUIRE(ring.init(arena, 5));
  const float a[] = {1, 2, 3, 4};
  float out[8] = {};
  REQUIRE(ring.write(a, 4) == 4);
  REQUIRE(ring.read(out, 3) == 3);  // head moves to index 3
  REQUIRE(ring.write(a, 4) == 4);   // wraps around the end
  REQUIRE(ring.size() == 5);
  REQUIRE(ring.write(a, 1) == 0);   // full
  REQUIRE(ring.peek(out, 8) == 5);
  const float want[] = {4, 1, 2, 3, 4};
  REQUIRE(std::memcmp(out, want, sizeof(want)) == 0);
  REQUIRE(ring.discard(2) == 2);
  REQUIRE(ring.write_zeros(9) == 2);
  REQUIRE(ring.read(out, 8) == 5);
  const float want2[] = {2, 3, 4, 0, 0};
  REQUIRE(std::memcmp(out, want2, sizeof(want2)) == 0);
  REQUIRE(ring.size() == 0);
  REQUIRE(ring.read(out, 1) == 0);
  ring.clear();
  REQUIRE(ring.space() == 5);
}

TEST_CASE("ring buffer init fails without storage", "[ringbuf]") {
  Arena arena;
  RingBuffer ring;
  REQUIRE_FALSE(ring.init(arena, 4));  // arena has no block
  REQUIRE(arena.reserve(64));
  REQUIRE_FALSE(ring.init(arena, 0));
  REQUIRE_FALSE(ring.init(arena, 1000));  // larger than the arena
}
