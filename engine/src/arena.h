// Fixed-capacity bump allocator: the only memory the engine touches after em_create.
//
// em_create adds up the arena_bytes() of every component, reserves one block, carves
// every buffer out of it and then calls freeze(). From then on allocate() returns
// nullptr, so no code on the audio path can allocate even by mistake.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace earmark {

class Arena {
 public:
  /// Every block starts on a 64-byte boundary (cache line and any SIMD width).
  static constexpr std::size_t kAlignment = 64;

  Arena() = default;
  ~Arena();
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  /// `bytes` rounded up to kAlignment. Components use it in their arena_bytes().
  static constexpr std::size_t aligned(std::size_t bytes) {
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
  }

  /// Reserves the backing block with one heap allocation. Returns false when out of
  /// memory or when the arena already has a block.
  bool reserve(std::size_t bytes);

  /// Uses caller-owned memory (for example a static buffer) instead of reserving.
  /// `buffer` must be kAlignment-aligned and outlive the arena; it is never freed here.
  bool attach(void* buffer, std::size_t bytes);

  /// A zero-filled, kAlignment-aligned block of `bytes`. Returns nullptr when the
  /// arena is exhausted, frozen or has no block. Never allocates from the heap.
  void* allocate(std::size_t bytes);

  /// allocate() for `count` trivially constructible objects, with an overflow check.
  template <typename T>
  T* allocate_array(std::size_t count) {
    static_assert(std::is_trivially_default_constructible_v<T> && std::is_trivially_destructible_v<T>,
                  "the arena only holds trivial types");
    if (count > SIZE_MAX / sizeof(T)) return nullptr;
    return static_cast<T*>(allocate(count * sizeof(T)));
  }

  /// After this, allocate() always fails.
  void freeze() { frozen_ = true; }

  bool frozen() const { return frozen_; }
  std::size_t used() const { return used_; }
  std::size_t capacity() const { return capacity_; }

 private:
  unsigned char* base_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t used_ = 0;
  void* owned_ = nullptr;
  bool frozen_ = false;
};

}  // namespace earmark
