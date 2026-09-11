// Bounded single-threaded FIFO of floats with partial-transfer semantics.
//
// Every operation moves as much as it can and returns how many samples it moved:
// write() accepts at most space() samples, read() and peek() return at most size().
// The storage comes from the arena, so no operation ever allocates.
//
// The Python model of these semantics is earmark.export.reference.RingBufferModel;
// the ringbuf golden replays 400 random operations against both.
#pragma once

#include <cstddef>

#include "arena.h"

namespace earmark {

class RingBuffer {
 public:
  /// Arena bytes needed for a buffer of `capacity` samples.
  static std::size_t arena_bytes(std::size_t capacity) { return Arena::aligned(capacity * sizeof(float)); }

  /// Takes `capacity` samples of storage from `arena`. False if capacity is 0 or the
  /// arena is exhausted.
  bool init(Arena& arena, std::size_t capacity);

  /// Appends up to `count` samples from `data`; returns how many were accepted.
  std::size_t write(const float* data, std::size_t count);
  /// Appends up to `count` zeros; returns how many were accepted.
  std::size_t write_zeros(std::size_t count);
  /// Pops up to `count` samples, oldest first, into `out`; returns how many.
  std::size_t read(float* out, std::size_t count);
  /// Copies up to `count` samples, oldest first, without removing them.
  std::size_t peek(float* out, std::size_t count) const;
  /// Drops up to `count` of the oldest samples; returns how many.
  std::size_t discard(std::size_t count);
  /// Empties the buffer (the storage is kept).
  void clear();

  std::size_t size() const { return size_; }
  std::size_t space() const { return capacity_ - size_; }
  std::size_t capacity() const { return capacity_; }

 private:
  // Copies `count` samples starting `offset` samples after the oldest one.
  void copy_out(std::size_t offset, float* out, std::size_t count) const;

  float* data_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t head_ = 0;  // index of the oldest sample
  std::size_t size_ = 0;
};

}  // namespace earmark
