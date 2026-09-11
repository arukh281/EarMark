#include "ringbuf.h"

#include <algorithm>
#include <cstring>

namespace earmark {

bool RingBuffer::init(Arena& arena, std::size_t capacity) {
  if (capacity == 0) return false;
  data_ = arena.allocate_array<float>(capacity);
  if (data_ == nullptr) return false;
  capacity_ = capacity;
  clear();
  return true;
}

void RingBuffer::clear() {
  head_ = 0;
  size_ = 0;
}

std::size_t RingBuffer::write(const float* data, std::size_t count) {
  const std::size_t accepted = std::min(count, space());
  if (accepted == 0) return 0;
  const std::size_t tail = (head_ + size_) % capacity_;
  const std::size_t first = std::min(accepted, capacity_ - tail);
  std::memcpy(data_ + tail, data, first * sizeof(float));
  std::memcpy(data_, data + first, (accepted - first) * sizeof(float));
  size_ += accepted;
  return accepted;
}

std::size_t RingBuffer::write_zeros(std::size_t count) {
  const std::size_t accepted = std::min(count, space());
  if (accepted == 0) return 0;
  const std::size_t tail = (head_ + size_) % capacity_;
  const std::size_t first = std::min(accepted, capacity_ - tail);
  std::memset(data_ + tail, 0, first * sizeof(float));
  std::memset(data_, 0, (accepted - first) * sizeof(float));
  size_ += accepted;
  return accepted;
}

void RingBuffer::copy_out(std::size_t offset, float* out, std::size_t count) const {
  const std::size_t start = (head_ + offset) % capacity_;
  const std::size_t first = std::min(count, capacity_ - start);
  std::memcpy(out, data_ + start, first * sizeof(float));
  std::memcpy(out + first, data_, (count - first) * sizeof(float));
}

std::size_t RingBuffer::peek(float* out, std::size_t count) const {
  const std::size_t taken = std::min(count, size_);
  if (taken > 0) copy_out(0, out, taken);
  return taken;
}

std::size_t RingBuffer::read(float* out, std::size_t count) {
  const std::size_t taken = peek(out, count);
  discard(taken);
  return taken;
}

std::size_t RingBuffer::discard(std::size_t count) {
  const std::size_t taken = std::min(count, size_);
  if (taken == 0) return 0;
  head_ = (head_ + taken) % capacity_;
  size_ -= taken;
  if (size_ == 0) head_ = 0;
  return taken;
}

}  // namespace earmark
