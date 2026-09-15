#include "arena.h"

#include <cstdlib>
#include <cstring>

namespace earmark {

Arena::~Arena() { std::free(owned_); }

bool Arena::reserve(std::size_t bytes) {
  if (base_ != nullptr || bytes > SIZE_MAX - kAlignment) return false;
  // One allocation, over-sized so that the base can be aligned by hand
  // (std::aligned_alloc is missing on older macOS and needs size % align == 0).
  void* block = std::malloc(bytes + kAlignment);
  if (block == nullptr) return false;
  const auto address = reinterpret_cast<std::uintptr_t>(block);
  const std::uintptr_t aligned_address = (address + kAlignment - 1) & ~std::uintptr_t{kAlignment - 1};
  owned_ = block;
  base_ = reinterpret_cast<unsigned char*>(aligned_address);
  capacity_ = bytes;
  used_ = 0;
  frozen_ = false;
  return true;
}

bool Arena::attach(void* buffer, std::size_t bytes) {
  if (base_ != nullptr || buffer == nullptr) return false;
  if (reinterpret_cast<std::uintptr_t>(buffer) % kAlignment != 0) return false;
  base_ = static_cast<unsigned char*>(buffer);
  capacity_ = bytes;
  used_ = 0;
  frozen_ = false;
  return true;
}

void* Arena::allocate(std::size_t bytes) {
  if (frozen_ || base_ == nullptr) return nullptr;
  if (bytes > SIZE_MAX - kAlignment) return nullptr;
  const std::size_t size = aligned(bytes == 0 ? 1 : bytes);
  if (size > capacity_ - used_) return nullptr;
  unsigned char* block = base_ + used_;
  used_ += size;
  std::memset(block, 0, size);
  return block;
}

}  // namespace earmark
