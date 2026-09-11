#include "alloc_counter.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define EARMARK_TEST_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define EARMARK_TEST_ASAN 1
#endif

namespace {

std::atomic<bool> g_enabled{false};
std::atomic<uint64_t> g_count{0};

inline void note_allocation() {
  if (g_enabled.load(std::memory_order_relaxed)) g_count.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

// ------------------------------------------------------------------------ backends

#if defined(EARMARK_TEST_ASAN)

#include <sanitizer/allocator_interface.h>

namespace {
void on_malloc(const volatile void*, std::size_t) { note_allocation(); }
void on_free(const volatile void*) {}
struct HookInstaller {
  HookInstaller() { __sanitizer_install_malloc_and_free_hooks(on_malloc, on_free); }
};
const HookInstaller g_installer;
void backend_start() {}
void backend_stop() {}
}  // namespace
#define EARMARK_TEST_BACKEND "sanitizer-hooks"
#define EARMARK_TEST_SEES_MALLOC true

#elif defined(__APPLE__)

// libmalloc calls this (when set) for every allocation event; type bit 2 is
// MALLOC_LOG_TYPE_ALLOCATE (realloc reports allocate | deallocate).
typedef void(malloc_logger_t)(uint32_t type, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3, uintptr_t result,
                              uint32_t num_hot_frames_to_skip);
extern "C" malloc_logger_t* malloc_logger;

namespace {
void log_event(uint32_t type, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uint32_t) {
  if ((type & 2u) != 0) note_allocation();
}
void backend_start() { malloc_logger = log_event; }
void backend_stop() { malloc_logger = nullptr; }
}  // namespace
#define EARMARK_TEST_BACKEND "darwin-malloc-logger"
#define EARMARK_TEST_SEES_MALLOC true

#elif defined(__GLIBC__)

#include <cerrno>
#include <stdlib.h>

// Definitions in the executable interpose glibc's for the whole process (ELF symbol
// interposition); the __libc_* entry points are glibc's own implementations.
extern "C" {
void* __libc_malloc(std::size_t size);
void* __libc_calloc(std::size_t count, std::size_t size);
void* __libc_realloc(void* ptr, std::size_t size);
void* __libc_memalign(std::size_t alignment, std::size_t size);

void* malloc(std::size_t size) __THROW {
  note_allocation();
  return __libc_malloc(size);
}
void* calloc(std::size_t count, std::size_t size) __THROW {
  note_allocation();
  return __libc_calloc(count, size);
}
void* realloc(void* ptr, std::size_t size) __THROW {
  note_allocation();
  return __libc_realloc(ptr, size);
}
void* memalign(std::size_t alignment, std::size_t size) __THROW {
  note_allocation();
  return __libc_memalign(alignment, size);
}
void* aligned_alloc(std::size_t alignment, std::size_t size) __THROW {
  note_allocation();
  return __libc_memalign(alignment, size);
}
int posix_memalign(void** out, std::size_t alignment, std::size_t size) __THROW {
  note_allocation();
  void* block = __libc_memalign(alignment, size);
  if (block == nullptr) return ENOMEM;
  *out = block;
  return 0;
}
}  // extern "C"

namespace {
void backend_start() {}
void backend_stop() {}
}  // namespace
#define EARMARK_TEST_BACKEND "glibc-interpose"
#define EARMARK_TEST_SEES_MALLOC true

#else

// Fallback: count C++ allocations only.
void* operator new(std::size_t size) {
  note_allocation();
  if (void* block = std::malloc(size == 0 ? 1 : size)) return block;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
  note_allocation();
  if (void* block = std::malloc(size == 0 ? 1 : size)) return block;
  throw std::bad_alloc();
}
void operator delete(void* block) noexcept { std::free(block); }
void operator delete[](void* block) noexcept { std::free(block); }
void operator delete(void* block, std::size_t) noexcept { std::free(block); }
void operator delete[](void* block, std::size_t) noexcept { std::free(block); }

namespace {
void backend_start() {}
void backend_stop() {}
}  // namespace
#define EARMARK_TEST_BACKEND "operator-new-only"
#define EARMARK_TEST_SEES_MALLOC false

#endif

// ------------------------------------------------------------------------- public

namespace earmark_test {

void alloc_counter_start() {
  g_count.store(0, std::memory_order_relaxed);
  backend_start();
  g_enabled.store(true, std::memory_order_seq_cst);
}

uint64_t alloc_counter_stop() {
  g_enabled.store(false, std::memory_order_seq_cst);
  backend_stop();
  return g_count.load(std::memory_order_relaxed);
}

const char* alloc_counter_backend() { return EARMARK_TEST_BACKEND; }

bool alloc_counter_sees_malloc() { return EARMARK_TEST_SEES_MALLOC; }

}  // namespace earmark_test
