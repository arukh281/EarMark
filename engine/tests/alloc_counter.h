// Counts heap allocations made by any code in the process between start and stop.
//
// Backends, picked at compile time:
// * "sanitizer-hooks": ASan builds, via __sanitizer_install_malloc_and_free_hooks.
// * "darwin-malloc-logger": macOS, via libmalloc's malloc_logger callback.
// * "glibc-interpose": Linux/glibc, by defining malloc & co. in the test executable.
// * "operator-new-only": anything else; sees C++ operator new but not raw malloc.
// The positive-control test in test_alloc.cpp proves the active backend works, so a
// broken hook fails loudly instead of passing vacuously.
#pragma once

#include <cstdint>

namespace earmark_test {

/// Resets the count and starts counting.
void alloc_counter_start();
/// Stops counting and returns the allocations seen since alloc_counter_start().
uint64_t alloc_counter_stop();
/// Name of the active backend.
const char* alloc_counter_backend();
/// False for the operator-new-only fallback.
bool alloc_counter_sees_malloc();

}  // namespace earmark_test
