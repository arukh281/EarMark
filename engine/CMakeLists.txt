# Earmark C++17 streaming engine: native build (Ninja) with Catch2 tests.
# The WASM build is engine/wasm/build.sh (Emscripten, CI only).
#
#   cmake -S engine -B engine/build -G Ninja
#   cmake --build engine/build && ctest --test-dir engine/build --output-on-failure
#
# Options: -DEARMARK_ASAN=ON -DEARMARK_UBSAN=ON (sanitizers on every target),
# -DEARMARK_BUILD_TESTS=OFF, -DEARMARK_BUILD_SHARED=OFF, -DEARMARK_WERROR=OFF.
cmake_minimum_required(VERSION 3.20)
project(earmark_engine VERSION 0.1.0 LANGUAGES CXX)

option(EARMARK_BUILD_TESTS "Build the Catch2 test executables" ON)
option(EARMARK_BUILD_SHARED "Build libearmark (shared library exporting only the C ABI)" ON)
option(EARMARK_ASAN "Build every target with AddressSanitizer" OFF)
option(EARMARK_UBSAN "Build every target with UndefinedBehaviorSanitizer" OFF)
option(EARMARK_WERROR "Treat warnings in engine sources as errors" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Sanitizers go on every target (engine, Catch2, tests) so the runtimes agree.
set(EARMARK_SANITIZERS "")
if(EARMARK_ASAN)
  list(APPEND EARMARK_SANITIZERS address)
endif()
if(EARMARK_UBSAN)
  list(APPEND EARMARK_SANITIZERS undefined)
endif()
if(EARMARK_SANITIZERS)
  list(JOIN EARMARK_SANITIZERS "," EARMARK_SANITIZER_LIST)
  add_compile_options(-fsanitize=${EARMARK_SANITIZER_LIST} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=${EARMARK_SANITIZER_LIST})
endif()

# ---------------------------------------------------------------------------- engine

set(EARMARK_SOURCES
  src/arena.cpp
  src/engine.cpp
  src/erb.cpp
  src/fft.cpp
  src/gru.cpp
  src/matvec_f32.cpp
  src/network.cpp
  src/resampler.cpp
  src/ringbuf.cpp
  src/state.cpp
  src/stft.cpp
  src/weights.cpp
)

add_library(earmark_objects OBJECT ${EARMARK_SOURCES})
target_include_directories(earmark_objects PUBLIC include PRIVATE src)
target_include_directories(earmark_objects SYSTEM PRIVATE third_party)
target_compile_definitions(earmark_objects PRIVATE POCKETFFT_NO_MULTITHREADING POCKETFFT_CACHE_SIZE=0)
# -ffp-contract=off: no FMA contraction, so native results match the WASM build
# (which has no FMA) and do not depend on the CPU. -fno-exceptions/-fno-rtti: the
# engine never throws; fft.cpp alone re-enables exceptions for pocketfft's init path.
target_compile_options(earmark_objects PRIVATE
  -Wall -Wextra -Wpedantic -Wshadow -Wdouble-promotion -Wnon-virtual-dtor
  -ffp-contract=off -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
  $<$<BOOL:${EARMARK_WERROR}>:-Werror>
)
set_source_files_properties(src/fft.cpp PROPERTIES COMPILE_OPTIONS "-fexceptions;-frtti")

# Static library for the tests and tools; its internal headers are visible to them.
add_library(earmark_core STATIC $<TARGET_OBJECTS:earmark_objects>)
target_include_directories(earmark_core PUBLIC include src)

if(EARMARK_BUILD_SHARED)
  # Exports only the EM_API functions (for the Python ctypes binding).
  add_library(earmark SHARED $<TARGET_OBJECTS:earmark_objects>)
  target_include_directories(earmark PUBLIC include)
endif()

# ----------------------------------------------------------------------------- tests

if(EARMARK_BUILD_TESTS)
  enable_testing()

  add_library(earmark_catch2 STATIC third_party/catch2/catch_amalgamated.cpp)
  target_include_directories(earmark_catch2 SYSTEM PUBLIC third_party/catch2)

  set(EARMARK_TEST_DEFINITIONS EARMARK_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/goldens")
  set(EARMARK_TEST_OPTIONS -Wall -Wextra -ffp-contract=off)

  add_executable(earmark_tests
    tests/test_support.cpp
    tests/test_ringbuf.cpp
    tests/test_resampler.cpp
    tests/test_stft.cpp
    tests/test_erb.cpp
    tests/test_gru.cpp
    tests/test_matvec.cpp
    tests/test_weights.cpp
    tests/test_engine.cpp
    tests/test_network.cpp
  )
  target_link_libraries(earmark_tests PRIVATE earmark_core earmark_catch2)
  target_compile_definitions(earmark_tests PRIVATE ${EARMARK_TEST_DEFINITIONS})
  target_compile_options(earmark_tests PRIVATE ${EARMARK_TEST_OPTIONS})

  add_executable(earmark_alloc_tests tests/test_alloc.cpp tests/alloc_counter.cpp tests/test_support.cpp)
  target_link_libraries(earmark_alloc_tests PRIVATE earmark_core earmark_catch2)
  target_compile_definitions(earmark_alloc_tests PRIVATE ${EARMARK_TEST_DEFINITIONS})
  target_compile_options(earmark_alloc_tests PRIVATE ${EARMARK_TEST_OPTIONS})

  foreach(tag ringbuf resampler stft erb gru matvec weights arena capi network)
    add_test(NAME engine.${tag} COMMAND earmark_tests "[${tag}]")
  endforeach()
  add_test(NAME engine.no_alloc_after_create COMMAND earmark_alloc_tests "[alloc]")

  if(APPLE)
    set(EARMARK_ASAN_OPTIONS "ASAN_OPTIONS=detect_leaks=0:abort_on_error=1")  # LSan is unsupported on macOS
  else()
    set(EARMARK_ASAN_OPTIONS "ASAN_OPTIONS=detect_leaks=1:abort_on_error=1")
  endif()
  get_property(EARMARK_ALL_TESTS DIRECTORY PROPERTY TESTS)
  set_tests_properties(${EARMARK_ALL_TESTS} PROPERTIES
    ENVIRONMENT "${EARMARK_ASAN_OPTIONS};UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"
    TIMEOUT 120)
endif()
