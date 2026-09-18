#!/usr/bin/env bash
# Builds earmark.wasm: the engine's C ABI (include/earmark.h) as a standalone
# WebAssembly module with SIMD128, a fixed memory and no JavaScript glue.
#
# CI only: GitHub Actions installs a pinned Emscripten with setup-emsdk. There is no
# local Emscripten, and the native build (engine/CMakeLists.txt) covers local work.
#
#   engine/wasm/build.sh [out_dir]          (default: engine/build-wasm)
#
# Environment: EARMARK_WASM_MEMORY (bytes, default 64 MiB), EARMARK_WASM_STACK
# (bytes, default 1 MiB).
#
# The module:
# * is built with -sSTANDALONE_WASM --no-entry and exports only the em_* functions
#   plus malloc/free (the page copies the weight blob in through malloc);
# * has a fixed INITIAL_MEMORY and no ALLOW_MEMORY_GROWTH, so the audio worklet's
#   view of memory never detaches;
# * uses no threads, no SharedArrayBuffer, no -sAUDIO_WORKLET and no -sWASM_WORKERS.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine="$(cd "${here}/.." && pwd)"
out="${1:-${engine}/build-wasm}"
memory="${EARMARK_WASM_MEMORY:-67108864}"
stack="${EARMARK_WASM_STACK:-1048576}"

if ! command -v em++ >/dev/null 2>&1; then
  echo "build.sh: em++ not found. The WASM build runs in CI (setup-emsdk); build natively with CMake instead." >&2
  exit 1
fi
em++ --version | head -n 1

mkdir -p "${out}/obj"

common=(
  -std=c++17 -O3 -DNDEBUG -msimd128 -ffp-contract=off
  -DPOCKETFFT_NO_MULTITHREADING -DPOCKETFFT_CACHE_SIZE=0
  -I"${engine}/include" -I"${engine}/src" -isystem "${engine}/third_party"
  -Wall -Wextra -Wpedantic -Wshadow -Wdouble-promotion -Werror
)
sources=(arena engine erb fft gru matvec_f32 network resampler ringbuf state stft weights)

objects=()
for name in "${sources[@]}"; do
  flags=("${common[@]}")
  # fft.cpp keeps Emscripten's default exception mode (throw -> abort) because the
  # vendored pocketfft contains throw statements; every other file forbids them.
  if [[ "${name}" != "fft" ]]; then
    flags+=(-fno-exceptions -fno-rtti)
  fi
  em++ "${flags[@]}" -c "${engine}/src/${name}.cpp" -o "${out}/obj/${name}.o"
  objects+=("${out}/obj/${name}.o")
done

exports=(
  em_abi_version em_contract_hash em_status_string em_build_info
  em_create em_destroy em_set_embedding em_process em_process_hop_16k em_reset
  em_device_rate em_has_network em_latency_samples em_latency_seconds em_arena_bytes em_state_bytes em_xruns
  malloc free
)
export_list="$(printf '_%s,' "${exports[@]}")"
export_list="${export_list%,}"

em++ "${objects[@]}" -O3 -msimd128 -o "${out}/earmark.wasm" \
  -sSTANDALONE_WASM=1 --no-entry \
  -sINITIAL_MEMORY="${memory}" -sALLOW_MEMORY_GROWTH=0 -sSTACK_SIZE="${stack}" \
  -sEXPORTED_FUNCTIONS="${export_list}" \
  -sERROR_ON_UNDEFINED_SYMBOLS=1 -sFILESYSTEM=0 -sASSERTIONS=0

bytes="$(wc -c < "${out}/earmark.wasm" | tr -d ' ')"
echo "wrote ${out}/earmark.wasm (${bytes} bytes)"

# Print the module's imports and exports; the loader must satisfy every import.
if command -v node >/dev/null 2>&1; then
  node -e '
    const fs = require("fs");
    const mod = new WebAssembly.Module(fs.readFileSync(process.argv[1]));
    const imports = WebAssembly.Module.imports(mod).map((i) => `${i.module}.${i.name} (${i.kind})`);
    const exports = WebAssembly.Module.exports(mod).map((e) => e.name);
    console.log(JSON.stringify({ imports, exports }, null, 2));
  ' "${out}/earmark.wasm"
fi
