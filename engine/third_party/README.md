# Vendored third-party code

The engine has no package dependencies. These files are copied in unchanged from
their official sources, except for the one documented patch to pocketfft. Their
licences are kept next to them.

| File | Upstream | Version | Licence |
| --- | --- | --- | --- |
| `pocketfft_hdronly.hpp` | https://github.com/mreineck/pocketfft (`cpp` branch, file `pocketfft_hdronly.h`) | commit `c90e55b3d529f8efa40ed01a20de22405f45fc65` (2026-06-30), plus `patches/pocketfft_noalloc.patch` | BSD-3-Clause, `POCKETFFT_LICENSE.md` |
| `catch2/catch_amalgamated.hpp`, `catch2/catch_amalgamated.cpp` | https://github.com/catchorg/Catch2 (`extras/`) | tag `v3.16.0` (2026-08-25) | BSL-1.0, `catch2/LICENSE.txt` |

Catch2 is used only by the test executables. It is never linked into the engine
library or the WASM module.

## SHA-256

| File | SHA-256 |
| --- | --- |
| upstream `pocketfft_hdronly.h` at the commit above | `3e9a05318d8e3b1446bda1c4617e6a103cdd23599ae0a776a92a6e8800e92fdc` |
| `pocketfft_hdronly.hpp` (patched, as committed) | `6e46491f7ef7ac502ec3211d8e53a5c0e51a80a54df728caea64099b59b393c3` |
| `catch2/catch_amalgamated.hpp` | `d4cc143ea76ae212204363922d8adf376d66a1fda5a33ac73f93a7d1c119f4e0` |
| `catch2/catch_amalgamated.cpp` | `1fe7f10334e0ae5494419cfa84c270f15235eada0fc02bdb493d1def537601f3` |

## The pocketfft patch

Upstream `rfftp::exec` allocates a scratch array (`arr<T> ch(length)`) on every call.
The engine forbids allocation after `em_create`, and the malloc-counter test would
catch that call. `patches/pocketfft_noalloc.patch` makes two additive changes and
alters no arithmetic:

- `rfftp::exec` gains an overload that takes caller-owned scratch. The original
  signature allocates and then forwards to it, so upstream behaviour is unchanged.
- `pocketfft_r` gains `supports_noalloc()` and `exec_noalloc(c, fct, fwd, buf)`.
  These return false for Bluestein plans, which still allocate. The engine's only
  size, N_FFT = 320 = 4·4·4·5, always gets a packed plan; `RealFft::init` checks this
  and fails otherwise.

To re-vendor, download the upstream file at a new commit, rename it to
`pocketfft_hdronly.hpp`, apply the patch with
`patch -p1 -d engine/third_party < engine/third_party/patches/pocketfft_noalloc.patch`,
then update the hashes above.

The engine compiles pocketfft with `POCKETFFT_NO_MULTITHREADING` (no threads, which
WASM needs) and `POCKETFFT_CACHE_SIZE=0` (no global plan cache). Plans are built once
in `RealFft::init`, on the heap, during `em_create`.
