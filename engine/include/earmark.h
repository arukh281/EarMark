/*
 * Earmark streaming engine: C ABI (C99 / C++17, native and WASM).
 *
 * One em_engine processes one mono stream. Every allocation happens inside em_create:
 * the engine reserves one fixed arena (plus the FFT plan) and the audio calls
 * (em_process, em_process_hop_16k, em_set_embedding, em_reset) never allocate, lock or
 * throw. The engine is single-threaded: do not call it on one handle from two threads
 * at once.
 *
 * Per 16 kHz hop, em_process runs the whole model: WOLA analysis, ERB and low-band
 * features, the network (encoders, FiLM, GRU body, VAD / gain / deep-filter heads), the
 * ERB gains, the deep filter and synthesis. It matches EarmarkNet.step in PyTorch (see
 * engine/tests/test_network.cpp). Only GRU bodies are supported (M, M-256, S-GRU).
 *
 * A blob without any network tensors (a signal-path test blob) still loads: the gains
 * are then fixed at 1, the output is the input delayed by em_latency_samples(), and
 * *vad_out is 0. em_has_network() tells the two apart.
 *
 * Weights come from a .emwb blob and its JSON manifest, both written by
 * earmark.export.blob. em_create refuses a blob or manifest whose contract hash
 * differs from EARMARK_CONTRACT_HASH (em_create then reports EM_ERR_CONTRACT).
 */
#ifndef EARMARK_H_
#define EARMARK_H_

#include <stddef.h>
#include <stdint.h>

#include "earmark_constants.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define EM_API __attribute__((visibility("default")))
#else
#define EM_API
#endif

/** Bumped whenever a signature or a documented behaviour of this header changes. */
#define EM_ABI_VERSION 2

/** Status codes (em_status). Zero is success; errors are negative. */
#define EM_OK 0
#define EM_ERR_ARGUMENT (-1)       /* null handle/pointer or a value out of range */
#define EM_ERR_BLOB (-2)           /* blob malformed, truncated or corrupt (CRC) */
#define EM_ERR_CONTRACT (-3)       /* blob or manifest was exported under another contract */
#define EM_ERR_MANIFEST (-4)       /* manifest malformed or does not describe the blob */
#define EM_ERR_MISSING_TENSOR (-5) /* a required tensor is absent or has the wrong shape */
#define EM_ERR_RATE (-6)           /* device sample rate unsupported */
#define EM_ERR_NO_MEMORY (-7)
#define EM_ERR_FFT (-8)            /* FFT plan could not be built allocation-free */

/** Device sample rates em_create accepts (inclusive). */
#define EM_MIN_DEVICE_RATE 8000
#define EM_MAX_DEVICE_RATE 192000

typedef int32_t em_status;
typedef struct em_engine em_engine;

/** EM_ABI_VERSION of the compiled library. */
EM_API uint32_t em_abi_version(void);

/** The contract hash compiled into the library (EARMARK_CONTRACT_HASH). */
EM_API const char* em_contract_hash(void);

/** Static description of a status code. */
EM_API const char* em_status_string(em_status status);

/** Static one-line build description (ABI, contract, matvec backend, skeleton flag). */
EM_API const char* em_build_info(void);

/**
 * Creates an engine for a device running at `device_rate` Hz.
 *
 * blob / blob_bytes: the .emwb weight blob. It is copied into the engine's arena, so it
 *   may be freed as soon as em_create returns and need not be aligned.
 * manifest / manifest_bytes: the blob's JSON manifest, or NULL to rely on the blob
 *   header alone. manifest_bytes == 0 means `manifest` is NUL-terminated. When given,
 *   its contract hash, blob size and CRCs must match the blob.
 * status: receives EM_OK or the reason for failure (may be NULL).
 *
 * The blob must contain const.erb_norm_init [ERB_BANDS], const.spec_norm_init [DF_BINS]
 * and conditioner.null_embedding [EMBEDDING_DIM], all float32. If it has any network
 * tensor it must have all of them with consistent shapes and a GRU body; otherwise
 * em_create reports EM_ERR_MISSING_TENSOR.
 * Returns NULL on failure.
 */
EM_API em_engine* em_create(const uint8_t* blob, size_t blob_bytes, const char* manifest, size_t manifest_bytes,
                            int32_t device_rate, em_status* status);

/** Frees the engine (NULL is ignored). */
EM_API void em_destroy(em_engine* engine);

/**
 * Sets the target speaker's enrolment embedding (float[EARMARK_EMBEDDING_DIM]), or
 * the learned NULL embedding when `embedding` is NULL (Denoise mode). Values must be
 * finite. The embedding is kept across em_reset.
 */
EM_API em_status em_set_embedding(em_engine* engine, const float* embedding);

/**
 * Processes `n` device-rate samples from `in` into exactly `n` samples in `out`
 * (`in` and `out` may be the same buffer). Any block size works, including 0 and sizes
 * that are not a multiple of the hop. `vad_out` (may be NULL) receives the personal-VAD
 * probability of the most recent hop.
 */
EM_API em_status em_process(em_engine* engine, const float* in, size_t n, float* out, float* vad_out);

/**
 * Processes one 16 kHz hop (EARMARK_HOP_LENGTH samples) directly, bypassing the
 * resamplers and FIFOs; the output lags the input by exactly one hop. It shares the
 * model state with em_process, so use one path or the other per stream (or em_reset
 * when switching).
 */
EM_API em_status em_process_hop_16k(em_engine* engine, const float* in, float* out, float* vad_out);

/** Clears all stream state (buffers, normalisation, resamplers, FIFOs); keeps the embedding. */
EM_API em_status em_reset(em_engine* engine);

/** The device rate the engine was created for (0 for NULL). */
EM_API int32_t em_device_rate(const em_engine* engine);
/** 1 when the blob held a network (em_process enhances), 0 for a signal-path-only blob or NULL. */
EM_API int32_t em_has_network(const em_engine* engine);

/**
 * End-to-end latency of em_process in device samples, rounded to the nearest sample:
 * the FIFO priming, one hop of WOLA delay and both resamplers' group delays.
 */
EM_API int32_t em_latency_samples(const em_engine* engine);

/**
 * The same latency in seconds, unrounded. The pipeline is linear phase end to end, so
 * a band-limited input x(t) comes out as x(t - em_latency_seconds()); use this to
 * align engine output with a reference at non-integer delays (44.1 kHz, 22.05 kHz).
 */
EM_API double em_latency_seconds(const em_engine* engine);

/** Bytes reserved for the arena (weights copy, filter taps, buffers and state). */
EM_API size_t em_arena_bytes(const em_engine* engine);

/** Bytes of mutable per-stream state (the part em_reset clears, plus the embedding). */
EM_API size_t em_state_bytes(const em_engine* engine);

/** FIFO underruns or overruns since em_create; always 0 unless the engine has a bug. */
EM_API uint64_t em_xruns(const em_engine* engine);

#ifdef __cplusplus
}
#endif

#endif /* EARMARK_H_ */
