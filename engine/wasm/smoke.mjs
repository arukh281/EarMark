#!/usr/bin/env node
// Smoke test for the CI-built earmark.wasm (run by `make wasm` and the CI wasm job).
//
//   node engine/wasm/smoke.mjs [path/to/earmark.wasm]     (default: engine/build-wasm)
//
// It loads the module the way the page will: no Emscripten JS glue, every import stubbed
// (fd_write is implemented so a stray printf cannot spin). It then checks:
//   * every em_* export plus malloc, free and memory is present, and the ABI version and
//     contract hash match include/earmark.h and web/src/constants.js;
//   * at 16 kHz, em_process returns the input delayed by em_latency_samples() within 1e-5
//     (the network is not wired yet, so the signal path is an exact identity);
//   * at 48 kHz, a 997 Hz tone comes back after em_latency_seconds() with SNR above 90 dB,
//     the same bar as the native Catch2 test, and with zero FIFO under- or overruns.
// Exit status: 0 pass, 1 a check failed.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const engineDir = resolve(here, "..");
const repoRoot = resolve(engineDir, "..");
const wasmPath = resolve(process.argv[2] ?? resolve(engineDir, "build-wasm", "earmark.wasm"));
const contract = await import(pathToFileURL(resolve(repoRoot, "web", "src", "constants.js")).href);
const goldens = resolve(engineDir, "tests", "goldens");

const EM_OK = 0;
const EM_ABI_VERSION = 1;
const failures = [];
const report = { wasm: wasmPath };

function check(condition, message) {
  if (!condition) failures.push(message);
  return condition;
}

// ------------------------------------------------------------------ instantiate

const module = new WebAssembly.Module(readFileSync(wasmPath));
let memory = null;

function fdWrite(fd, iovs, iovsLen, nwrittenPtr) {
  // WASI fd_write: gather the iovecs, echo them to stderr and report every byte written.
  const view = new DataView(memory.buffer);
  const chunks = [];
  let total = 0;
  for (let i = 0; i < iovsLen; i += 1) {
    const ptr = view.getUint32(iovs + 8 * i, true);
    const len = view.getUint32(iovs + 8 * i + 4, true);
    chunks.push(Buffer.from(new Uint8Array(memory.buffer, ptr, len)));
    total += len;
  }
  process.stderr.write(`[wasm fd ${fd}] ${Buffer.concat(chunks).toString("utf8")}`);
  view.setUint32(nwrittenPtr, total, true);
  return 0;
}

const imports = {};
report.imports = [];
for (const imp of WebAssembly.Module.imports(module)) {
  report.imports.push(`${imp.module}.${imp.name} (${imp.kind})`);
  if (imp.kind !== "function") {
    failures.push(`unexpected ${imp.kind} import ${imp.module}.${imp.name}: the module must own its memory`);
    continue;
  }
  imports[imp.module] ??= {};
  if (imp.name === "fd_write") {
    imports[imp.module][imp.name] = fdWrite;
  } else if (/abort/.test(imp.name) || imp.name === "proc_exit" || imp.name === "exit") {
    imports[imp.module][imp.name] = (...args) => {
      throw new Error(`wasm called ${imp.module}.${imp.name}(${args.join(", ")})`);
    };
  } else {
    imports[imp.module][imp.name] = () => 0;
  }
}
if (failures.length > 0) finish();

const instance = new WebAssembly.Instance(module, imports);
const ex = instance.exports;
memory = ex.memory;
check(memory instanceof WebAssembly.Memory, "the module does not export its memory as `memory`");
if (typeof ex._initialize === "function") ex._initialize(); // reactor: run static constructors

const required = [
  "em_abi_version", "em_contract_hash", "em_status_string", "em_build_info",
  "em_create", "em_destroy", "em_set_embedding", "em_process", "em_process_hop_16k", "em_reset",
  "em_device_rate", "em_latency_samples", "em_latency_seconds", "em_arena_bytes", "em_state_bytes",
  "em_xruns", "malloc", "free",
];
const missing = required.filter((name) => typeof ex[name] !== "function");
if (!check(missing.length === 0, `missing exports: ${missing.join(", ")}`)) finish();

report.memory_bytes = memory.buffer.byteLength;
try {
  memory.grow(1);
  report.memory_growable = true;
} catch {
  report.memory_growable = false; // expected: fixed INITIAL_MEMORY, no ALLOW_MEMORY_GROWTH
}

// ------------------------------------------------------------------ helpers

const u32 = (value) => Number(value) >>> 0;

function cString(ptr) {
  const bytes = new Uint8Array(memory.buffer);
  let end = u32(ptr);
  while (end < bytes.length && bytes[end] !== 0) end += 1;
  return new TextDecoder().decode(bytes.subarray(u32(ptr), end));
}

function alloc(bytes) {
  const ptr = u32(ex.malloc(bytes));
  if (ptr === 0) throw new Error(`malloc(${bytes}) failed`);
  return ptr;
}

function copyIn(bytes) {
  const ptr = alloc(bytes.length + 1);
  const view = new Uint8Array(memory.buffer, ptr, bytes.length + 1);
  view.set(bytes);
  view[bytes.length] = 0;
  return ptr;
}

function statusName(status) {
  return `${status} (${cString(ex.em_status_string(status))})`;
}

const blob = new Uint8Array(readFileSync(resolve(goldens, "weights_small.emwb")));
const manifest = new Uint8Array(readFileSync(resolve(goldens, "weights_small.json")));
const blobPtr = copyIn(blob);
const manifestPtr = copyIn(manifest);
const statusPtr = alloc(4);

function create(rate) {
  const handle = u32(ex.em_create(blobPtr, blob.length, manifestPtr, manifest.length, rate, statusPtr));
  const status = new Int32Array(memory.buffer, statusPtr, 1)[0];
  if (!check(handle !== 0 && status === EM_OK, `em_create at ${rate} Hz failed: ${statusName(status)}`)) {
    finish();
  }
  const embedded = ex.em_set_embedding(handle, 0); // NULL: the learned NULL embedding
  check(embedded === EM_OK, `em_set_embedding(NULL) returned ${statusName(embedded)}`);
  return handle;
}

// Streams `x` through em_process in the given cyclic block sizes; returns [y, maxVad].
function stream(handle, x, blocks) {
  const cap = Math.max(...blocks);
  const inPtr = alloc(4 * cap);
  const outPtr = alloc(4 * cap);
  const vadPtr = alloc(4);
  const y = new Float32Array(x.length);
  let vadMax = 0;
  for (let pos = 0, k = 0; pos < x.length; k += 1) {
    const n = Math.min(blocks[k % blocks.length], x.length - pos);
    new Float32Array(memory.buffer, inPtr, n).set(x.subarray(pos, pos + n));
    const status = ex.em_process(handle, inPtr, n, outPtr, vadPtr);
    if (!check(status === EM_OK, `em_process returned ${statusName(status)}`)) break;
    y.set(new Float32Array(memory.buffer, outPtr, n), pos);
    const vad = new Float32Array(memory.buffer, vadPtr, 1)[0];
    check(Number.isFinite(vad) && vad >= 0 && vad <= 1, `vad_out ${vad} outside [0, 1]`);
    vadMax = Math.max(vadMax, vad);
    pos += n;
  }
  for (const ptr of [inPtr, outPtr, vadPtr]) ex.free(ptr);
  return [y, vadMax];
}

// ------------------------------------------------------------------ checks

report.abi_version = u32(ex.em_abi_version());
check(report.abi_version === EM_ABI_VERSION, `em_abi_version ${report.abi_version} != ${EM_ABI_VERSION}`);
report.contract_hash = cString(ex.em_contract_hash());
check(
  report.contract_hash === contract.CONTRACT_HASH,
  `em_contract_hash ${report.contract_hash} != web/src/constants.js ${contract.CONTRACT_HASH}`,
);
report.build_info = cString(ex.em_build_info());

{
  // 16 kHz: exact identity after the reported latency (white noise from a fixed LCG).
  const rate = contract.SAMPLE_RATE;
  const handle = create(rate);
  const latency = ex.em_latency_samples(handle);
  const x = new Float32Array(rate);
  let seed = 12345;
  for (let i = 0; i < x.length; i += 1) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    x[i] = seed / 4294967296 - 0.5;
  }
  const [y] = stream(handle, x, [160, 37, 480, 1]);
  let maxErr = 0;
  for (let i = latency; i < x.length; i += 1) maxErr = Math.max(maxErr, Math.abs(y[i] - x[i - latency]));
  report.identity_16k = { latency_samples: latency, max_abs_error: maxErr };
  check(latency === 2 * contract.HOP_LENGTH - 1, `16 kHz latency ${latency} != ${2 * contract.HOP_LENGTH - 1}`);
  check(maxErr < 1e-5, `16 kHz output differs from the delayed input by ${maxErr}`);
  check(Number(ex.em_xruns(handle)) === 0, "16 kHz stream reported FIFO xruns");
  ex.em_destroy(handle);
}

{
  // 48 kHz: the resampled tone test from engine/tests/test_engine.cpp.
  const rate = 48000;
  const f = 997;
  const handle = create(rate);
  check(ex.em_device_rate(handle) === rate, "em_device_rate does not echo the requested rate");
  const n = rate / 2;
  const x = new Float32Array(n);
  for (let i = 0; i < n; i += 1) x[i] = 0.5 * Math.sin((2 * Math.PI * f * i) / rate);
  const [y, vadMax] = stream(handle, x, [128, 480, 1, 441, 1024]);
  const latency = ex.em_latency_samples(handle);
  const delay = ex.em_latency_seconds(handle) * rate;
  let sig = 0;
  let err = 0;
  for (let i = n / 4; i < n; i += 1) {
    const want = 0.5 * Math.sin((2 * Math.PI * f * (i - delay)) / rate);
    sig += want * want;
    err += (y[i] - want) * (y[i] - want);
  }
  const snr = 10 * Math.log10(sig / err);
  const xruns = Number(ex.em_xruns(handle));
  report.tone_48k = { latency_samples: latency, exact_delay: delay, snr_db: snr, xruns, vad_max: vadMax };
  report.arena_bytes = u32(ex.em_arena_bytes(handle));
  report.state_bytes = u32(ex.em_state_bytes(handle));
  check(snr > 90, `48 kHz tone SNR ${snr.toFixed(1)} dB <= 90 dB`);
  check(Math.abs(delay - latency) <= 0.5, `em_latency_samples ${latency} is not the rounded delay ${delay}`);
  check(xruns === 0, `48 kHz stream reported ${xruns} FIFO xruns`);
  check(ex.em_reset(handle) === EM_OK, "em_reset failed");
  ex.em_destroy(handle);
}

finish();

function finish() {
  console.log(JSON.stringify(report, null, 2));
  if (failures.length > 0) {
    for (const message of failures) console.error(`smoke: FAIL ${message}`);
    process.exit(1);
  }
  console.log("smoke: PASS earmark.wasm");
  process.exit(0);
}
