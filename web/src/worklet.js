// The engine on the audio thread: earmark.wasm inside an AudioWorkletProcessor.
//
// The processor owns one em_engine for the context's sample rate and feeds it every
// render quantum (128 frames). It always runs the model, whatever the mode, so the
// stream state stays warm and switching modes is instant; `mode` only decides which
// signal reaches the output:
//
//   off       the microphone, untouched (the engine still runs, so A/B stays aligned)
//   denoise   the model with its learned NULL embedding
//   personal  the model with the enrolled embedding (falls back to denoise until then)
//
// Messages in:  {type:"init", wasm, blob, manifest} | {type:"mode", mode}
//               {type:"embedding", embedding} | {type:"record", on}
// Messages out: {type:"ready"|"error"|"meters"|"recording", ...}
const EM_OK = 0;
const METER_INTERVAL_QUANTA = 8; // ~21 ms at 48 kHz: smooth on screen, cheap here
const MAX_RECORD_SECONDS = 30;

class EarmarkProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.engine = 0;
    this.mode = "denoise";
    this.hasEmbedding = false;
    this.quanta = 0;
    this.vad = 0;
    this.inPeak = 0;
    this.outPeak = 0;
    this.recording = false;
    this.rawChunks = [];
    this.wetChunks = [];
    this.recordedFrames = 0;
    this.port.onmessage = (event) => {
      this.handle(event.data).catch((error) => {
        this.port.postMessage({ type: "error", error: String(error && error.message ? error.message : error) });
      });
    };
  }

  async handle(msg) {
    if (msg.type === "init") return this.init(msg);
    if (msg.type === "mode") return this.setMode(msg.mode);
    if (msg.type === "embedding") return this.setEmbedding(msg.embedding);
    if (msg.type === "record") return this.setRecording(msg.on);
    return undefined;
  }

  // ------------------------------------------------------------------ set-up

  async init({ wasm, blob, manifest }) {
    const module = await WebAssembly.compile(wasm);
    const imports = {};
    for (const imp of WebAssembly.Module.imports(module)) {
      imports[imp.module] ??= {};
      // The engine never writes, exits or aborts on the audio path; stub the WASI surface.
      imports[imp.module][imp.name] = () => 0;
    }
    const instance = await WebAssembly.instantiate(module, imports);
    this.ex = instance.exports;
    this.memory = this.ex.memory;
    if (this.ex.em_abi_version() !== 2) throw new Error(`engine ABI ${this.ex.em_abi_version()}, expected 2`);

    const blobPtr = this.copyIn(new Uint8Array(blob));
    const manifestPtr = this.copyIn(new Uint8Array(manifest));
    const statusPtr = this.ex.malloc(4);
    this.engine = this.ex.em_create(blobPtr, blob.byteLength, manifestPtr, manifest.byteLength, sampleRate, statusPtr);
    const status = new Int32Array(this.memory.buffer, statusPtr, 1)[0];
    this.ex.free(blobPtr);
    this.ex.free(manifestPtr);
    this.ex.free(statusPtr);
    if (!this.engine || status !== EM_OK) throw new Error(`em_create failed with status ${status}`);
    if (this.ex.em_has_network(this.engine) !== 1) throw new Error("those weights contain no network");

    this.inPtr = this.ex.malloc(4 * 128);
    this.outPtr = this.ex.malloc(4 * 128);
    this.vadPtr = this.ex.malloc(4);
    this.embeddingPtr = this.ex.malloc(4 * 256);
    this.capacity = 128;
    this.ex.em_set_embedding(this.engine, 0);
    this.port.postMessage({
      type: "ready",
      latencySamples: this.ex.em_latency_samples(this.engine),
      latencySeconds: this.ex.em_latency_seconds(this.engine),
      arenaBytes: this.ex.em_arena_bytes(this.engine),
      sampleRate,
    });
  }

  copyIn(bytes) {
    const ptr = this.ex.malloc(bytes.length);
    if (!ptr) throw new Error(`malloc(${bytes.length}) failed`);
    new Uint8Array(this.memory.buffer, ptr, bytes.length).set(bytes);
    return ptr;
  }

  setMode(mode) {
    this.mode = mode;
    if (!this.engine) return;
    const personal = mode === "personal" && this.hasEmbedding;
    this.ex.em_set_embedding(this.engine, personal ? this.embeddingPtr : 0);
  }

  setEmbedding(embedding) {
    if (!this.engine) return;
    if (embedding.length !== 256) throw new Error(`embedding has ${embedding.length} values, expected 256`);
    new Float32Array(this.memory.buffer, this.embeddingPtr, 256).set(embedding);
    this.hasEmbedding = true;
    this.setMode(this.mode);
  }

  setRecording(on) {
    if (on) {
      this.rawChunks = [];
      this.wetChunks = [];
      this.recordedFrames = 0;
      this.recording = true;
      return;
    }
    this.recording = false;
    const raw = concat(this.rawChunks, this.recordedFrames);
    const processed = concat(this.wetChunks, this.recordedFrames);
    this.rawChunks = [];
    this.wetChunks = [];
    this.port.postMessage({ type: "recording", raw, processed, sampleRate }, [raw.buffer, processed.buffer]);
  }

  /** Keeps both signals while recording, up to MAX_RECORD_SECONDS. */
  capture(raw, wet, frames) {
    if (!this.recording || this.recordedFrames >= MAX_RECORD_SECONDS * sampleRate) return;
    this.rawChunks.push(new Float32Array(raw));
    this.wetChunks.push(new Float32Array(wet));
    this.recordedFrames += frames;
  }

  // ------------------------------------------------------------------ audio

  process(inputs, outputs) {
    const input = inputs[0] && inputs[0][0];
    const output = outputs[0][0];
    if (!input) {
      output.fill(0);
      return true;
    }
    if (!this.engine) {
      // Still starting up, or this is a plain tap (enrolment): pass the microphone through.
      output.set(input);
      this.capture(input, input, input.length);
      return true;
    }
    const frames = input.length;
    if (frames > this.capacity) return true; // never happens: the quantum is fixed at 128

    new Float32Array(this.memory.buffer, this.inPtr, frames).set(input);
    const status = this.ex.em_process(this.engine, this.inPtr, frames, this.outPtr, this.vadPtr);
    if (status !== EM_OK) {
      output.set(input);
      return true;
    }
    const processed = new Float32Array(this.memory.buffer, this.outPtr, frames);
    output.set(this.mode === "off" ? input : processed);
    this.vad = new Float32Array(this.memory.buffer, this.vadPtr, 1)[0];

    for (let i = 0; i < frames; i += 1) {
      const dry = Math.abs(input[i]);
      const wet = Math.abs(output[i]);
      if (dry > this.inPeak) this.inPeak = dry;
      if (wet > this.outPeak) this.outPeak = wet;
    }
    this.capture(input, processed, frames);
    this.quanta += 1;
    if (this.quanta % METER_INTERVAL_QUANTA === 0) {
      this.port.postMessage({ type: "meters", vad: this.vad, input: this.inPeak, output: this.outPeak });
      this.inPeak = 0;
      this.outPeak = 0;
    }
    return true;
  }
}

function concat(chunks, frames) {
  const out = new Float32Array(frames);
  let offset = 0;
  for (const chunk of chunks) {
    if (offset + chunk.length > frames) break;
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

registerProcessor("earmark-engine", EarmarkProcessor);
