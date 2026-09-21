// The demo page: microphone -> earmark.wasm (AudioWorklet) -> headphones.
//
// This file does the parts that must not happen on the audio thread: fetching the
// engine and the weights, asking for the microphone, sending the five-second enrolment
// clip to the local server, and drawing. The processing itself is in worklet.js.
import { SAMPLE_RATE } from "/src/constants.js";

const ENROL_SECONDS = 5;
const RECORD_SECONDS = 12;

const ui = {
  status: document.getElementById("status"),
  controls: document.getElementById("controls"),
  start: document.getElementById("start"),
  fileButton: document.getElementById("file-btn"),
  file: document.getElementById("file"),
  enrol: document.getElementById("enrol"),
  modes: Array.from(document.querySelectorAll(".mode")),
  inBar: document.getElementById("in-bar"),
  outBar: document.getElementById("out-bar"),
  vadBar: document.getElementById("vad-bar"),
  recordPanel: document.getElementById("record-panel"),
  record: document.getElementById("record"),
  compare: document.getElementById("compare"),
  facts: {
    model: document.getElementById("fact-model"),
    params: document.getElementById("fact-params"),
    latency: document.getElementById("fact-latency"),
    rate: document.getElementById("fact-rate"),
  },
};

const state = {
  context: null,
  node: null,
  stream: null,
  mode: "denoise",
  meters: { vad: 0, input: 0, output: 0 },
  enrolled: false,
  recording: false,
  assets: null,
  clip: null, // an AudioBuffer when playing a file instead of the microphone
  clipSource: null,
};

// Exposed so an automated browser test can drive the page without a real microphone.
window.earmark = state;

function setStatus(text, kind = "") {
  ui.status.textContent = text;
  ui.status.className = `status ${kind}`;
}

async function fetchAssets() {
  const [wasm, blob, manifest] = await Promise.all([
    fetch("/engine/earmark.wasm").then(expectOk).then((r) => r.arrayBuffer()),
    fetch("/model/weights.emwb").then(expectOk).then((r) => r.arrayBuffer()),
    fetch("/model/manifest.json").then(expectOk).then((r) => r.arrayBuffer()),
  ]);
  const meta = JSON.parse(new TextDecoder().decode(manifest));
  ui.facts.model.textContent = meta.model?.name ?? "unknown";
  ui.facts.params.textContent = (meta.model?.params ?? 0).toLocaleString();
  return { wasm, blob, manifest, meta };
}

function expectOk(response) {
  if (!response.ok) throw new Error(`${response.url.split("/").pop()}: HTTP ${response.status}`);
  return response;
}

/** Creates the audio graph and hands the engine to the audio thread (once). */
async function ensureEngine() {
  if (state.node) return;
  state.context = new AudioContext({ latencyHint: "interactive" });
  await state.context.audioWorklet.addModule("/src/worklet.js");
  state.node = new AudioWorkletNode(state.context, "earmark-engine", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
  });
  state.node.port.onmessage = (event) => onWorkletMessage(event.data);
  const { wasm, blob, manifest } = state.assets;
  state.node.port.postMessage({ type: "init", wasm, blob, manifest }, [wasm, blob, manifest]);
  state.assets = null; // the buffers moved to the audio thread
  state.node.connect(state.context.destination);
  await state.context.resume();
  ui.facts.rate.textContent = `${state.context.sampleRate.toLocaleString()} Hz`;
}

async function start() {
  ui.start.disabled = true;
  try {
    setStatus("Starting…");
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        // The browser's own cleanup would fight the model, so ask for the raw signal.
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    await ensureEngine();
    stopClip();
    state.context.createMediaStreamSource(state.stream).connect(state.node);
    setStatus("Listening", "good");
  } catch (error) {
    ui.start.disabled = false;
    setStatus(error.message, "bad");
  }
}

// ------------------------------------------------------------------- sound file

async function loadFile(file) {
  try {
    await ensureEngine();
    const bytes = await file.arrayBuffer();
    state.clip = await state.context.decodeAudioData(bytes);
    setStatus(`Playing ${file.name}`, "good");
    playClip();
  } catch (error) {
    setStatus(error.message, "bad");
  }
}

function playClip() {
  if (!state.clip) return;
  stopClip();
  const source = new AudioBufferSourceNode(state.context, { buffer: state.clip, loop: true });
  source.connect(state.node);
  source.start();
  state.clipSource = source;
}

function stopClip() {
  if (!state.clipSource) return;
  state.clipSource.stop();
  state.clipSource.disconnect();
  state.clipSource = null;
}

function onWorkletMessage(msg) {
  if (msg.type === "ready") {
    setStatus("Listening", "good");
    ui.facts.latency.textContent = `${(msg.latencySeconds * 1000).toFixed(1)} ms`;
    ui.enrol.disabled = false;
    ui.recordPanel.hidden = false;
    setMode(state.mode);
    return;
  }
  if (msg.type === "meters") {
    state.meters = msg;
    return;
  }
  if (msg.type === "recording") {
    showComparison(msg);
    return;
  }
  if (msg.type === "error") setStatus(`Engine error: ${msg.error}`, "bad");
}

function setMode(mode) {
  state.mode = mode;
  for (const button of ui.modes) button.classList.toggle("is-on", button.dataset.mode === mode);
  state.node?.port.postMessage({ type: "mode", mode });
}

// ---------------------------------------------------------------------- enrolment

async function enrol() {
  if (!state.context) return;
  ui.enrol.disabled = true;
  if (state.clip) {
    // Enrol from the start of the loaded recording, where the target voice speaks alone.
    const wanted = Math.min(state.clip.length, Math.round(ENROL_SECONDS * state.clip.sampleRate));
    const samples = state.clip.getChannelData(0).slice(0, wanted);
    await sendEnrolment(samples, state.clip.sampleRate);
    return;
  }
  const deadline = Date.now() + ENROL_SECONDS * 1000;
  const chunks = [];
  const source = state.context.createMediaStreamSource(state.stream);
  const tap = new AudioWorkletNode(state.context, "earmark-engine", { numberOfOutputs: 1, outputChannelCount: [1] });
  // The tap has no engine, so it just forwards the microphone; record through it.
  tap.port.onmessage = (event) => {
    if (event.data.type === "recording") chunks.push(event.data);
  };
  // A worklet only runs when it reaches the destination, so route it through silence.
  const silence = new GainNode(state.context, { gain: 0 });
  source.connect(tap);
  tap.connect(silence).connect(state.context.destination);
  tap.port.postMessage({ type: "record", on: true });
  const tick = setInterval(() => {
    const left = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
    setStatus(`Learning… ${left}`);
  }, 200);

  await new Promise((resolve) => setTimeout(resolve, ENROL_SECONDS * 1000));
  clearInterval(tick);
  tap.port.postMessage({ type: "record", on: false });
  await new Promise((resolve) => setTimeout(resolve, 120));
  source.disconnect();
  tap.disconnect();
  silence.disconnect();
  const captured = chunks[0];
  if (!captured) {
    setStatus("Nothing recorded", "bad");
    ui.enrol.disabled = false;
    return;
  }
  await sendEnrolment(captured.raw, captured.sampleRate);
}

/** Resamples to 16 kHz and posts the clip to the local encoder. */
async function sendEnrolment(samples, rate) {
  setStatus("Learning…");
  const audio = rate === SAMPLE_RATE ? samples : await resample(samples, rate, SAMPLE_RATE);
  try {
    const response = await fetch("/enrol", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: audio.buffer,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error ?? `HTTP ${response.status}`);
    state.node.port.postMessage({ type: "embedding", embedding: payload.embedding });
    state.enrolled = true;
    ui.modes.find((b) => b.dataset.mode === "personal").disabled = false;
    setMode("personal");
    setStatus("Personal mode on", "good");
  } catch (error) {
    setStatus(error.message, "bad");
  } finally {
    ui.enrol.disabled = false;
  }
}

async function resample(samples, from, to) {
  const frames = Math.max(1, Math.round((samples.length * to) / from));
  const offline = new OfflineAudioContext(1, frames, to);
  const buffer = new AudioBuffer({ length: samples.length, sampleRate: from, numberOfChannels: 1 });
  buffer.copyToChannel(samples, 0);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}

// ---------------------------------------------------------------------- comparison

function toggleRecording() {
  if (!state.node) return;
  state.recording = !state.recording;
  state.node.port.postMessage({ type: "record", on: state.recording });
  ui.record.textContent = state.recording ? "Stop" : "Record comparison";
  ui.record.classList.toggle("is-on", state.recording);
  if (state.recording) {
    setStatus("Recording…", "good");
    setTimeout(() => {
      if (state.recording) toggleRecording();
    }, RECORD_SECONDS * 1000);
  } else {
    setStatus("Listening", "good");
  }
}

function showComparison({ raw, processed, sampleRate }) {
  state.lastRecording = { raw, processed, sampleRate };
  ui.compare.hidden = false;
  drawWave(document.getElementById("wave-raw"), raw, "#7c8796");
  drawWave(document.getElementById("wave-wet"), processed, "#f5b942");
  document.getElementById("audio-raw").src = URL.createObjectURL(wavBlob(raw, sampleRate));
  document.getElementById("audio-wet").src = URL.createObjectURL(wavBlob(processed, sampleRate));
}

function drawWave(canvas, samples, colour) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
  if (!samples.length) return;
  const step = Math.max(1, Math.floor(samples.length / width));
  ctx.fillStyle = colour;
  for (let x = 0; x < width; x += 1) {
    let peak = 0;
    const start = x * step;
    for (let i = start; i < Math.min(start + step, samples.length); i += 1) {
      const value = Math.abs(samples[i]);
      if (value > peak) peak = value;
    }
    const bar = Math.max(1, peak * height * 0.9);
    ctx.fillRect(x, (height - bar) / 2, 1, bar);
  }
}

/** 16-bit PCM WAV of float samples, so the clips can be played and saved. */
function wavBlob(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const ascii = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

// ---------------------------------------------------------------------- meters

function paint() {
  const { vad, input, output } = state.meters;
  ui.inBar.style.width = `${Math.min(100, input * 140)}%`;
  ui.outBar.style.width = `${Math.min(100, output * 140)}%`;
  ui.vadBar.style.width = `${Math.min(100, vad * 100)}%`;
  requestAnimationFrame(paint);
}

// ---------------------------------------------------------------------- boot

ui.start.addEventListener("click", start);
ui.fileButton.addEventListener("click", () => ui.file.click());
ui.file.addEventListener("change", () => {
  const [file] = ui.file.files ?? [];
  if (file) loadFile(file);
});
ui.enrol.addEventListener("click", enrol);
ui.record.addEventListener("click", toggleRecording);
for (const button of ui.modes) button.addEventListener("click", () => setMode(button.dataset.mode));

try {
  state.assets = await fetchAssets();
  ui.controls.hidden = false;
  setStatus("Ready");
  requestAnimationFrame(paint);
  if (new URLSearchParams(location.search).has("autostart")) await start();
} catch (error) {
  setStatus(error.message, "bad");
}
