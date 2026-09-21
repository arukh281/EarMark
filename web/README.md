# Earmark demo page

Microphone in, Earmark out, in the browser: the same C++ engine as everywhere else,
compiled to WebAssembly and run in an AudioWorklet at the device's sample rate.

```sh
python -m earmark.export.blob checkpoint <ckpt.pt> --config M --out results/export
gh run download --name earmark-wasm -D engine/build-wasm      # or engine/wasm/build.sh
make demo MODEL=results/export/earmark-m-<name>.emwb
```

Then open <http://127.0.0.1:8000> and **wear headphones**: without them the speakers feed
back into the microphone and you hear the model chasing its own output.

## What you can do on the page

| Control | What it does |
| --- | --- |
| Start microphone | asks for the raw microphone (the browser's own noise suppression and echo cancellation are switched off, because they would fight the model) |
| Use a sound file | plays a recording through the engine instead, for a repeatable demo |
| Learn the voice | takes five seconds, sends it to the local server, and switches to Personal mode |
| Off / Denoise / Personal | the raw microphone, the model with its learned NULL embedding, or the model conditioned on the enrolled voice |
| Record a comparison | keeps what the microphone heard and what Earmark produced, side by side, to play back and save |

The meters show the microphone level, the output level and the model's personal-VAD
probability ("you are speaking"), which is what the barge-in gate would use.

## How it fits together

| Piece | Where |
| --- | --- |
| `index.html`, `src/style.css` | the page |
| `src/app.js` | fetching the engine and weights, the microphone, enrolment, drawing, WAV export |
| `src/worklet.js` | the AudioWorklet: `em_create` once, then `em_process` every 128-frame quantum |
| `src/constants.js` | generated from `contract/signal.yaml`; never edit by hand |
| `python/earmark/demo.py` | the local server: the page, `earmark.wasm`, the weights, and `POST /enrol` |

Enrolment is the one thing the page cannot do on its own: the voice print comes from the
frozen WeSpeaker encoder, so the five-second clip goes to the local server, which runs
that model (with the front end in `earmark.data.fbank`) and returns 256 numbers. The
audio is used for that and dropped. Everything else — the weights, the audio, the
processing — stays in the page.

`python -m earmark.demo` binds to 127.0.0.1 and serves one model from disk. It is a
development server for one listener: do not put it on a network.
