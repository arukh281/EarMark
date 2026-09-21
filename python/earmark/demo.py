"""Local demo server: the browser page, the WASM engine, a model blob and enrolment.

    python -m earmark.demo --model results/export/earmark-m-v1.emwb

It serves ``web/`` plus three things the page fetches:

* ``/engine/earmark.wasm`` - the standalone WASM build (``engine/wasm/build.sh``, or the
  ``earmark-wasm`` artefact from CI: ``gh run download --name earmark-wasm``);
* ``/model/weights.emwb`` and ``/model/manifest.json`` - the blob and manifest given by
  ``--model`` (export one with ``python -m earmark.export.blob checkpoint ...``);
* ``POST /enrol`` - 16 kHz float32 mono audio in, the speaker embedding out, computed by
  the same frozen WeSpeaker encoder the model was trained against.

Nothing leaves the machine: it binds to 127.0.0.1, the weights stay where they are on
disk, and enrolment audio is used to compute one embedding and then dropped. It is a
development server for one listener, not something to expose to a network.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import cached_property
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from earmark import constants as C

__all__ = ["DemoConfig", "build_handler", "main", "serve"]

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "web"
DEFAULT_WASM = REPO_ROOT / "engine" / "build-wasm" / "earmark.wasm"
#: Enrolment audio bounds (16 kHz mono float32): long enough to be a voice print, short
#: enough that a stray upload cannot fill memory.
MIN_ENROL_SECONDS = 1.0
MAX_ENROL_SECONDS = 30.0
#: Below this RMS the clip is silence, and the embedding would be noise.
MIN_ENROL_RMS = 1e-3

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".svg": "image/svg+xml",
    ".emwb": "application/octet-stream",
}


@dataclass(frozen=True)
class DemoConfig:
    """What the server serves. ``model`` is the ``.emwb`` blob; its manifest sits next to it."""

    model: Path
    wasm: Path = DEFAULT_WASM
    web_root: Path = WEB_ROOT
    host: str = "127.0.0.1"
    port: int = 8000
    enrol: bool = True

    @property
    def manifest(self) -> Path:
        return self.model.with_suffix(".json")

    def check(self) -> None:
        """Fail early, with the command that produces each missing file."""
        if not self.model.is_file():
            raise FileNotFoundError(
                f"no model blob at {self.model}; export one with "
                "`python -m earmark.export.blob checkpoint <ckpt.pt> --config M`"
            )
        if not self.manifest.is_file():
            raise FileNotFoundError(f"no manifest at {self.manifest} (it is written beside the blob)")
        if not self.wasm.is_file():
            raise FileNotFoundError(
                f"no WASM engine at {self.wasm}; build it with engine/wasm/build.sh or fetch CI's "
                "artefact: `gh run download --name earmark-wasm -D engine/build-wasm`"
            )
        if not (self.web_root / "index.html").is_file():
            raise FileNotFoundError(f"no page at {self.web_root / 'index.html'}")


class Enroller:
    """The frozen speaker encoder, loaded on first use."""

    def __init__(self) -> None:
        self._encoder: Any = None

    @cached_property
    def encoder(self) -> Any:
        from earmark.data.embeddings import WeSpeakerOnnxEncoder

        return WeSpeakerOnnxEncoder.from_hub()

    def embed(self, audio: np.ndarray) -> dict[str, Any]:
        """Embedding of 16 kHz mono float32 ``audio``; raises ValueError if it is unusable."""
        import torch

        seconds = audio.size / C.SAMPLE_RATE
        if not MIN_ENROL_SECONDS <= seconds <= MAX_ENROL_SECONDS:
            raise ValueError(f"enrolment must be {MIN_ENROL_SECONDS}-{MAX_ENROL_SECONDS} s, got {seconds:.2f} s")
        if not np.isfinite(audio).all():
            raise ValueError("enrolment audio contains NaN or infinity")
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
        if rms < MIN_ENROL_RMS:
            raise ValueError("that recording is silent; speak while enrolling")
        embedding = self.encoder.embed(torch.from_numpy(audio)[None])[0]
        return {
            "embedding": [float(v) for v in embedding],
            "encoder": self.encoder.name,
            "seconds": round(seconds, 3),
            "rms": round(rms, 6),
        }


def build_handler(config: DemoConfig, enroller: Enroller | None = None) -> type[BaseHTTPRequestHandler]:
    """The request handler class for ``config`` (a class, as http.server expects)."""
    enrol = enroller if enroller is not None else Enroller()

    class Handler(BaseHTTPRequestHandler):
        server_version = "earmark-demo"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002 - http.server's name
            sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

        # -------------------------------------------------------------- helpers

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # The page uses an AudioWorklet and fetches the wasm; no caching while developing.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _send_file(self, path: Path) -> None:
            try:
                body = path.read_bytes()
            except OSError:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"{path.name} is missing"})
                return
            self._send(HTTPStatus.OK, body, CONTENT_TYPES.get(path.suffix, "application/octet-stream"))

        def _static(self, route: str) -> Path | None:
            """The file for a URL path, or None when it escapes the web root."""
            relative = "index.html" if route == "/" else route.lstrip("/")
            candidate = (config.web_root / relative).resolve()
            root = config.web_root.resolve()
            if candidate == root or root in candidate.parents:
                return candidate
            return None

        # ---------------------------------------------------------------- routes

        def do_HEAD(self) -> None:  # noqa: N802 - http.server's name
            self.do_GET()

        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            route = self.path.split("?", 1)[0]
            if route == "/health":
                self._send_json(HTTPStatus.OK, {"ok": True, "model": config.model.name, "enrol": config.enrol})
                return
            if route == "/engine/earmark.wasm":
                self._send_file(config.wasm)
                return
            if route == "/model/weights.emwb":
                self._send_file(config.model)
                return
            if route == "/model/manifest.json":
                self._send_file(config.manifest)
                return
            path = self._static(route)
            if path is None or not path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"no route {route}"})
                return
            self._send_file(path)

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            if self.path.split("?", 1)[0] != "/enrol":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})
                return
            if not config.enrol:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "enrolment is disabled (--no-enrol)"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            limit = int(MAX_ENROL_SECONDS * C.SAMPLE_RATE) * 4
            if length > limit:
                # Refuse without reading it: hang up rather than take the upload.
                self.close_connection = True
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": f"body must be at most {limit} bytes, got {length}"}
                )
                return
            if length <= 0:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"body must be 1..{limit} bytes, got {length}"})
                return
            raw = self.rfile.read(length)
            if len(raw) % 4:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "body must be float32 samples"})
                return
            audio = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            try:
                payload = enrol.embed(audio)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:  # the encoder download or onnxruntime failed
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"enrolment failed: {exc}"})
                return
            self._send_json(HTTPStatus.OK, payload)

    return Handler


def serve(config: DemoConfig) -> ThreadingHTTPServer:
    """A server bound to ``config.host:port``; call ``serve_forever()`` on it."""
    config.check()
    return ThreadingHTTPServer((config.host, config.port), build_handler(config))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m earmark.demo", description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, type=Path, help="model blob (.emwb); its .json sits beside it")
    parser.add_argument("--wasm", type=Path, default=DEFAULT_WASM, help=f"WASM engine (default {DEFAULT_WASM})")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (keep it local)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-enrol", action="store_true", help="serve without the speaker encoder")
    args = parser.parse_args(argv)

    config = DemoConfig(
        model=args.model, wasm=args.wasm, host=args.host, port=args.port, enrol=not args.no_enrol
    )
    try:
        server = serve(config)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"earmark demo: http://{config.host}:{config.port}  (model {config.model.name}, "
          f"enrolment {'on' if config.enrol else 'off'})")
    print("open that address, allow the microphone, and wear headphones.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
