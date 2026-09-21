"""The local demo server: routes, guards and enrolment validation (no browser, no encoder)."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from earmark import constants as C
from earmark.demo import DemoConfig, build_handler

PAGE = "<!doctype html><title>demo</title>"


class StubEnroller:
    """Stands in for the WeSpeaker encoder; keeps the real length checks in DemoConfig's path."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed(self, audio: np.ndarray) -> dict[str, Any]:
        seconds = audio.size / C.SAMPLE_RATE
        if not 1.0 <= seconds <= 30.0:
            raise ValueError(f"enrolment must be 1-30 s, got {seconds:.2f} s")
        if float(np.sqrt(np.mean(np.square(audio.astype(np.float64))))) < 1e-3:
            raise ValueError("that recording is silent; speak while enrolling")
        self.calls.append(audio.size)
        return {"embedding": [0.0] * C.EMBEDDING_DIM, "encoder": "stub", "seconds": seconds, "rms": 1.0}


@pytest.fixture
def demo(tmp_path: Path) -> Any:
    web = tmp_path / "web"
    (web / "src").mkdir(parents=True)
    (web / "index.html").write_text(PAGE)
    (web / "src" / "app.js").write_text("// app")
    (tmp_path / "secret.txt").write_text("not served")
    model = tmp_path / "model.emwb"
    model.write_bytes(b"EMWBLOB\x00blob")
    model.with_suffix(".json").write_text(json.dumps({"model": {"name": "M"}}))
    wasm = tmp_path / "earmark.wasm"
    wasm.write_bytes(b"\x00asm")

    config = DemoConfig(model=model, wasm=wasm, web_root=web, port=0)
    config.check()
    enroller = StubEnroller()
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(config, enroller))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, enroller, config
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def get(base: str, route: str) -> tuple[int, bytes]:
    try:
        with urlopen(f"{base}{route}", timeout=10) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def post(base: str, route: str, body: bytes) -> tuple[int, dict[str, Any]]:
    request = Request(f"{base}{route}", data=body, method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def pcm(seconds: float, amplitude: float = 0.1) -> bytes:
    n = int(seconds * C.SAMPLE_RATE)
    rng = np.random.default_rng(0)
    return (amplitude * rng.standard_normal(n)).astype("<f4").tobytes()


def test_serves_the_page_engine_and_weights(demo: Any) -> None:
    base, _, config = demo
    status, body = get(base, "/")
    assert status == 200 and body.decode() == PAGE
    assert get(base, "/src/app.js")[1] == b"// app"
    assert get(base, "/engine/earmark.wasm")[1] == config.wasm.read_bytes()
    assert get(base, "/model/weights.emwb")[1] == config.model.read_bytes()
    assert json.loads(get(base, "/model/manifest.json")[1])["model"]["name"] == "M"
    assert json.loads(get(base, "/health")[1])["ok"] is True


def test_unknown_routes_and_paths_outside_the_web_root_are_refused(demo: Any) -> None:
    base, _, _ = demo
    assert get(base, "/nope")[0] == 404
    for escape in ("/../secret.txt", "/src/../../secret.txt", "//etc/passwd"):
        status, body = get(base, escape)
        assert status == 404, escape
        assert b"not served" not in body


def test_enrolment_returns_an_embedding(demo: Any) -> None:
    base, enroller, _ = demo
    status, payload = post(base, "/enrol", pcm(5.0))
    assert status == 200
    assert len(payload["embedding"]) == C.EMBEDDING_DIM
    assert enroller.calls == [5 * C.SAMPLE_RATE]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (pcm(0.2), "1-30 s"),
        (pcm(5.0, amplitude=0.0), "silent"),
        (b"", "bytes"),
        (b"\x00\x00\x00", "float32"),
    ],
)
def test_bad_enrolment_audio_is_refused(demo: Any, body: bytes, message: str) -> None:
    base, enroller, _ = demo
    status, payload = post(base, "/enrol", body)
    assert status == 400
    assert message in payload["error"]
    assert enroller.calls == []


def test_oversized_enrolment_is_refused_before_reading(demo: Any) -> None:
    base, enroller, _ = demo
    # The server answers 413 and hangs up without taking the upload, so the client may
    # also just see the connection go away; either way nothing reaches the encoder.
    try:
        status, payload = post(base, "/enrol", b"\x00" * (31 * C.SAMPLE_RATE * 4))
        assert status == 413 and "bytes" in payload["error"]
    except URLError:
        pass
    assert enroller.calls == []


def test_enrolment_can_be_switched_off(tmp_path: Path, demo: Any) -> None:
    base, _, config = demo
    off = DemoConfig(model=config.model, wasm=config.wasm, web_root=config.web_root, port=0, enrol=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(off, StubEnroller()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = post(f"http://127.0.0.1:{server.server_address[1]}", "/enrol", pcm(5.0))
        assert status == 503 and "disabled" in payload["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_missing_files_are_reported_with_the_command_that_makes_them(tmp_path: Path) -> None:
    config = DemoConfig(model=tmp_path / "absent.emwb", wasm=tmp_path / "absent.wasm", web_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="export.blob checkpoint"):
        config.check()
