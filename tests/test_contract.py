"""The M0 signal contract: generated files are fresh and Python, C++ and JS agree.

``contract/signal.yaml`` is rendered by ``contract/codegen.py`` into three committed files.
These tests regenerate into a temporary directory and fail when a committed file is
stale, then read each language's constants back (by parsing, and by compiling or
executing them when a C/C++ compiler or Node is available) and require every value to
match the contract exactly.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT = REPO_ROOT / "contract" / "signal.yaml"


def _load_codegen() -> ModuleType:
    name = "earmark_contract_codegen"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "contract" / "codegen.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codegen = _load_codegen()


@pytest.fixture(scope="module")
def expected() -> dict[str, Any]:
    return {c.name: c.value for c in codegen.load_contract(CONTRACT)}


def _normalise(value: Any) -> Any:
    """Collapse language-specific shapes (lists, 0/1 bools) for comparison."""
    if isinstance(value, list | tuple):
        return tuple(_normalise(v) for v in value)
    if isinstance(value, bool):
        return int(value)
    return value


def _assert_same(actual: dict[str, Any], expected: dict[str, Any], source: str) -> None:
    assert set(actual) == set(expected), f"{source}: constant names differ from the contract"
    for name, want in expected.items():
        got = actual[name]
        assert _normalise(got) == _normalise(want), f"{source}: {name} = {got!r}, want {want!r}"
        if isinstance(want, float):
            assert math.isfinite(float(got))


# ---- parsers for each language ----------------------------------------------------------


def python_constants() -> dict[str, Any]:
    import earmark.constants as module

    return {name: getattr(module, name) for name in module.__all__}


_C_DEFINE = re.compile(r"^#define EARMARK_(\w+) (.+?)(?: \\)?$")


def cpp_macro_constants(text: str) -> dict[str, Any]:
    """Read the C macros; array initialisers span several continuation lines."""
    values: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = _C_DEFINE.match(lines[i])
        i += 1
        if not match:
            continue
        name, body = match.groups()
        if name.endswith("_COUNT"):
            continue
        if name.endswith("_INIT"):
            parts = [body]
            while not parts[-1].rstrip().endswith("}"):
                parts.append(lines[i].rstrip(" \\"))
                i += 1
            joined = " ".join(parts).replace("{", "(").replace("}", ")")
            values[name.removesuffix("_INIT")] = tuple(ast.literal_eval(joined))
        else:
            values[name] = ast.literal_eval(body)
    return values


_JS_CONST = re.compile(r"^export const (\w+) = (.+?);?$")


def js_static_constants(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = _JS_CONST.match(lines[i])
        i += 1
        if not match:
            continue
        name, body = match.groups()
        if body.startswith("Object.freeze(["):
            parts: list[str] = []
            while not lines[i].startswith("]);"):
                parts.append(lines[i])
                i += 1
            values[name] = tuple(json.loads("[" + "".join(parts).rstrip(", ") + "]"))
        elif body in ("true", "false"):
            values[name] = body == "true"
        else:
            values[name] = json.loads(body)
    return values


# ---- freshness --------------------------------------------------------------------------


def test_committed_files_match_a_fresh_render(tmp_path: Path) -> None:
    constants = codegen.load_contract(CONTRACT)
    codegen.write_outputs(constants, tmp_path)
    for rel in codegen.OUTPUTS.values():
        committed = REPO_ROOT / rel
        assert committed.is_file(), f"{rel} is missing: run `make codegen`"
        fresh = (tmp_path / rel).read_text(encoding="utf-8")
        assert committed.read_text(encoding="utf-8") == fresh, f"{rel} is stale: run `make codegen`"
    assert codegen.stale_outputs(constants, REPO_ROOT) == []


def test_check_mode_flags_a_stale_file(tmp_path: Path) -> None:
    constants = codegen.load_contract(CONTRACT)
    codegen.write_outputs(constants, tmp_path)
    assert codegen.main(["--check", "--out-root", str(tmp_path)]) == 0
    js = tmp_path / codegen.OUTPUTS["js"]
    js.write_text(js.read_text(encoding="utf-8").replace("16000", "48000"), encoding="utf-8")
    assert codegen.main(["--check", "--out-root", str(tmp_path)]) == 1


# ---- agreement across languages -----------------------------------------------------------


def test_python_constants_match_contract(expected: dict[str, Any]) -> None:
    _assert_same(python_constants(), expected, "python/earmark/constants.py")


def test_cpp_macros_match_contract(expected: dict[str, Any]) -> None:
    text = (REPO_ROOT / codegen.OUTPUTS["cpp"]).read_text(encoding="utf-8")
    _assert_same(cpp_macro_constants(text), expected, "earmark_constants.h")


def test_js_constants_match_contract(expected: dict[str, Any]) -> None:
    text = (REPO_ROOT / codegen.OUTPUTS["js"]).read_text(encoding="utf-8")
    _assert_same(js_static_constants(text), expected, "web/src/constants.js")


def test_all_three_languages_agree() -> None:
    py = python_constants()
    cpp = cpp_macro_constants((REPO_ROOT / codegen.OUTPUTS["cpp"]).read_text(encoding="utf-8"))
    js = js_static_constants((REPO_ROOT / codegen.OUTPUTS["js"]).read_text(encoding="utf-8"))
    assert set(py) == set(cpp) == set(js)
    for name in py:
        assert _normalise(py[name]) == _normalise(cpp[name]) == _normalise(js[name]), name


_CPP_PROBE = r"""
#include <array>
#include <cstddef>
#include <cstdio>
#include <string_view>

#include "earmark_constants.h"

namespace c = earmark::contract;
static bool first = true;
static void key(const char* name) {
  std::printf("%s\"%s\": ", first ? "" : ", ", name);
  first = false;
}
static void emit(const char* n, bool v) { key(n); std::printf("%s", v ? "true" : "false"); }
static void emit(const char* n, int v) { key(n); std::printf("%d", v); }
static void emit(const char* n, double v) { key(n); std::printf("%.17g", v); }
static void emit(const char* n, std::string_view v) {
  key(n);
  std::printf("\"%.*s\"", static_cast<int>(v.size()), v.data());
}
template <std::size_t N>
static void emit(const char* n, const std::array<int, N>& v) {
  key(n);
  std::printf("[");
  for (std::size_t i = 0; i < N; ++i) std::printf("%s%d", i ? ", " : "", v[i]);
  std::printf("]");
}
int main() {
  std::printf("{");
@EMITS@
  std::printf("}\n");
  return 0;
}
"""

_C_PROBE = r"""
#include <stdio.h>
#include "earmark_constants.h"
static const int erb[] = EARMARK_ERB_WIDTHS_INIT;
int main(void) {
  int total = 0;
  for (unsigned i = 0; i < sizeof erb / sizeof erb[0]; ++i) total += erb[i];
  printf("%d %d %s %.17g\n", EARMARK_N_BINS, total, EARMARK_CONTRACT_HASH, EARMARK_NORM_ALPHA);
  return (int)(sizeof erb / sizeof erb[0]) == EARMARK_ERB_WIDTHS_COUNT ? 0 : 1;
}
"""


def _compiler(*names: str) -> str | None:
    for name in names:
        if path := shutil.which(name):
            return path
    return None


def _build_and_run(compiler: str, flags: list[str], source: str, tmp_path: Path, stem: str) -> str:
    src = tmp_path / stem
    src.write_text(source, encoding="utf-8")
    exe = tmp_path / f"{src.stem}.out"
    include = str(REPO_ROOT / "engine" / "include")
    cmd = [compiler, *flags, "-Wall", "-Wextra", "-Werror", "-I", include, str(src), "-o", str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
    return run.stdout


def test_cpp_header_compiles_and_agrees(expected: dict[str, Any], tmp_path: Path) -> None:
    cxx = _compiler("c++", "clang++", "g++")
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    emits = "\n".join(f'  emit("{name}", c::{name});' for name in expected)
    out = _build_and_run(cxx, ["-std=c++17"], _CPP_PROBE.replace("@EMITS@", emits), tmp_path, "probe.cpp")
    _assert_same(json.loads(out), expected, "earmark_constants.h (compiled C++17)")


def test_c_macros_compile_as_c(expected: dict[str, Any], tmp_path: Path) -> None:
    cc = _compiler("cc", "clang", "gcc")
    if cc is None:
        pytest.skip("no C compiler on PATH")
    out = _build_and_run(cc, ["-std=c11"], _C_PROBE, tmp_path, "probe.c").split()
    assert int(out[0]) == int(out[1]) == expected["N_BINS"]
    assert out[2] == expected["CONTRACT_HASH"]
    assert float(out[3]) == expected["NORM_ALPHA"]


def test_js_module_executes_and_agrees(expected: dict[str, Any]) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    url = (REPO_ROOT / codegen.OUTPUTS["js"]).as_uri()
    script = (
        f"const c = await import({json.dumps(url)});"
        "if (!Object.isFrozen(c.ERB_WIDTHS)) throw new Error('ERB_WIDTHS not frozen');"
        "console.log(JSON.stringify(c));"
    )
    run = subprocess.run(
        [node, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stderr
    _assert_same(json.loads(run.stdout), expected, "web/src/constants.js (node)")


# ---- the contract's own invariants ------------------------------------------------------


def test_m0_values(expected: dict[str, Any]) -> None:
    assert expected["SAMPLE_RATE"] == 16000 and expected["NUM_CHANNELS"] == 1
    assert (expected["WINDOW_LENGTH"], expected["HOP_LENGTH"], expected["N_FFT"]) == (320, 160, 320)
    assert expected["N_BINS"] == 161
    assert expected["ALGORITHMIC_LATENCY_MS"] == 20 and expected["ALGORITHMIC_LATENCY_SAMPLES"] == 320
    assert expected["LOOKAHEAD_FRAMES"] == 0 and expected["DF_LOOKAHEAD_FRAMES"] == 0
    assert expected["ERB_BANDS"] == 32
    assert (expected["DF_ORDER"], expected["DF_BINS"]) == (3, 64)
    assert expected["NORM_TAU_S"] == 1.0
    assert expected["NORM_ALPHA"] == pytest.approx(math.exp(-0.01), abs=1e-15)
    assert expected["EMBEDDING_DIM"] == 256
    assert expected["VAD_THRESHOLD_DB"] == -40.0
    assert (expected["VAD_HANGOVER_MS"], expected["VAD_HANGOVER_FRAMES"]) == (50, 5)
    assert (expected["BARGEIN_MIN_ACTIVE_FRAMES"], expected["BARGEIN_MIN_SILENCE_FRAMES"]) == (20, 30)
    assert re.fullmatch(r"[0-9a-f]{16}", expected["CONTRACT_HASH"])


def test_erb_widths_tile_the_spectrum(expected: dict[str, Any]) -> None:
    widths = expected["ERB_WIDTHS"]
    assert len(widths) == expected["ERB_BANDS"]
    assert sum(widths) == expected["N_BINS"]
    assert min(widths) >= expected["ERB_MIN_BINS"]
    assert list(widths[:-1]) == sorted(widths[:-1]), "ERB bands must widen with frequency"


def test_sqrt_hann_wola_reconstructs_perfectly(expected: dict[str, Any], rng: np.random.Generator) -> None:
    """The documented window and FFT convention give identity reconstruction (no scaling)."""
    win, hop, n_fft = expected["WINDOW_LENGTH"], expected["HOP_LENGTH"], expected["N_FFT"]
    window = np.sin(np.pi * np.arange(win) / win)
    assert np.allclose(window[:hop] ** 2 + window[hop:] ** 2, 1.0, atol=1e-12)

    x = rng.standard_normal(hop * 50)
    y = np.zeros_like(x)
    for start in range(0, len(x) - win + 1, hop):
        spec = np.fft.rfft(window * x[start : start + win], n=n_fft)  # unscaled forward
        assert spec.shape == (expected["N_BINS"],)
        y[start : start + win] += window * np.fft.irfft(spec, n=n_fft)[:win]  # 1/N inverse
    interior = slice(win, len(x) - win)
    assert np.max(np.abs(y[interior] - x[interior])) < 1e-12


def test_hash_tracks_values() -> None:
    constants = codegen.load_contract(CONTRACT)
    body = constants[2:]
    assert codegen.contract_hash(body) == constants[1].value
    changed = [codegen.Constant(c.name, 48000 if c.name == "SAMPLE_RATE" else c.value, c.doc) for c in body]
    assert codegen.contract_hash(changed) != constants[1].value


# ---- validation -------------------------------------------------------------------------


def _tree() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("stft", "hop_length", 100, "50% overlap"),
        ("stft", "n_fft", 321, "even"),
        ("stft", "center", True, "lookahead"),
        ("stft", "window", "hann", "one of"),
        ("audio", "channels", 2, "mono"),
        ("audio", "sample_rate", "16k", "integer"),
        ("latency", "algorithmic_ms", 30, "algorithmic_ms"),
        ("deep_filter", "n_bins", 200, "exceeds"),
        ("vad_label", "hangover_ms", 55, "whole number"),
        ("vad_label", "threshold_db", 3.0, "negative"),
        ("erb", "n_bands", 90, "exceed"),
    ],
)
def test_invalid_contract_is_rejected(section: str, key: str, value: Any, message: str) -> None:
    tree = _tree()
    tree[section][key] = value
    with pytest.raises(codegen.ContractError, match=message):
        codegen.build_constants(tree)


def test_unknown_and_missing_keys_are_rejected() -> None:
    tree = _tree()
    tree["stft"]["hop_lenght"] = 160
    with pytest.raises(codegen.ContractError, match="unknown keys"):
        codegen.build_constants(tree)
    tree = _tree()
    del tree["embedding"]["dim"]
    with pytest.raises(codegen.ContractError, match="missing key"):
        codegen.build_constants(tree)
