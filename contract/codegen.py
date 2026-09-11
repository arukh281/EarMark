#!/usr/bin/env python3
"""Generate the Earmark signal-contract constants for Python, C++ and JavaScript.

``contract/signal.yaml`` is the single source of truth for every signal-level constant
(sample rate, WOLA framing, ERB bands, deep-filter shape, normalisation, embedding size,
VAD label rule, barge-in event). This script validates it, derives the dependent values
and renders three files that must be committed alongside the YAML:

* ``python/earmark/constants.py``
* ``engine/include/earmark_constants.h``
* ``web/src/constants.js``

Every constant has the same UPPER_SNAKE name in all three languages (C macros carry an
``EARMARK_`` prefix). Usage::

    python contract/codegen.py                  # regenerate the committed files
    python contract/codegen.py --check          # exit 1 if any committed file is stale
    python contract/codegen.py --out-root DIR   # render into DIR instead of the repo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACT = REPO_ROOT / "contract" / "signal.yaml"

#: Output files, relative to the output root (the repository root by default).
OUTPUTS: dict[str, Path] = {
    "python": Path("python/earmark/constants.py"),
    "cpp": Path("engine/include/earmark_constants.h"),
    "js": Path("web/src/constants.js"),
}

SUPPORTED_VERSION = 1
HASH_HEX_DIGITS = 16
_WRAP = 16  # integers per line when rendering arrays

Value = bool | int | float | str | tuple[int, ...]


class ContractError(ValueError):
    """Raised when ``signal.yaml`` is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class Constant:
    """One generated constant: identical name and value in every language."""

    name: str
    value: Value
    doc: str


# --------------------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------------------


def _leaf_paths(node: Mapping[str, Any], prefix: str = "") -> Iterator[str]:
    for key, child in node.items():
        path = f"{prefix}{key}"
        if isinstance(child, Mapping):
            yield from _leaf_paths(child, f"{path}.")
        else:
            yield path


class _Reader:
    """Typed, strict accessor over the parsed YAML that tracks which keys were used."""

    def __init__(self, tree: object) -> None:
        if not isinstance(tree, Mapping):
            raise ContractError("the contract must be a YAML mapping at the top level")
        self._tree: Mapping[str, Any] = tree
        self._used: set[str] = set()

    def _raw(self, path: str) -> object:
        node: object = self._tree
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise ContractError(f"missing key {path!r}")
            node = node[part]
        self._used.add(path)
        return node

    def get_int(self, path: str, *, minimum: int = 1) -> int:
        value = self._raw(path)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{path} must be an integer, got {value!r}")
        if value < minimum:
            raise ContractError(f"{path} must be >= {minimum}, got {value}")
        return value

    def get_float(self, path: str) -> float:
        value = self._raw(path)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ContractError(f"{path} must be a number, got {value!r}")
        result = float(value)
        if not math.isfinite(result):
            raise ContractError(f"{path} must be finite, got {value!r}")
        return result

    def get_bool(self, path: str) -> bool:
        value = self._raw(path)
        if not isinstance(value, bool):
            raise ContractError(f"{path} must be true or false, got {value!r}")
        return value

    def get_choice(self, path: str, allowed: Sequence[str]) -> str:
        value = self._raw(path)
        if value not in allowed:
            raise ContractError(f"{path} must be one of {list(allowed)}, got {value!r}")
        return str(value)

    def check_all_used(self) -> None:
        unused = sorted(set(_leaf_paths(self._tree)) - self._used)
        if unused:
            raise ContractError(f"unknown keys in the contract: {unused}")


def _exact_div(numerator: int, denominator: int, what: str) -> int:
    if numerator % denominator:
        raise ContractError(f"{what} is not a whole number ({numerator} / {denominator})")
    return numerator // denominator


def _hz_to_erb(freq_hz: float) -> float:
    return 9.265 * math.log1p(freq_hz / (24.7 * 9.265))


def _erb_to_hz(n_erb: float) -> float:
    return 24.7 * 9.265 * math.expm1(n_erb / 9.265)


def erb_band_widths(sample_rate: int, n_fft: int, n_bands: int, min_bins: int) -> tuple[int, ...]:
    """Split the ``n_fft // 2 + 1`` one-sided bins into ``n_bands`` contiguous ERB bands.

    This is the DeepFilterNet ``erb_fb`` rule, computed in double precision: band upper
    edges are equally spaced on the ERB-rate scale and rounded (half away from zero) to
    the nearest bin; any band narrower than ``min_bins`` is widened to ``min_bins`` and
    the surplus is taken from the following band(s); the last band absorbs the remainder
    so the widths sum to exactly ``n_fft // 2 + 1``.
    """
    n_bins = n_fft // 2 + 1
    if n_bands * min_bins > n_bins:
        raise ContractError(f"{n_bands} ERB bands of >= {min_bins} bins exceed {n_bins} bins")
    bin_hz = sample_rate / n_fft
    erb_low = _hz_to_erb(0.0)
    step = (_hz_to_erb(sample_rate / 2) - erb_low) / n_bands
    widths: list[int] = []
    prev_edge = 0
    overflow = 0
    for band in range(1, n_bands + 1):
        edge = math.floor(_erb_to_hz(erb_low + band * step) / bin_hz + 0.5)
        width = edge - prev_edge - overflow
        if width < min_bins:
            overflow = min_bins - width
            width = min_bins
        else:
            overflow = 0
        widths.append(width)
        prev_edge = edge
    widths[-1] += n_bins - sum(widths)
    if widths[-1] < min_bins:
        raise ContractError("the last ERB band ends up narrower than min_bins_per_band")
    return tuple(widths)


def contract_hash(constants: Sequence[Constant]) -> str:
    """Hash of every (name, value) pair; identifies the contract in manifests and goldens."""
    payload = json.dumps([[c.name, c.value] for c in constants], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:HASH_HEX_DIGITS]


def build_constants(tree: object) -> list[Constant]:
    """Validate a parsed contract and return the full, ordered constant table."""
    r = _Reader(tree)

    version = r.get_int("version")
    if version != SUPPORTED_VERSION:
        raise ContractError(f"unsupported contract version {version}; expected {SUPPORTED_VERSION}")

    sample_rate = r.get_int("audio.sample_rate")
    channels = r.get_int("audio.channels")
    if channels != 1:
        raise ContractError(f"the model boundary is mono; audio.channels must be 1, got {channels}")

    window = r.get_choice("stft.window", ["sqrt_hann_periodic"])
    window_length = r.get_int("stft.window_length")
    hop_length = r.get_int("stft.hop_length")
    n_fft = r.get_int("stft.n_fft")
    fft_norm = r.get_choice("stft.fft_norm", ["backward"])
    center = r.get_bool("stft.center")
    if n_fft % 2:
        raise ContractError(f"stft.n_fft must be even, got {n_fft}")
    if window_length > n_fft:
        raise ContractError("stft.window_length must not exceed stft.n_fft")
    if 2 * hop_length != window_length:
        raise ContractError(
            "sqrt-Hann WOLA reconstructs perfectly only at 50% overlap: "
            f"2 * hop_length ({2 * hop_length}) != window_length ({window_length})"
        )
    if center:
        raise ContractError("stft.center must be false: centred framing adds lookahead")
    n_bins = n_fft // 2 + 1
    hop_ms = _exact_div(hop_length * 1000, sample_rate, "hop length in ms")
    window_ms = _exact_div(window_length * 1000, sample_rate, "window length in ms")
    frame_rate_hz = _exact_div(sample_rate, hop_length, "frame rate in Hz")

    latency_ms = r.get_int("latency.algorithmic_ms")
    lookahead_frames = r.get_int("latency.lookahead_frames", minimum=0)
    latency_samples = window_length + lookahead_frames * hop_length
    if latency_samples * 1000 != latency_ms * sample_rate:
        raise ContractError(
            f"latency.algorithmic_ms ({latency_ms}) != (window_length + lookahead_frames * "
            f"hop_length) / sample_rate ({latency_samples * 1000 / sample_rate} ms)"
        )

    erb_bands = r.get_int("erb.n_bands")
    erb_min_bins = r.get_int("erb.min_bins_per_band")
    erb_widths = erb_band_widths(sample_rate, n_fft, erb_bands, erb_min_bins)

    df_order = r.get_int("deep_filter.order")
    df_bins = r.get_int("deep_filter.n_bins")
    df_lookahead = r.get_int("deep_filter.lookahead_frames", minimum=0)
    if df_bins > n_bins:
        raise ContractError(f"deep_filter.n_bins ({df_bins}) exceeds the {n_bins} STFT bins")
    if df_lookahead >= df_order:
        raise ContractError("deep_filter.lookahead_frames must be smaller than deep_filter.order")
    if df_lookahead > lookahead_frames:
        raise ContractError("deep_filter.lookahead_frames exceeds latency.lookahead_frames")

    norm_kind = r.get_choice("normalization.kind", ["causal_exponential_mean"])
    norm_tau_s = r.get_float("normalization.tau_s")
    if norm_tau_s <= 0.0:
        raise ContractError("normalization.tau_s must be positive")
    norm_alpha = math.exp(-hop_length / (sample_rate * norm_tau_s))

    embedding_dim = r.get_int("embedding.dim")

    vad_reference = r.get_choice("vad_label.reference", ["utterance_peak"])
    vad_threshold_db = r.get_float("vad_label.threshold_db")
    if vad_threshold_db >= 0.0:
        raise ContractError("vad_label.threshold_db is relative to the peak and must be negative")
    vad_hangover_ms = r.get_int("vad_label.hangover_ms", minimum=0)
    vad_hangover_frames = _exact_div(vad_hangover_ms, hop_ms, "VAD hangover in frames")

    bargein_active_ms = r.get_int("bargein.min_active_ms")
    bargein_silence_ms = r.get_int("bargein.min_silence_ms")
    bargein_active_frames = _exact_div(bargein_active_ms, hop_ms, "barge-in activity in frames")
    bargein_silence_frames = _exact_div(bargein_silence_ms, hop_ms, "barge-in silence in frames")

    r.check_all_used()

    body = [
        Constant("SAMPLE_RATE", sample_rate, "Internal model sample rate in Hz."),
        Constant("NUM_CHANNELS", channels, "Channel count at the model boundary (mono)."),
        Constant(
            "WINDOW",
            window,
            "Analysis and synthesis window: w[n] = sin(pi * n / WINDOW_LENGTH), n in [0, WINDOW_LENGTH).",
        ),
        Constant("WINDOW_LENGTH", window_length, "WOLA window length in samples."),
        Constant("HOP_LENGTH", hop_length, "WOLA hop in samples; one model step per hop."),
        Constant("N_FFT", n_fft, "FFT size in samples."),
        Constant("N_BINS", n_bins, "One-sided STFT bins, N_FFT // 2 + 1."),
        Constant("FFT_NORM", fft_norm, "Forward DFT unscaled, inverse DFT scaled by 1 / N_FFT."),
        Constant(
            "STFT_CENTER",
            center,
            "False: frame t covers samples [t * HOP_LENGTH, t * HOP_LENGTH + WINDOW_LENGTH).",
        ),
        Constant("HOP_MS", hop_ms, "Hop duration in milliseconds."),
        Constant("WINDOW_MS", window_ms, "Window duration in milliseconds."),
        Constant("FRAME_RATE_HZ", frame_rate_hz, "Model frames (hops) per second."),
        Constant("ALGORITHMIC_LATENCY_MS", latency_ms, "Algorithmic latency in milliseconds."),
        Constant(
            "ALGORITHMIC_LATENCY_SAMPLES",
            latency_samples,
            "Algorithmic latency in samples: WINDOW_LENGTH + LOOKAHEAD_FRAMES * HOP_LENGTH.",
        ),
        Constant("LOOKAHEAD_FRAMES", lookahead_frames, "Future frames the model may see (zero)."),
        Constant("ERB_BANDS", erb_bands, "Number of ERB bands for gains and log-power features."),
        Constant("ERB_MIN_BINS", erb_min_bins, "Minimum STFT bins per ERB band."),
        Constant(
            "ERB_WIDTHS",
            erb_widths,
            "STFT bins per ERB band, low to high; contiguous and summing to N_BINS.",
        ),
        Constant("DF_ORDER", df_order, "Deep-filter taps per bin (current and previous frames)."),
        Constant("DF_BINS", df_bins, "Deep filter applies to STFT bins [0, DF_BINS)."),
        Constant("DF_LOOKAHEAD_FRAMES", df_lookahead, "Future frames used by the deep filter."),
        Constant("NORM_KIND", norm_kind, "Feature normalisation: causal exponential moving mean."),
        Constant("NORM_TAU_S", norm_tau_s, "Normalisation time constant in seconds."),
        Constant(
            "NORM_ALPHA",
            norm_alpha,
            "Per-hop decay exp(-HOP_LENGTH / (SAMPLE_RATE * NORM_TAU_S)): m = a * m + (1 - a) * x.",
        ),
        Constant("EMBEDDING_DIM", embedding_dim, "Speaker-embedding dimension."),
        Constant("VAD_REFERENCE", vad_reference, "VAD label threshold reference level."),
        Constant(
            "VAD_THRESHOLD_DB",
            vad_threshold_db,
            "Frame is target-active when direct-path energy exceeds the peak by this many dB.",
        ),
        Constant("VAD_HANGOVER_MS", vad_hangover_ms, "VAD label hangover in milliseconds."),
        Constant("VAD_HANGOVER_FRAMES", vad_hangover_frames, "VAD label hangover in frames."),
        Constant("BARGEIN_MIN_ACTIVE_MS", bargein_active_ms, "Barge-in onset: minimum activity in ms."),
        Constant(
            "BARGEIN_MIN_ACTIVE_FRAMES", bargein_active_frames, "Barge-in onset: minimum activity in frames."
        ),
        Constant(
            "BARGEIN_MIN_SILENCE_MS", bargein_silence_ms, "Barge-in onset: minimum preceding silence in ms."
        ),
        Constant(
            "BARGEIN_MIN_SILENCE_FRAMES",
            bargein_silence_frames,
            "Barge-in onset: minimum preceding silence in frames.",
        ),
    ]
    header = [
        Constant("CONTRACT_VERSION", version, "Schema version of contract/signal.yaml."),
        Constant(
            "CONTRACT_HASH",
            contract_hash(body),
            "SHA-256 prefix of every other constant; stamp it into weight manifests and goldens.",
        ),
    ]
    return header + body


def load_contract(path: Path = DEFAULT_CONTRACT) -> list[Constant]:
    """Parse and validate ``path`` and return the ordered constant table."""
    try:
        tree = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ContractError(f"{path}: invalid YAML: {exc}") from exc
    return build_constants(tree)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

_BANNER = (
    "GENERATED by contract/codegen.py from contract/signal.yaml. DO NOT EDIT.",
    "Regenerate with `make codegen`; `make contract-check` fails when this file is stale.",
)


def _wrap_ints(values: Sequence[int], indent: str) -> list[str]:
    return [
        indent + ", ".join(str(v) for v in values[i : i + _WRAP]) + ","
        for i in range(0, len(values), _WRAP)
    ]


def _float_literal(value: float) -> str:
    text = repr(value)  # shortest string that round-trips to the same double
    return text if any(ch in text for ch in ".eE") else f"{text}.0"


def render_python(constants: Sequence[Constant]) -> str:
    """Render ``python/earmark/constants.py``."""
    lines = ['"""Earmark signal-contract constants.', "", *_BANNER, '"""', ""]
    lines += ["from typing import Final", ""]
    for c in constants:
        lines.append(f"#: {c.doc}")
        match c.value:
            case bool():
                lines.append(f"{c.name}: Final[bool] = {c.value!r}")
            case int():
                lines.append(f"{c.name}: Final[int] = {c.value}")
            case float():
                lines.append(f"{c.name}: Final[float] = {_float_literal(c.value)}")
            case str():
                lines.append(f"{c.name}: Final[str] = {json.dumps(c.value)}")
            case tuple():
                lines.append(f"{c.name}: Final[tuple[int, ...]] = (")
                lines += _wrap_ints(c.value, "    ")
                lines.append(")")
        lines.append("")
    lines.append("__all__ = [")
    lines += [f'    "{c.name}",' for c in constants]
    lines.append("]")
    return "\n".join(lines) + "\n"


def _c_scalar(value: bool | int | float | str) -> str:
    match value:
        case bool():
            return "1" if value else "0"
        case int():
            return f"({value})" if value < 0 else str(value)
        case float():
            text = _float_literal(value)
            return f"({text})" if value < 0 else text
        case str():
            return json.dumps(value)
    raise TypeError(f"unsupported scalar {value!r}")


def render_cpp(constants: Sequence[Constant]) -> str:
    """Render ``engine/include/earmark_constants.h`` (C macros plus C++17 constexpr)."""
    guard = "EARMARK_CONSTANTS_H_"
    lines = ["/*", " * Earmark signal-contract constants.", " *"]
    lines += [f" * {line}" for line in _BANNER]
    lines += [
        " *",
        " * The EARMARK_* macros are usable from C (the earmark.h ABI); C++ code should use",
        " * the typed constants in namespace earmark::contract.",
        " */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
    ]
    for c in constants:
        lines.append(f"/* {c.doc} */")
        if isinstance(c.value, tuple):
            lines.append(f"#define EARMARK_{c.name}_COUNT {len(c.value)}")
            lines.append(f"#define EARMARK_{c.name}_INIT {{ \\")
            lines += [f"{line} \\" for line in _wrap_ints(c.value, "    ")]
            lines.append("}")
        else:
            lines.append(f"#define EARMARK_{c.name} {_c_scalar(c.value)}")
    lines += [
        "",
        "#ifdef __cplusplus",
        "",
        "#include <array>",
        "#include <cstddef>",
        "#include <string_view>",
        "",
        "namespace earmark::contract {",
        "",
    ]
    for c in constants:
        lines.append(f"/** {c.doc} */")
        match c.value:
            case bool():
                lines.append(f"inline constexpr bool {c.name} = EARMARK_{c.name} != 0;")
            case int():
                lines.append(f"inline constexpr int {c.name} = EARMARK_{c.name};")
            case float():
                lines.append(f"inline constexpr double {c.name} = EARMARK_{c.name};")
            case str():
                lines.append(f"inline constexpr std::string_view {c.name} = EARMARK_{c.name};")
            case tuple():
                lines.append(
                    f"inline constexpr std::array<int, EARMARK_{c.name}_COUNT> {c.name} = "
                    f"EARMARK_{c.name}_INIT;"
                )
    lines += [
        "",
        "namespace detail {",
        "template <std::size_t N>",
        "constexpr int sum(const std::array<int, N>& values) {",
        "  int total = 0;",
        "  for (const int v : values) total += v;",
        "  return total;",
        "}",
        "}  // namespace detail",
        "",
        'static_assert(N_BINS == N_FFT / 2 + 1, "N_BINS must be N_FFT / 2 + 1");',
        'static_assert(2 * HOP_LENGTH == WINDOW_LENGTH, "sqrt-Hann WOLA needs 50% overlap");',
        'static_assert(detail::sum(ERB_WIDTHS) == N_BINS, "ERB bands must cover every bin");',
        'static_assert(static_cast<int>(ERB_WIDTHS.size()) == ERB_BANDS, "ERB band count");',
        'static_assert(DF_BINS <= N_BINS, "deep-filter bins exceed the spectrum");',
        "",
        "}  // namespace earmark::contract",
        "",
        "#endif  /* __cplusplus */",
        "",
        f"#endif  /* {guard} */",
    ]
    return "\n".join(lines) + "\n"


def render_js(constants: Sequence[Constant]) -> str:
    """Render ``web/src/constants.js`` (an ES module; arrays are frozen)."""
    lines = ["/**", " * Earmark signal-contract constants.", " *"]
    lines += [f" * {line}" for line in _BANNER]
    lines += [" */", ""]
    for c in constants:
        lines.append(f"/** {c.doc} */")
        match c.value:
            case bool():
                lines.append(f"export const {c.name} = {'true' if c.value else 'false'};")
            case int():
                lines.append(f"export const {c.name} = {c.value};")
            case float():
                lines.append(f"export const {c.name} = {_float_literal(c.value)};")
            case str():
                lines.append(f"export const {c.name} = {json.dumps(c.value)};")
            case tuple():
                lines.append(f"export const {c.name} = Object.freeze([")
                lines += _wrap_ints(c.value, "  ")
                lines.append("]);")
    return "\n".join(lines) + "\n"


RENDERERS = {"python": render_python, "cpp": render_cpp, "js": render_js}


def render_all(constants: Sequence[Constant]) -> dict[Path, str]:
    """Map each output path (relative to the output root) to its rendered text."""
    return {OUTPUTS[lang]: render(constants) for lang, render in RENDERERS.items()}


def write_outputs(constants: Sequence[Constant], out_root: Path) -> list[Path]:
    """Render and write every output under ``out_root``; return the written paths."""
    written: list[Path] = []
    for rel, text in render_all(constants).items():
        target = out_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        written.append(target)
    return written


def stale_outputs(constants: Sequence[Constant], out_root: Path) -> list[Path]:
    """Return the outputs under ``out_root`` that are missing or differ from a fresh render."""
    stale: list[Path] = []
    for rel, text in render_all(constants).items():
        target = out_root / rel
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            stale.append(target)
    return stale


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT, help="signal.yaml path")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT, help="output root directory")
    parser.add_argument("--check", action="store_true", help="fail if any output is stale")
    args = parser.parse_args(argv)

    try:
        constants = load_contract(args.contract)
    except (ContractError, OSError) as exc:
        print(f"codegen: {exc}", file=sys.stderr)
        return 2

    if args.check:
        stale = stale_outputs(constants, args.out_root)
        if stale:
            for path in stale:
                print(f"codegen: stale or missing: {path}", file=sys.stderr)
            print("codegen: run `make codegen` and commit the result", file=sys.stderr)
            return 1
        print(f"codegen: contract {constants[1].value} is in sync ({len(OUTPUTS)} files)")
        return 0

    for path in write_outputs(constants, args.out_root):
        print(f"codegen: wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
