"""Per-layer golden files for the C++ engine.

Every golden is an ``.emwb`` blob (see :mod:`earmark.export.blob`) of named inputs,
parameters and expected outputs, built from random data with a fixed seed per layer.
Expected outputs are computed in float64, either with :mod:`earmark.model.dsp` (WOLA,
ERB, normalisation) or with :mod:`earmark.export.reference` (ring buffer, resampler,
GRU, matvec), and stored as float32. Inputs are rounded to float32 *before* the
reference runs, so the engine sees exactly the values the reference used.

======================  ===================================================================
Golden                  Contents
======================  ===================================================================
``ringbuf``             400 write/read/write_zeros/discard/peek ops on a capacity-37 ring
``resampler``           48k<->16k and 44.1k<->16k: design sizes, taps, irregular blocks
``stft``                sqrt-Hann window, model-framed analysis, WOLA synthesis
``erb``                 ERB band power and features, unit-norm low band, ERB gain expansion
``gru``                 two stacked 2-layer GRUs (40->48 and 37->30), a sequence each
``matvec``              dense ``W x + b`` for awkward shapes, plus a grouped linear
``weights_small``       small blob covering every dtype and rank, with a manifest
======================  ===================================================================

Regenerate with ``python -m earmark.export.golden`` (default output
``engine/tests/goldens``). ``--check`` compares the committed files with a fresh
rebuild within a tolerance (float64 libm results can differ in the last bit across
platforms) and exits 1 when they are stale.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from earmark import constants as C
from earmark.export import blob as B
from earmark.export import reference as R
from earmark.model import dsp

#: Base seed; golden ``name`` draws from ``default_rng([GOLDEN_SEED, crc32(name)])``.
GOLDEN_SEED: Final[int] = 20260911

#: Where the engine's Catch2 tests read the goldens from.
DEFAULT_GOLDEN_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "engine" / "tests" / "goldens"

#: Ring-buffer op codes (also in engine/tests/test_ringbuf.cpp).
RING_WRITE, RING_READ, RING_WRITE_ZEROS, RING_DISCARD, RING_PEEK = 0, 1, 2, 3, 4

#: Resampler cases (input rate, output rate) and the taps stride stored for each.
RESAMPLER_CASES: Final[tuple[tuple[int, int], ...]] = (
    (48000, 16000),
    (16000, 48000),
    (44100, 16000),
    (16000, 44100),
)
#: Irregular block sizes fed to the streaming resampler (cycled).
BLOCK_PATTERN: Final[tuple[int, ...]] = (1, 7, 128, 3, 441, 480, 64, 1000, 2, 250)

#: Tolerance of :func:`compare_goldens`, relative to max(1, peak |value|) per tensor.
CHECK_TOLERANCE: Final[float] = 1e-6


def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng([GOLDEN_SEED, zlib.crc32(name.encode("ascii"))])


def _f32(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).astype(np.float32)


def _i32(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def _real_view(spec: torch.Tensor) -> np.ndarray:
    """Complex ``[..., F]`` -> float32 ``[..., F, 2]`` (re, im)."""
    return _f32(torch.view_as_real(spec.to(torch.complex128)).numpy())


def _complex64_to_128(spec_ri: np.ndarray) -> torch.Tensor:
    """float32 ``[..., F, 2]`` -> complex128 tensor ``[..., F]`` (exact widening)."""
    return torch.view_as_complex(torch.from_numpy(spec_ri.astype(np.float64)).contiguous())


def _model_frames(x32: np.ndarray) -> torch.Tensor:
    """Model framing of a float32 signal: frame t = previous hop (zeros at t=0) + hop t."""
    x64 = torch.from_numpy(x32.astype(np.float64))[None]
    padded = torch.cat([torch.zeros(1, C.HOP_LENGTH, dtype=torch.float64), x64], dim=-1)
    return dsp.frame_signal(padded)[0]


def _blocks(total: int) -> np.ndarray:
    sizes: list[int] = []
    remaining = total
    index = 0
    while remaining > 0:
        size = min(BLOCK_PATTERN[index % len(BLOCK_PATTERN)], remaining)
        sizes.append(size)
        remaining -= size
        index += 1
    return _i32(sizes)


# ------------------------------------------------------------------------------ goldens


def ringbuf_golden() -> dict[str, np.ndarray]:
    """Random op schedule on a ring of odd capacity, with every return value and read."""
    rng = _rng("ringbuf")
    capacity = 37
    model = R.RingBufferModel(capacity)
    ops: list[tuple[int, int]] = []
    counts: list[int] = []
    sizes: list[int] = []
    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    kinds = np.array([RING_WRITE, RING_READ, RING_WRITE_ZEROS, RING_DISCARD, RING_PEEK])
    for _ in range(400):
        kind = int(rng.choice(kinds, p=[0.36, 0.3, 0.08, 0.1, 0.16]))
        n = int(rng.integers(0, 46))
        if kind == RING_WRITE:
            values = _f32(rng.standard_normal(n))
            inputs.append(values)
            count = model.write(values)
        elif kind == RING_READ:
            out = model.read(n)
            outputs.append(out)
            count = len(out)
        elif kind == RING_WRITE_ZEROS:
            count = model.write_zeros(n)
        elif kind == RING_DISCARD:
            count = model.discard(n)
        else:
            out = model.peek(n)
            outputs.append(out)
            count = len(out)
        ops.append((kind, n))
        counts.append(count)
        sizes.append(model.size)
    return {
        "capacity": _i32([capacity]),
        "ops": _i32(ops),
        "input": _f32(np.concatenate(inputs)),
        "expected_counts": _i32(counts),
        "expected_sizes": _i32(sizes),
        "expected_output": _f32(np.concatenate(outputs)),
    }


def resampler_golden() -> dict[str, np.ndarray]:
    """Design sizes, (strided) prototype taps and a streamed signal for each rate pair."""
    out: dict[str, np.ndarray] = {"cases": _i32(RESAMPLER_CASES)}
    for index, (in_rate, out_rate) in enumerate(RESAMPLER_CASES):
        rng = _rng(f"resampler.{in_rate}.{out_rate}")
        design = R.design_resampler(in_rate, out_rate)
        taps = R.resampler_taps(design)
        stride = 1 if design.prototype_taps <= 4096 else 16
        n_in = in_rate // 10
        t = np.arange(n_in) / in_rate
        x = 0.25 * rng.standard_normal(n_in) + 0.3 * np.sin(2 * math.pi * 997.0 * t)
        x32 = _f32(x)
        y = R.resample_reference(x32, design, taps)
        prefix = f"case{index}."
        out[prefix + "rates"] = _i32([in_rate, out_rate])
        out[prefix + "design"] = _i32(
            [design.up, design.down, design.taps_per_phase, design.prototype_taps]
        )
        out[prefix + "delay_out_samples"] = _f32([design.delay_out_samples])
        out[prefix + "taps_stride"] = _i32([stride])
        out[prefix + "taps"] = taps[::stride].copy()
        out[prefix + "input"] = x32
        out[prefix + "blocks"] = _blocks(n_in)
        out[prefix + "expected"] = _f32(y)
    return out


def stft_golden() -> dict[str, np.ndarray]:
    """Window, model-framed analysis of a test signal, and WOLA synthesis of a modified spectrum."""
    rng = _rng("stft")
    frames = 16
    n = frames * C.HOP_LENGTH
    t = np.arange(n) / C.SAMPLE_RATE
    x = (
        0.1 * rng.standard_normal(n)
        + 0.3 * np.sin(2 * math.pi * 440.0 * t)
        + 0.2 * np.sin(2 * math.pi * 3150.5 * t + 0.3)
    )
    x32 = _f32(x)
    window = dsp.sqrt_hann_window(dtype=torch.float64)
    spec = dsp.analysis_frame(_model_frames(x32), window)

    gains = rng.uniform(0.0, 1.5, spec.shape) * np.exp(1j * rng.uniform(-math.pi, math.pi, spec.shape))
    modified = spec.numpy() * gains
    modified[:, 0] = modified[:, 0].real  # c2r transforms ignore Im of DC and Nyquist
    modified[:, -1] = modified[:, -1].real
    synth_ri = _real_view(torch.from_numpy(modified))
    synth = _complex64_to_128(synth_ri)
    wav, tail = dsp.overlap_add(dsp.synthesis_frame(synth[None], window))
    return {
        "window": _f32(window.numpy()),
        "input": x32,
        "spec": _real_view(spec),
        "synth_spec": synth_ri,
        "synth_output": _f32(wav[0].numpy()),
        "synth_tail": _f32(tail[0].numpy()),
    }


def erb_golden() -> dict[str, np.ndarray]:
    """ERB power/features and unit-norm features over a level ramp, a silent gap and loud noise."""
    rng = _rng("erb")
    frames = 32
    n = frames * C.HOP_LENGTH
    level_db = np.concatenate(
        [
            np.linspace(-70.0, 0.0, 12 * C.HOP_LENGTH),
            np.full(4 * C.HOP_LENGTH, -np.inf),  # exact silence
            np.full(n - 16 * C.HOP_LENGTH, 0.0),
        ]
    )
    x = 0.5 * rng.standard_normal(n) * np.power(10.0, level_db / 20.0)
    spec64 = dsp.analysis_frame(_model_frames(_f32(x)), dsp.sqrt_hann_window(dtype=torch.float64))
    spec_ri = _real_view(spec64)
    spec = _complex64_to_128(spec_ri)  # the exact values the engine receives

    erb_init = _f32(dsp.erb_norm_init(dtype=torch.float64).numpy())
    spec_init = _f32(dsp.unit_norm_init(dtype=torch.float64).numpy())
    matrix = dsp.erb_matrix(dtype=torch.float64)
    power = dsp.erb_power(spec, matrix)
    erb_feat, erb_last = dsp.erb_features(
        spec[None], torch.from_numpy(erb_init.astype(np.float64))[None], matrix
    )
    spec_feat, spec_last = dsp.unit_norm_features(
        spec[None, :, : C.DF_BINS], torch.from_numpy(spec_init.astype(np.float64))[None]
    )
    gains = _f32(rng.uniform(0.0, 1.0, (frames, C.ERB_BANDS)))
    gained = spec * dsp.erb_expand(torch.from_numpy(gains.astype(np.float64)))
    return {
        "erb_widths": _i32(C.ERB_WIDTHS),
        "spec": spec_ri,
        "erb_norm_init": erb_init,
        "spec_norm_init": spec_init,
        "erb_power": _f32(power.numpy()),
        "erb_feat": _f32(erb_feat[0].numpy()),
        "erb_norm_final": _f32(erb_last[0].numpy()),
        "spec_feat": _real_view(spec_feat[0]),
        "spec_norm_final": _f32(spec_last[0].numpy()),
        "gains": gains,
        "gained_spec": _real_view(gained),
    }


#: GRU golden cases: name -> (input size, hidden size, layers, frames).
GRU_CASES: Final[dict[str, tuple[int, int, int, int]]] = {"a": (40, 48, 2, 12), "b": (37, 30, 2, 9)}


def gru_golden() -> dict[str, np.ndarray]:
    """Stacked GRUs with PyTorch-style init, run over a short sequence from a random state."""
    out: dict[str, np.ndarray] = {}
    for case, (inp, hidden, layers, frames) in GRU_CASES.items():
        rng = _rng(f"gru.{case}")
        bound = 1.0 / math.sqrt(hidden)
        weights: list[dict[str, np.ndarray]] = []
        for layer in range(layers):
            in_l = inp if layer == 0 else hidden
            weights.append(
                {
                    "weight_ih": _f32(rng.uniform(-bound, bound, (3 * hidden, in_l))),
                    "weight_hh": _f32(rng.uniform(-bound, bound, (3 * hidden, hidden))),
                    "bias_ih": _f32(rng.uniform(-bound, bound, 3 * hidden)),
                    "bias_hh": _f32(rng.uniform(-bound, bound, 3 * hidden)),
                }
            )
        x = _f32(1.5 * rng.standard_normal((frames, inp)))
        h0 = _f32(0.5 * rng.standard_normal((layers, hidden)))
        y, h_final = R.gru_reference(x, weights, h0)
        out[f"{case}.dims"] = _i32([inp, hidden, layers, frames])
        for layer, layer_weights in enumerate(weights):
            for key, value in layer_weights.items():
                out[f"{case}.{key}_l{layer}"] = value
        out[f"{case}.x"] = x
        out[f"{case}.h0"] = h0
        out[f"{case}.y"] = _f32(y)
        out[f"{case}.h_final"] = _f32(h_final)
    return out


#: Dense matvec shapes (rows, cols): odd sizes exercise every SIMD tail.
MATVEC_SHAPES: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (1, 7),
    (3, 5),
    (4, 16),
    (5, 9),
    (37, 53),
    (64, 128),
    (67, 131),
)
#: Grouped linear dims (in, out, groups).
GROUPED_DIMS: Final[tuple[int, int, int]] = (64, 96, 4)


def matvec_golden() -> dict[str, np.ndarray]:
    """``W x + b`` for each shape in :data:`MATVEC_SHAPES`, plus one GroupedLinear."""
    rng = _rng("matvec")
    out: dict[str, np.ndarray] = {"shapes": _i32(MATVEC_SHAPES)}
    for index, (rows, cols) in enumerate(MATVEC_SHAPES):
        w = _f32(rng.standard_normal((rows, cols)) / math.sqrt(cols))
        x = _f32(rng.standard_normal(cols))
        b = _f32(0.1 * rng.standard_normal(rows))
        out[f"case{index}.w"] = w
        out[f"case{index}.x"] = x
        out[f"case{index}.b"] = b
        out[f"case{index}.y"] = _f32(R.matvec_reference(w, x, b))
    inp, outp, groups = GROUPED_DIMS
    w = _f32(rng.standard_normal((groups, outp // groups, inp // groups)) / math.sqrt(inp // groups))
    x = _f32(rng.standard_normal(inp))
    b = _f32(0.1 * rng.standard_normal(outp))
    out["grouped.dims"] = _i32(GROUPED_DIMS)
    out["grouped.w"] = w
    out["grouped.x"] = x
    out["grouped.b"] = b
    out["grouped.y"] = _f32(R.grouped_matvec_reference(w, x, b))
    return out


def weights_small_tensors() -> dict[str, np.ndarray]:
    """Tensors of the small loader/engine test blob; all values are exact in float32."""
    null = np.array([((i * 37) % 17 - 8) / 16.0 for i in range(C.EMBEDDING_DIM)])
    return {
        "test.arange": _f32(np.arange(24).reshape(2, 3, 4) * 0.5),
        "test.int": _i32([[1, -2, 3], [4, 5, -6]]),
        "test.scalar": _f32(3.5),
        "test.empty": _f32(np.zeros(0)),
        "test.rank6": _f32(np.arange(12).reshape(1, 2, 1, 3, 1, 2) - 5.0),
        "conditioner.null_embedding": _f32(null),
        B.ERB_NORM_INIT_TENSOR: _f32(dsp.erb_norm_init(dtype=torch.float64).numpy()),
        B.SPEC_NORM_INIT_TENSOR: _f32(dsp.unit_norm_init(dtype=torch.float64).numpy()),
    }


def weights_small_manifest(blob_bytes: bytes) -> dict[str, Any]:
    """Manifest for the small test blob (flat blob fields plus its tensor table)."""
    parsed = B.unpack_blob(blob_bytes)
    return {
        **B.blob_fields(blob_bytes, "weights_small.emwb"),
        "created_by": "earmark.export.golden",
        "random_weights": True,
        "note": "engine test blob: every dtype and rank, no model weights",
        "tensors": [entry.as_json() for entry in parsed.entries],
    }


#: Golden name -> builder. ``weights_small`` also gets a manifest (see :func:`write_goldens`).
GOLDENS: Final[dict[str, Callable[[], dict[str, np.ndarray]]]] = {
    "ringbuf": ringbuf_golden,
    "resampler": resampler_golden,
    "stft": stft_golden,
    "erb": erb_golden,
    "gru": gru_golden,
    "matvec": matvec_golden,
    "weights_small": weights_small_tensors,
}


# ------------------------------------------------------------------------------ writing


def build_golden(name: str) -> bytes:
    """Blob bytes of one golden."""
    if name not in GOLDENS:
        raise KeyError(f"unknown golden {name!r}; choose from {sorted(GOLDENS)}")
    return B.pack_blob(GOLDENS[name]())


def write_goldens(out_dir: str | Path = DEFAULT_GOLDEN_DIR, names: Iterable[str] | None = None) -> list[Path]:
    """Write ``<name>.emwb`` for each golden (and ``weights_small.json``); returns the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in names or GOLDENS:
        data = build_golden(name)
        path = out / f"{name}{B.BLOB_SUFFIX}"
        path.write_bytes(data)
        written.append(path)
        if name == "weights_small":
            manifest = out / f"{name}{B.MANIFEST_SUFFIX}"
            manifest.write_text(json.dumps(weights_small_manifest(data), indent=2) + "\n", encoding="utf-8")
            written.append(manifest)
    return written


def _compare_blobs(name: str, fresh: B.Blob, committed: B.Blob, tolerance: float) -> list[str]:
    problems: list[str] = []
    if committed.contract_hash != fresh.contract_hash:
        problems.append(f"{name}: contract {committed.contract_hash} != {fresh.contract_hash}")
    if committed.names() != fresh.names():
        problems.append(f"{name}: tensor names differ")
        return problems
    for entry in fresh.entries:
        want = fresh[entry.name]
        got = committed[entry.name]
        if got.dtype != want.dtype or got.shape != want.shape:
            problems.append(f"{name}:{entry.name}: {got.dtype}{got.shape} != {want.dtype}{want.shape}")
            continue
        if want.dtype == np.int32:
            if not np.array_equal(got, want):
                problems.append(f"{name}:{entry.name}: int values differ")
            continue
        if want.size == 0:
            continue
        peak = max(1.0, float(np.max(np.abs(want))))
        err = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
        if err > tolerance * peak:
            problems.append(f"{name}:{entry.name}: max error {err:.3g} > {tolerance:g} x {peak:.3g}")
    return problems


def compare_goldens(
    golden_dir: str | Path = DEFAULT_GOLDEN_DIR,
    names: Iterable[str] | None = None,
    tolerance: float = CHECK_TOLERANCE,
) -> list[str]:
    """Differences between committed goldens and a fresh rebuild; an empty list means in sync."""
    root = Path(golden_dir)
    problems: list[str] = []
    for name in names or GOLDENS:
        path = root / f"{name}{B.BLOB_SUFFIX}"
        if not path.exists():
            problems.append(f"{name}: {path} is missing")
            continue
        try:
            committed_bytes = path.read_bytes()
            committed = B.unpack_blob(committed_bytes)
        except B.BlobFormatError as exc:
            problems.append(f"{name}: {exc}")
            continue
        problems.extend(_compare_blobs(name, B.unpack_blob(build_golden(name)), committed, tolerance))
        if name == "weights_small":
            manifest_path = root / f"{name}{B.MANIFEST_SUFFIX}"
            try:
                B.verify_manifest(B.read_manifest(manifest_path), committed_bytes)
            except (OSError, ValueError) as exc:
                problems.append(f"{name}: manifest: {exc}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m earmark.export.golden", description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_GOLDEN_DIR), help="golden directory")
    parser.add_argument("--check", action="store_true", help="compare instead of writing; exit 1 if stale")
    parser.add_argument("names", nargs="*", help=f"subset of {sorted(GOLDENS)}")
    args = parser.parse_args(argv)
    names = args.names or None
    if args.check:
        problems = compare_goldens(args.out, names)
        for problem in problems:
            print(problem)
        print("goldens are stale" if problems else "goldens are in sync")
        return 1 if problems else 0
    for path in write_goldens(args.out, names):
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCK_PATTERN",
    "CHECK_TOLERANCE",
    "DEFAULT_GOLDEN_DIR",
    "GOLDENS",
    "GOLDEN_SEED",
    "GROUPED_DIMS",
    "GRU_CASES",
    "MATVEC_SHAPES",
    "RESAMPLER_CASES",
    "build_golden",
    "compare_goldens",
    "erb_golden",
    "gru_golden",
    "matvec_golden",
    "resampler_golden",
    "ringbuf_golden",
    "stft_golden",
    "weights_small_manifest",
    "weights_small_tensors",
    "write_goldens",
]
