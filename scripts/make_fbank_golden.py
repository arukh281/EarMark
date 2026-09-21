#!/usr/bin/env python3
"""Write the Kaldi-fbank golden that ``tests/data/test_fbank.py`` pins the port to.

Run this in an environment that has **torchaudio** (the repo venv usually does not: the
demo runs on a torch release torchaudio has no wheel for). For example::

    uv venv /tmp/ta && uv pip install --python /tmp/ta/bin/python "torch==2.11.*" "torchaudio==2.11.*"
    /tmp/ta/bin/python scripts/make_fbank_golden.py

The golden holds the settings, the input waveforms and torchaudio's features, so the
test needs neither torchaudio nor this script. Regenerate it only when the settings in
``earmark.data.embeddings.KALDI_FBANK`` change; the test fails if they drift apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torchaudio.compliance import kaldi

#: Must match earmark.data.embeddings.KALDI_FBANK (the test checks this).
SETTINGS = {
    "num_mel_bins": 80,
    "frame_length": 25.0,
    "frame_shift": 10.0,
    "dither": 0.0,
    "window_type": "hamming",
    "use_energy": False,
    "sample_frequency": 16000.0,
}
#: WeSpeaker scales float audio to the int16 range before the fbank (embeddings.WAVE_SCALE).
WAVE_SCALE = 32768.0
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "kaldi_fbank.npz"


def waves() -> dict[str, np.ndarray]:
    """Inputs covering silence, noise, voiced speech-like sound and a one-frame clip."""
    rng = np.random.default_rng(20260918)
    rate = int(SETTINGS["sample_frequency"])
    t = np.arange(rate) / rate
    f0 = 120.0 + 60.0 * np.sin(2.0 * math.pi * 0.8 * t)
    phase = 2.0 * math.pi * np.cumsum(f0) / rate
    voiced = sum(np.sin(k * phase) / k for k in range(1, 16)) * 0.1
    return {
        "noise": (0.05 * rng.standard_normal(rate)).astype(np.float32),
        "voiced": (voiced + 0.01 * rng.standard_normal(rate)).astype(np.float32),
        "quiet": (1e-5 * rng.standard_normal(rate // 2)).astype(np.float32),
        "loud": np.clip(2.5 * voiced[: rate // 2], -1.0, 1.0).astype(np.float32),
        "one_frame": (0.2 * rng.standard_normal(400)).astype(np.float32),
    }


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    for name, wave in waves().items():
        feats = kaldi.fbank(torch.from_numpy(wave)[None] * WAVE_SCALE, **SETTINGS)
        arrays[f"{name}.wave"] = wave
        arrays[f"{name}.feats"] = feats.numpy().astype(np.float32)
    arrays["settings"] = np.frombuffer(json.dumps(SETTINGS, sort_keys=True).encode(), dtype=np.uint8)
    arrays["versions"] = np.frombuffer(
        json.dumps({"torch": torch.__version__, "torchaudio": __import__("torchaudio").__version__}).encode(),
        dtype=np.uint8,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes): {sorted(waves())}")


if __name__ == "__main__":
    main()
