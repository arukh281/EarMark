"""The Kaldi fbank port against a golden written by torchaudio (scripts/make_fbank_golden.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from earmark.data.embeddings import KALDI_FBANK, WAVE_SCALE
from earmark.data.fbank import kaldi_fbank, mel_banks

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "kaldi_fbank.npz"
#: The port reorders nothing, so it agrees with torchaudio to float32 rounding.
TOLERANCE = 2e-4


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(GOLDEN) as data:
        return {name: data[name] for name in data.files}


def _text(array: np.ndarray) -> dict[str, object]:
    return json.loads(array.tobytes().decode())


def test_golden_was_built_with_the_settings_the_encoder_uses(golden: dict[str, np.ndarray]) -> None:
    assert _text(golden["settings"]) == dict(KALDI_FBANK), "regenerate with scripts/make_fbank_golden.py"


@pytest.mark.parametrize("case", ["noise", "voiced", "quiet", "loud", "one_frame"])
def test_matches_torchaudio(golden: dict[str, np.ndarray], case: str) -> None:
    wave = torch.from_numpy(golden[f"{case}.wave"])
    want = golden[f"{case}.feats"]
    got = kaldi_fbank(wave[None] * WAVE_SCALE, **KALDI_FBANK).numpy()
    assert got.shape == want.shape
    peak = max(1.0, float(np.abs(want).max()))
    assert np.abs(got - want).max() <= TOLERANCE * peak


def test_frame_count_and_layout() -> None:
    rate = int(KALDI_FBANK["sample_frequency"])
    wave = torch.zeros(rate)
    feats = kaldi_fbank(wave[None], **KALDI_FBANK)
    assert feats.shape == (1 + (rate - 400) // 160, KALDI_FBANK["num_mel_bins"])
    # Silence floors at log(eps) everywhere, never NaN or -inf.
    assert torch.isfinite(feats).all()
    assert torch.allclose(feats, torch.full_like(feats, float(np.log(np.finfo(np.float32).eps))))


def test_mel_banks_tile_the_spectrum() -> None:
    banks = mel_banks(80, 512, 16000.0, 20.0, 0.0)
    assert banks.shape == (80, 256)
    assert (banks >= 0).all()
    assert banks.sum() > 0
    # Bin centres move up in frequency (the lowest bins are narrower than an FFT bin,
    # so neighbours there can peak on the same one).
    peaks = banks.argmax(dim=1)
    assert torch.all(peaks[1:] >= peaks[:-1])
    assert peaks[-1] > peaks[0]


def test_unsupported_settings_are_refused() -> None:
    wave = torch.zeros(4000)
    for bad in ({"dither": 1.0}, {"window_type": "povey"}, {"use_energy": True}):
        with pytest.raises(ValueError):
            kaldi_fbank(wave[None], **{**KALDI_FBANK, **bad})
    with pytest.raises(ValueError):
        kaldi_fbank(torch.zeros(2, 4000), **KALDI_FBANK)
