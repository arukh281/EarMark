"""Unit tests for the vendored GTCRN model and its PyTorch adapter (random weights, no downloads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from earmark.eval.assets import AssetError
from earmark.eval.baselines import gtcrn_torch as G
from earmark.eval.baselines.gtcrn_model import GTConvBlock


def test_model_size_matches_the_published_48k() -> None:
    model = G.build_model()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == pytest.approx(G.GTCRN_PARAMS, rel=0.01)  # 48,245 including fixed ERB matrices
    assert not model.training


def test_forward_shape_on_random_spectrum() -> None:
    torch.manual_seed(0)
    model = G.build_model()
    spec = torch.randn(2, 257, 20, 2)
    with torch.inference_mode():
        out = model(spec)
    assert out.shape == spec.shape
    assert torch.isfinite(out).all()


def test_shuffle_equals_upstream_einops_rearrange() -> None:
    """The einops-free reshape interleaves channels exactly like 'b c g t f -> b (c g) t f'."""
    block = GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1))
    x1 = torch.arange(2 * 8 * 3 * 4, dtype=torch.float32).reshape(2, 8, 3, 4)
    x2 = -x1 - 1
    out = block.shuffle(x1, x2)
    assert out.shape == (2, 16, 3, 4)
    for c in range(8):
        assert torch.equal(out[:, 2 * c], x1[:, c])
        assert torch.equal(out[:, 2 * c + 1], x2[:, c])


def _fake_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(1)
    model = G.build_model()
    path = tmp_path / "fake_gtcrn.tar"
    torch.save({"epoch": 0, "model": model.state_dict()}, path)
    return path


def test_adapter_runs_and_keeps_length(tmp_path: Path, rng: np.random.Generator) -> None:
    enh = G.GtcrnTorch("vb", checkpoint_path=_fake_checkpoint(tmp_path), verify_sha256=False)
    assert enh.info.name == "gtcrn-vb" and enh.info.inference_path == "pytorch-offline"
    assert enh.info.params == G.GTCRN_PARAMS and enh.info.mmac_per_s == G.GTCRN_MMAC_PER_S
    for n in (4000, 16000, 16001):
        out = enh.enhance(0.1 * rng.standard_normal(n).astype(np.float32))
        assert out.audio.shape == (n,) and out.audio.dtype == np.float32
        assert np.all(np.isfinite(out.audio)) and out.vad is None
    assert enh.enhance(np.zeros(0, dtype=np.float32)).audio.size == 0
    with pytest.raises(ValueError):
        enh.enhance(np.zeros((2, 100), dtype=np.float32))


def test_adapter_is_causal_up_to_one_stft_frame(tmp_path: Path, rng: np.random.Generator) -> None:
    """Changing the future must not change output more than one 512-sample window back."""
    enh = G.GtcrnTorch("dns3", checkpoint_path=_fake_checkpoint(tmp_path), verify_sha256=False)
    a = 0.1 * rng.standard_normal(16000).astype(np.float32)
    b = a.copy()
    b[12000:] = 0.1 * rng.standard_normal(4000).astype(np.float32)
    ya, yb = enh.enhance(a).audio, enh.enhance(b).audio
    np.testing.assert_allclose(ya[: 12000 - 2 * G.N_FFT], yb[: 12000 - 2 * G.N_FFT], atol=1e-5)
    assert not np.allclose(ya[12000:], yb[12000:])


def test_adapter_refuses_unverified_checkpoints(tmp_path: Path) -> None:
    fake = _fake_checkpoint(tmp_path)
    with pytest.raises(AssetError, match="bytes|sha256"):
        G.GtcrnTorch("vb", checkpoint_path=fake)
    with pytest.raises(AssetError, match="not found"):
        G.GtcrnTorch("vb", checkpoint_path=tmp_path / "missing.tar")
    with pytest.raises(ValueError):
        G.GtcrnTorch("librispeech")  # type: ignore[arg-type]


def test_official_checkpoint_loads_when_fetched() -> None:
    """Runs only after scripts/fetch_eval_data.sh gtcrn; a missing file is not an error here."""
    try:
        enh = G.GtcrnTorch("vb")
    except AssetError:
        pytest.skip("official GTCRN checkpoint not fetched")
    out = enh.enhance(np.zeros(8000, dtype=np.float32))
    assert out.audio.shape == (8000,)
