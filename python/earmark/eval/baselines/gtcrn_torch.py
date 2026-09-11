"""GTCRN baseline in PyTorch, using the official model code and checkpoints.

GTCRN (Rong et al., ICASSP 2024) is the small-model reference point of every Earmark table:
48.2 K parameters (including the fixed ERB matrices) and 33.0 MMAC/s. Two official
checkpoints exist: one trained on VoiceBank+DEMAND (published VB-test PESQ-WB 2.87, STOI 0.940,
SI-SNR 18.83 dB) and one trained on DNS3 (which, like Earmark, has not seen VB training data).

Licence: the upstream repository is MIT (Copyright (c) 2024 Rong Xiaobin), checked on
2026-09-11, so the model code is vendored in :mod:`gtcrn_model` with its notice in
``GTCRN_LICENSE.txt``. Checkpoints are not committed: ``scripts/fetch_eval_data.sh gtcrn``
downloads them into ``.cache/models/gtcrn`` and this adapter re-checks their SHA-256 pins.

Inference follows the official ``infer.py``: a 512-point STFT with a 256-sample hop and a
square-root Hann window (``torch.hann_window(512).pow(0.5)``, periodic, centred with reflect
padding), a whole-utterance forward pass, and the matching inverse STFT trimmed to the input
length. The output is time-aligned with the input, so no latency compensation is needed.
This is the ``pytorch-offline`` row; the ONNX streaming row is a separate adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from earmark import constants as C
from earmark.eval.assets import get_asset, verify_asset
from earmark.eval.baselines.gtcrn_model import GTCRN
from earmark.eval.systems import EnhancerOutput, SystemInfo

__all__ = ["CHECKPOINTS", "GTCRN_COMMIT", "GTCRN_MMAC_PER_S", "GTCRN_PARAMS", "GtcrnTorch", "build_model"]

GTCRN_COMMIT = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"
#: Parameters as published in the upstream README (includes the non-trainable ERB matrices).
GTCRN_PARAMS = 48_200
GTCRN_MMAC_PER_S = 33.0

Checkpoint = Literal["vb", "dns3"]

#: Checkpoint key -> (asset name in assets.tsv, training data description).
CHECKPOINTS: dict[str, tuple[str, str]] = {
    "vb": ("model_trained_on_vctk.tar", "VoiceBank+DEMAND train (28 speakers)"),
    "dns3": ("model_trained_on_dns3.tar", "DNS Challenge 3 (likely includes DEMAND noise and LibriVox speech)"),
}

N_FFT = 512
HOP = 256


def build_model() -> GTCRN:
    """An untrained GTCRN in eval mode (for tests and for loading checkpoints)."""
    return GTCRN().eval()


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a GTCRN state dict")
    return state


class GtcrnTorch:
    """GTCRN enhancer on CPU (or another torch device) implementing :class:`Enhancer`."""

    sample_rate: int = C.SAMPLE_RATE

    def __init__(
        self,
        checkpoint: Checkpoint = "vb",
        *,
        checkpoint_path: Path | None = None,
        verify_sha256: bool = True,
        device: str = "cpu",
    ) -> None:
        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"checkpoint must be one of {sorted(CHECKPOINTS)}, got {checkpoint!r}")
        asset_name, training = CHECKPOINTS[checkpoint]
        if verify_sha256:
            path = verify_asset(asset_name, checkpoint_path)
        else:
            path = checkpoint_path if checkpoint_path is not None else get_asset(asset_name).path()
        self.device = torch.device(device)
        self.model = build_model()
        self.model.load_state_dict(_load_state_dict(path), strict=True)
        self.model.to(self.device)
        self.window = torch.hann_window(N_FFT, device=self.device).pow(0.5)
        self.checkpoint_path = path
        self.info = SystemInfo(
            name=f"gtcrn-{checkpoint}",
            inference_path="pytorch-offline",
            params=GTCRN_PARAMS,
            mmac_per_s=GTCRN_MMAC_PER_S,
            training_data=training,
            licence="MIT (code and checkpoints, github.com/Xiaobin-Rong/gtcrn)",
            notes=f"official code and checkpoint at commit {GTCRN_COMMIT[:12]}; 512/256 sqrt-Hann STFT",
        )

    @torch.inference_mode()
    def enhance(self, mixture: NDArray[np.float32], embedding: NDArray[np.float32] | None = None) -> EnhancerOutput:
        """Enhance one 16 kHz mono utterance; ``embedding`` is ignored (GTCRN is not personal)."""
        x = torch.as_tensor(np.asarray(mixture, dtype=np.float32), device=self.device)
        if x.ndim != 1:
            raise ValueError(f"mixture must be 1-D, got shape {tuple(x.shape)}")
        n = x.shape[0]
        if n == 0:
            return EnhancerOutput(np.zeros(0, dtype=np.float32))
        spec = torch.stft(x, N_FFT, HOP, N_FFT, self.window, return_complex=True)  # (F, T)
        out = self.model(torch.view_as_real(spec)[None])[0]  # (F, T, 2)
        y = torch.istft(torch.view_as_complex(out.contiguous()), N_FFT, HOP, N_FFT, self.window, length=n)
        return EnhancerOutput(y.cpu().numpy().astype(np.float32, copy=False))
