"""Validation on held-back mixer batches: loss terms, SI-SDR improvement and VAD AUC.

The batches come from the *training* pools under another seed, so they measure fit and
catch divergence during a run. They are not the dev score: Earmark-Synth dev, with
held-out speakers, is scored separately by the PyTorch-stream dev runner.

``vad_auc`` covers personal examples (a real enrolment embedding), the quantity behind the
week-1 "AUC >= 0.9" check; ``vad_auc_null`` covers NULL-embedding examples, whose labels
mark all speech.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import torch
from scipy.stats import rankdata

from earmark.data.mixer import Mixer
from earmark.model.earmark_net import EarmarkNet
from earmark.train.losses import TERMS, EarmarkLoss, align, reference_active, si_sdr, vad_pairs

#: Validation mixer seed = training seed + this, so its batches never repeat training's.
VAL_SEED_OFFSET: Final[int] = 1_000_003


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC by the Mann-Whitney statistic (ties get average ranks); NaN with one class."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positive = np.asarray(labels, dtype=np.float64).ravel() > 0.5
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def validate(
    model: EarmarkNet, loss_fn: EarmarkLoss, mixer: Mixer, n_batches: int, *, amp: bool = False
) -> dict[str, float]:
    """Mean loss terms, SI-SDR(i) on reference-active examples and VAD AUCs over
    ``mixer.batch(0 .. n_batches - 1)`` (the same batches every time)."""
    was_training = model.training
    model.train(False)
    device = next(model.parameters()).device
    totals: list[float] = []
    term_sums = dict.fromkeys(TERMS, 0.0)
    out_db: list[np.ndarray] = []
    in_db: list[np.ndarray] = []
    personal: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
    null: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
    try:
        for index in range(n_batches):
            batch = mixer.batch(index)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(batch["mixture"], batch["embedding"], null_mask=batch["null_embedding"])
            loss = loss_fn(out, batch)
            totals.append(float(loss.total))
            for name in TERMS:
                term_sums[name] += float(loss.terms[name])
            wav = out.wav.float()
            est, ref = align(wav, batch["target"].float())
            _, mix = align(wav, batch["mixture"].float())
            active = reference_active(ref, loss_fn.config.active_threshold_db)
            if bool(active.any()):
                out_db.append(si_sdr(est[active], ref[active]).cpu().numpy())
                in_db.append(si_sdr(mix[active], ref[active]).cpu().numpy())
            logits, labels = vad_pairs(out.vad_logit.float(), batch["vad"].float())
            is_null = batch["null_embedding"].bool()
            for mask, (scores, targets) in ((~is_null, personal), (is_null, null)):
                if bool(mask.any()):
                    scores.append(logits[mask].flatten().cpu().numpy())
                    targets.append(labels[mask].flatten().cpu().numpy())
    finally:
        model.train(was_training)
    result: dict[str, float] = {"loss": float(np.mean(totals)) if totals else math.nan}
    result.update({f"term_{name}": term_sums[name] / max(1, n_batches) for name in TERMS})
    if out_db:
        o, i = np.concatenate(out_db), np.concatenate(in_db)
        result.update(si_sdr_db=float(o.mean()), si_sdr_in_db=float(i.mean()), si_sdri_db=float((o - i).mean()))
    for key, (scores, targets) in (("vad_auc", personal), ("vad_auc_null", null)):
        result[key] = binary_auc(np.concatenate(scores), np.concatenate(targets)) if scores else math.nan
    return result


__all__ = ["VAL_SEED_OFFSET", "binary_auc", "validate"]
