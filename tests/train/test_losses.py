"""Loss sanity: every term's zero point, direction, asymmetry and masking; fp32 under
autocast; finite, non-zero gradients through each term and through the whole model."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from earmark import constants as C
from earmark.model import build
from earmark.train.losses import (
    TERMS,
    EarmarkLoss,
    LossConfig,
    absent_energy_db,
    align,
    compressed_spectral_terms,
    mr_stft_loss,
    reference_active,
    si_sdr,
    vad_pairs,
)

HOP = C.HOP_LENGTH
N = 50 * HOP  # 0.5 s


def speech_like(gen: torch.Generator, batch: int, n: int = N) -> torch.Tensor:
    """Harmonic bursts with random pitch and syllable phase, peak about 0.1."""
    t = torch.arange(n, dtype=torch.float64) / C.SAMPLE_RATE
    f0 = 100.0 + 150.0 * torch.rand(batch, 1, generator=gen, dtype=torch.float64)
    x = sum(torch.sin(2 * math.pi * k * f0 * t) / k for k in range(1, 12))
    phase = 2 * math.pi * torch.rand(batch, 1, generator=gen, dtype=torch.float64)
    env = torch.clamp(torch.sin(2 * math.pi * 4.0 * t + phase), min=0.0)
    return (0.03 * x * env).float()


def make_batch(seed: int = 0, batch: int = 4, silent: tuple[int, ...] = (2,)) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    target = speech_like(gen, batch)
    for row in silent:
        target[row] = 0.0
    noise = 0.03 * torch.randn(batch, N, generator=gen)
    frames = target.unfold(-1, C.WINDOW_LENGTH, HOP).square().mean(-1)
    return {"mixture": target + noise, "target": target, "vad": (frames > 1e-5).float()}


def delayed(x: torch.Tensor) -> torch.Tensor:
    """What the network outputs for an aligned estimate ``x`` (one hop late)."""
    return F.pad(x, (HOP, 0))[..., :-HOP]


def output(est: torch.Tensor, logits: torch.Tensor | None = None) -> SimpleNamespace:
    if logits is None:
        logits = torch.zeros(est.shape[0], est.shape[1] // HOP)
    return SimpleNamespace(wav=delayed(est), vad_logit=logits)


def perfect_logits(vad: torch.Tensor) -> torch.Tensor:
    """Model-frame logits whose frames 1.. match the labels confidently."""
    return F.pad((vad * 2.0 - 1.0) * 30.0, (1, 0))


def terms_of(result) -> dict[str, float]:
    return {name: float(result.terms[name]) for name in TERMS}


def test_defaults_follow_the_plan() -> None:
    cfg = LossConfig()
    assert (cfg.compression, cfg.asym_alpha, cfg.vad_weight) == (0.3, 10.0, 0.1)
    assert len(cfg.mr_stft_resolutions) >= 3


def test_a_perfect_output_scores_zero() -> None:
    batch = make_batch()
    result = EarmarkLoss()(output(batch["target"], perfect_logits(batch["vad"])), batch)
    for name, value in terms_of(result).items():
        assert value < 1e-3, name
    assert float(result.total) < 1e-3


def test_each_enhancement_term_grows_with_residual_noise() -> None:
    batch = make_batch()
    noise = batch["mixture"] - batch["target"]
    seen = [terms_of(EarmarkLoss()(output(batch["target"] + k * noise), batch)) for k in (0.03, 0.3, 1.0)]
    for name in ("mr_stft", "spectral", "asym", "sisdr", "absent"):
        values = [s[name] for s in seen]
        assert values[0] < values[1] < values[2], (name, values)


def test_asymmetric_term_weights_over_suppression_by_alpha_squared() -> None:
    ref = make_batch(silent=())["target"][:2]
    for gain, factor in ((0.5, 100.0), (2.0, 1.0)):  # over-suppressed everywhere, then under
        _, asym10 = compressed_spectral_terms(gain * ref, ref, alpha=10.0)
        _, asym1 = compressed_spectral_terms(gain * ref, ref, alpha=1.0)
        torch.testing.assert_close(asym10, factor * asym1, rtol=1e-5, atol=0.0)
    _, over = compressed_spectral_terms(0.5 * ref, ref)
    _, under = compressed_spectral_terms(2.0 * ref, ref)
    assert torch.all(over > 20.0 * under)  # -6 dB costs far more than +6 dB


def test_si_sdr_matches_the_textbook_formula_and_ignores_scale() -> None:
    gen = torch.Generator().manual_seed(3)
    ref = torch.randn(3, 4000, generator=gen, dtype=torch.float64)
    est = ref + 0.3 * torch.randn(3, 4000, generator=gen, dtype=torch.float64)
    got = si_sdr(est, ref)
    for b in range(3):
        s, y = ref[b] - ref[b].mean(), est[b] - est[b].mean()
        t = (y @ s) / (s @ s) * s
        assert float(got[b]) == pytest.approx(float(10 * torch.log10((t @ t) / ((y - t) @ (y - t)))), abs=1e-6)
    torch.testing.assert_close(si_sdr(5.0 * est, ref), got)
    capped = si_sdr(ref, ref, max_db=50.0)
    assert torch.all((capped > 49.9) & (capped <= 50.0))


@pytest.mark.parametrize("gain", [0.5, 2.0])
def test_mr_stft_of_a_scaled_signal_has_the_closed_form(gain: float) -> None:
    # Spectral convergence of g*x against x is |g - 1| and the log-magnitude term is |ln g|.
    ref = 0.1 * torch.randn(2, 8000, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    got = mr_stft_loss(gain * ref, ref)
    torch.testing.assert_close(got, torch.full_like(got, abs(gain - 1) + abs(math.log(gain))), rtol=1e-4, atol=1e-4)


def test_absent_term_measures_output_energy_against_the_mixture() -> None:
    mix = make_batch()["mixture"]
    full = torch.full((mix.shape[0],), 60.0)
    torch.testing.assert_close(absent_energy_db(mix, mix), full, atol=1e-3, rtol=0)
    torch.testing.assert_close(absent_energy_db(0.1 * mix, mix), full - 20.0, atol=1e-3, rtol=0)
    torch.testing.assert_close(absent_energy_db(torch.zeros_like(mix), mix), torch.zeros_like(full), atol=1e-6, rtol=0)


def test_terms_apply_only_to_their_examples() -> None:
    batch = make_batch(silent=(1, 3))
    target, mixture = batch["target"], batch["mixture"]
    assert reference_active(align(target, target)[1]).tolist() == [True, False, True, False]
    # Active examples perfect, silent ones pass the mixture: only the absent term fires.
    est = target.clone()
    est[[1, 3]] = mixture[[1, 3]]
    result = EarmarkLoss()(output(est), batch)
    assert float(result.terms["mr_stft"]) < 1e-3 and float(result.terms["sisdr"]) < 1e-3
    assert float(result.terms["absent"]) == pytest.approx(30.0, abs=0.01)  # 60 dB on half the batch
    # Silent examples silent, active ones pass the mixture: the absent term is zero.
    est = mixture.clone()
    est[[1, 3]] = 0.0
    result = EarmarkLoss()(output(est), batch)
    assert float(result.terms["absent"]) < 1e-6
    assert float(result.terms["mr_stft"]) > 0.1 and float(result.terms["sisdr"]) > 1.0
    assert (float(result.stats["n_active"]), float(result.stats["n_silent"])) == (2.0, 2.0)
    assert float(result.stats["si_sdri_db"]) == pytest.approx(0.0, abs=1e-4)  # output == input


def test_total_is_the_weighted_sum_of_the_terms() -> None:
    batch = make_batch()
    cfg = LossConfig()
    logits = torch.randn(4, N // HOP, generator=torch.Generator().manual_seed(5))
    est = batch["target"] + 0.5 * (batch["mixture"] - batch["target"])
    result = EarmarkLoss(cfg)(output(est, logits), batch)
    expected = sum(weight * float(result.terms[name]) for name, weight in cfg.weights().items())
    assert float(result.total) == pytest.approx(expected, rel=1e-5)
    bce = F.binary_cross_entropy_with_logits(logits[:, 1:], batch["vad"])
    torch.testing.assert_close(result.terms["vad"], bce)


def test_loss_runs_in_fp32_under_autocast() -> None:
    batch = make_batch()
    est = (batch["target"] + 0.1 * batch["mixture"]).to(torch.bfloat16)
    logits = torch.randn(4, N // HOP).to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        inside = EarmarkLoss()(output(est, logits), batch)
    outside = EarmarkLoss()(output(est.float(), logits.float()), batch)
    assert inside.total.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in inside.terms.values())
    assert torch.equal(inside.total, outside.total)


@pytest.mark.parametrize("term", TERMS)
def test_every_term_has_finite_nonzero_gradients(term: str) -> None:
    batch = make_batch()
    weights = {f"{name}_weight": 0.0 for name in TERMS}
    weights[f"{term}_weight"] = 1.0
    est = (batch["target"] + 0.3 * (batch["mixture"] - batch["target"])).requires_grad_(True)
    logits = torch.randn(4, N // HOP, requires_grad=True)
    result = EarmarkLoss(LossConfig(**weights))(SimpleNamespace(wav=delayed(est), vad_logit=logits), batch)
    result.total.backward()
    grad = logits.grad if term == "vad" else est.grad
    assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0


def test_silence_everywhere_gives_finite_values_and_gradients() -> None:
    zeros = torch.zeros(2, N)
    batch = {"mixture": zeros.clone(), "target": zeros.clone(), "vad": torch.zeros(2, N // HOP - 1)}
    est = torch.zeros(2, N, requires_grad=True)
    result = EarmarkLoss()(SimpleNamespace(wav=est, vad_logit=torch.zeros(2, N // HOP)), batch)
    result.total.backward()
    assert math.isfinite(float(result.total.detach())) and bool(torch.isfinite(est.grad).all())
    batch = make_batch()
    est = torch.zeros(4, N, requires_grad=True)
    result = EarmarkLoss()(SimpleNamespace(wav=est, vad_logit=torch.zeros(4, N // HOP)), batch)
    result.total.backward()
    assert math.isfinite(float(result.total.detach())) and bool(torch.isfinite(est.grad).all())


@pytest.mark.parametrize("config", ["S-GRU", "S-SSM"])
def test_gradients_reach_every_model_parameter(config: str) -> None:
    torch.manual_seed(0)
    net = build(config)
    batch = make_batch(silent=(2,))
    null = torch.tensor([False, True, False, False])
    emb = F.normalize(torch.randn(4, C.EMBEDDING_DIM), dim=-1)
    out = net(batch["mixture"], emb, null_mask=null)
    EarmarkLoss()(out, batch).total.backward()
    bad = [
        name
        for name, p in net.named_parameters()
        if p.grad is None or not bool(torch.isfinite(p.grad).all()) or float(p.grad.abs().sum()) == 0.0
    ]
    assert not bad, bad


def test_shape_mismatches_are_rejected() -> None:
    with pytest.raises(ValueError):
        align(torch.zeros(2, 320), torch.zeros(2, 480))
    with pytest.raises(ValueError):
        vad_pairs(torch.zeros(2, 10), torch.zeros(2, 10))


@pytest.mark.parametrize(
    "bad",
    [{"compression": 0.0}, {"asym_alpha": 0.5}, {"vad_weight": -1.0}, {"spectral_resolution": (256, 64, 512)},
     {"mr_stft_resolutions": ()}, {"absent_floor_db": 3.0}],
)  # fmt: skip
def test_loss_config_rejects_bad_settings(bad: dict) -> None:
    with pytest.raises(ValueError):
        LossConfig(**bad)
