"""EarmarkNet: streaming equals offline, causality, identity reconstruction, VAD range."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from earmark import constants as C
from earmark.model import (
    MODEL_FRAME_OFFSET,
    OUTPUT_DELAY_SAMPLES,
    EarmarkNet,
    build,
    config_for,
    stream_signal,
)

CONFIG_NAMES = ["S-GRU", "S-SSM", "M"]
HOP = C.HOP_LENGTH


def _signal(batch: int, n: int, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n, dtype=torch.float64) / C.SAMPLE_RATE
    tone = 0.3 * torch.sin(2 * math.pi * 220.0 * t) * (0.5 + 0.5 * torch.sin(2 * math.pi * 3.0 * t))
    noise = 0.1 * torch.randn(batch, n, generator=g, dtype=torch.float64)
    return (tone + noise).to(dtype)


def _embedding(batch: int, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(batch, C.EMBEDDING_DIM, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _net(name: str, dtype: torch.dtype = torch.float32) -> EarmarkNet:
    torch.manual_seed(0)
    return build(name).to(dtype).eval()


def _make_identity(net: EarmarkNet) -> None:
    """Gains of exactly 1 (sigmoid(30) rounds to 1 in fp32) and identity deep-filter taps."""
    with torch.no_grad():
        net.gain_head.weight.zero_()
        net.gain_head.bias.fill_(30.0)
        net.df_head.weight.zero_()
        net.df_head.bias.zero_()


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_streaming_equals_offline(name: str) -> None:
    net = _net(name)
    x = _signal(2, C.SAMPLE_RATE, seed=1)
    cond = net.condition(_embedding(2, seed=2), batch=2, null_mask=torch.tensor([False, True]))
    with torch.no_grad():
        offline = net(x, cond)
        wav, vad, state = stream_signal(net, x, cond)
    assert offline.wav.shape == x.shape and offline.vad.shape == (2, x.shape[1] // HOP)
    assert (offline.wav - wav).abs().max().item() < 1e-5
    assert (offline.vad - vad).abs().max().item() < 1e-5
    # erb_norm holds dB values near -50, so states get a relative tolerance as well.
    for field, value in offline.state.tensors().items():
        torch.testing.assert_close(value, state.tensors()[field], atol=1e-5, rtol=1e-5, msg=field)


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_offline_resumes_from_state(name: str) -> None:
    net = _net(name)
    x = _signal(1, 60 * HOP, seed=3)
    emb = _embedding(1, seed=4)
    with torch.no_grad():
        whole = net(x, emb)
        first = net(x[:, : 23 * HOP], emb)
        second = net(x[:, 23 * HOP :], emb, first.state)
    assert (torch.cat([first.wav, second.wav], dim=-1) - whole.wav).abs().max().item() < 1e-5
    assert (torch.cat([first.vad, second.vad], dim=-1) - whole.vad).abs().max().item() < 1e-5


@pytest.mark.parametrize("n0", [1599, 1600, 2345])
@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_causality_perturbation(name: str, n0: int) -> None:
    """Changing input sample n0 changes nothing before its hop, and its own hop at once."""
    net = _net(name, torch.float64)
    x = _signal(1, 30 * HOP, seed=5, dtype=torch.float64)
    emb = _embedding(1, seed=6, dtype=torch.float64)
    x2 = x.clone()
    x2[:, n0] += 0.5
    with torch.no_grad():
        a, b = net(x, emb), net(x2, emb)
    hop = n0 // HOP  # output hop k covers input up to sample (k + 1) * HOP
    wav_diff = (a.wav - b.wav).abs()
    vad_diff = (a.vad - b.vad).abs()
    assert wav_diff[:, : hop * HOP].max().item() < 1e-12
    assert vad_diff[:, :hop].max().item() < 1e-12
    assert wav_diff[:, hop * HOP : (hop + 1) * HOP].max().item() > 1e-6  # zero lookahead
    assert vad_diff[:, hop].item() > 0.0


@pytest.mark.parametrize("name", ["S-GRU", "S-SSM"])
def test_identity_heads_reconstruct_input(name: str) -> None:
    """Unit gains + identity deep filter: the model is the WOLA identity, delayed one hop."""
    net = _net(name)
    _make_identity(net)
    x = _signal(2, 40 * HOP, seed=7)
    with torch.no_grad():
        out = net(x, None)
        wav, _, _ = stream_signal(net, x, None)
    assert OUTPUT_DELAY_SAMPLES == HOP
    torch.testing.assert_close(out.wav[:, :HOP], torch.zeros(2, HOP), atol=1e-7, rtol=0)
    assert (out.wav[:, HOP:] - x[:, :-HOP]).abs().max().item() < 1e-5
    assert (wav - out.wav).abs().max().item() < 1e-5
    assert torch.equal(out.gains, torch.ones_like(out.gains))


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_vad_head_range(name: str) -> None:
    net = _net(name)
    signals = [
        torch.zeros(1, 20 * HOP),  # digital silence
        _signal(1, 20 * HOP, seed=8),
        torch.sign(_signal(1, 20 * HOP, seed=9)),  # full-scale clipped
        100.0 * _signal(1, 20 * HOP, seed=10),  # far beyond full scale
    ]
    with torch.no_grad():
        for x in signals:
            out = net(x, _embedding(1, seed=11))
            assert out.vad.shape == (1, 20)
            assert torch.isfinite(out.vad).all() and torch.isfinite(out.wav).all()
            assert (out.vad >= 0).all() and (out.vad <= 1).all()
            torch.testing.assert_close(out.vad, torch.sigmoid(out.vad_logit))
            _, vad, _ = net.step(x[:, :HOP], None, net.init_state(1))
            assert vad.shape == (1,) and 0.0 <= vad.item() <= 1.0


def test_vad_frames_align_with_contract_frames() -> None:
    assert MODEL_FRAME_OFFSET == 1
    net = _net("S-GRU")
    x = _signal(1, 10 * HOP, seed=12)
    with torch.no_grad():
        out = net(x, None)
    # model frame t analyses input samples [(t - 1) * HOP, (t + 1) * HOP): contract frame t - 1
    contract = torch.fft.rfft(x.double().unfold(-1, C.WINDOW_LENGTH, HOP) * torch.sin(
        torch.pi * torch.arange(C.WINDOW_LENGTH, dtype=torch.float64) / C.WINDOW_LENGTH))
    torch.testing.assert_close(out.noisy_spec[:, MODEL_FRAME_OFFSET:], contract.to(torch.complex64),
                               atol=1e-4, rtol=1e-5)


def test_null_embedding_and_mask() -> None:
    net = _net("S-GRU")
    x = _signal(2, 20 * HOP, seed=13)
    emb = _embedding(2, seed=14)
    with torch.no_grad():
        null = net(x, None)
        masked = net(x, emb, null_mask=torch.tensor([True, True]))
        mixed = net(x, emb, null_mask=torch.tensor([False, True]))
        personal = net(x, emb)
        scaled = net(x, 5.0 * emb)
    torch.testing.assert_close(null.wav, masked.wav)
    torch.testing.assert_close(mixed.wav[1], null.wav[1])
    torch.testing.assert_close(mixed.wav[0], personal.wav[0])
    torch.testing.assert_close(scaled.wav, personal.wav, atol=1e-6, rtol=1e-5)  # L2-normalised
    assert (personal.wav - null.wav).abs().max().item() > 1e-4  # FiLM is wired in


def test_trace_matches_between_forward_and_step() -> None:
    net = _net("S-SSM")
    x = _signal(1, 3 * HOP, seed=15)
    emb = _embedding(1, seed=16)
    fwd: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        net(x, emb, trace=fwd)
        state = net.init_state(1)
        for t in range(3):
            stp: dict[str, torch.Tensor] = {}
            _, _, state = net.step(x[:, t * HOP : (t + 1) * HOP], emb, state, trace=stp)
            assert stp.keys() == fwd.keys()
            for key, value in stp.items():
                if key == "wav":
                    ref = fwd[key][:, t * HOP : (t + 1) * HOP]
                else:
                    ref = fwd[key][:, t : t + 1]
                assert (value - ref).abs().max().item() < 1e-5, key


def test_backward_reaches_every_parameter() -> None:
    for name in ["S-GRU", "S-SSM"]:
        net = _net(name).train()
        x = _signal(2, 20 * HOP, seed=17)
        out = net(x, _embedding(2, seed=18), null_mask=torch.tensor([True, False]))
        loss = out.wav.square().mean() + F.binary_cross_entropy_with_logits(
            out.vad_logit, torch.ones_like(out.vad_logit)
        )
        loss.backward()
        for pname, param in net.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), f"{name}:{pname}"


@pytest.mark.parametrize("name", ["S-GRU", "S-SSM"])
def test_autocast_keeps_dsp_in_fp32(name: str) -> None:
    net = _net(name)
    x = _signal(1, 20 * HOP, seed=19)
    with torch.no_grad():
        ref = net(x, None)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = net(x, None)
    assert out.wav.dtype == torch.float32 and out.spec.dtype == torch.complex64
    assert torch.isfinite(out.wav).all()
    assert (out.wav - ref.wav).abs().max().item() < 0.1 * ref.wav.abs().max().item()


def test_build_and_input_validation() -> None:
    assert config_for("s_gru").name == "S-GRU"
    assert build("m", hidden=64).config.hidden == 64
    with pytest.raises(KeyError):
        build("XL")
    net = _net("S-GRU")
    with pytest.raises(ValueError):
        net(torch.zeros(1, HOP + 1))
    with pytest.raises(ValueError):
        net(torch.zeros(HOP))
    with pytest.raises(ValueError):
        net.step(torch.zeros(1, HOP - 1), None, net.init_state(1))
    with pytest.raises(ValueError):
        net(torch.zeros(1, HOP), torch.zeros(1, 7))
