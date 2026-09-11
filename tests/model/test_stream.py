"""Streaming state: fixed size, byte count, block-size independence."""

from __future__ import annotations

import pytest
import torch

from earmark import constants as C
from earmark.model import StreamState, Streamer, build, state_size_bytes
from earmark.model.stream import STATE_FIELDS

HOP = C.HOP_LENGTH

#: float32 bytes per stream: 832 fixed values plus the body state.
EXPECTED_BYTES = {
    "S-GRU": 4 * (832 + 2 * 128),
    "S-SSM": 4 * (832 + 2 * 128 * 32 * 2),
    "M": 4 * (832 + 2 * 384),
}


@pytest.mark.parametrize("name", list(EXPECTED_BYTES))
def test_state_is_fixed_size(name: str) -> None:
    torch.manual_seed(0)
    net = build(name).eval()
    state = net.init_state(1)
    assert tuple(state.tensors()) == STATE_FIELDS
    layout = state.layout()
    assert layout == net.state_layout()
    assert state_size_bytes(net) == state.per_stream_bytes() == EXPECTED_BYTES[name]
    x = 0.1 * torch.randn(1, 30 * HOP)
    with torch.no_grad():
        for t in range(30):
            _, _, state = net.step(x[:, t * HOP : (t + 1) * HOP], None, state)
    assert state.layout() == layout
    assert state.per_stream_bytes() == EXPECTED_BYTES[name]
    assert all(t.dtype == torch.float32 for t in state.tensors().values())


def test_initial_state_values() -> None:
    net = build("S-GRU")
    state = net.init_state(3)
    assert state.batch_size == 3
    assert state.erb_norm.shape == (3, C.ERB_BANDS) and state.spec_norm.shape == (3, C.DF_BINS)
    for name in ("in_buf", "ola_buf", "enc_erb_prev", "enc_df_prev", "df_hist", "body"):
        assert state.tensors()[name].abs().sum() == 0, name
    assert (state.spec_norm > 0).all()


def test_flatten_round_trip_and_select() -> None:
    torch.manual_seed(1)
    net = build("S-SSM")
    state = net.init_state(2).map(lambda t: t + torch.randn_like(t))
    flat = state.flatten()
    assert flat.shape == (2, state_size_bytes(net) // 4)
    back = StreamState.unflatten(flat, net.state_layout())
    for name, value in state.tensors().items():
        assert torch.equal(value, back.tensors()[name])
    one = state.select(1)
    assert one.batch_size == 1 and torch.equal(one.body[0], state.body[1])
    with pytest.raises(ValueError):
        StreamState.unflatten(flat[:, :-1], net.state_layout())


def test_step_does_not_mutate_state() -> None:
    net = build("S-GRU").eval()
    state = net.init_state(1)
    before = state.clone()
    with torch.no_grad():
        net.step(0.1 * torch.randn(1, HOP), None, state)
    for name, value in state.tensors().items():
        assert torch.equal(value, before.tensors()[name])


@pytest.mark.parametrize("name", ["S-GRU", "S-SSM"])
def test_streamer_accepts_any_block_size(name: str) -> None:
    torch.manual_seed(2)
    net = build(name).eval()
    emb = torch.randn(2, C.EMBEDDING_DIM)
    x = 0.2 * torch.randn(2, 50 * HOP)
    with torch.no_grad():
        ref = net(x, emb)
    streamer = Streamer(net, emb, batch=2)
    outs, vads = [], []
    start = 0
    for size in [1, 37, 160, 999, 3, 481, 2000, 1319]:
        out, vad = streamer.process(x[:, start : start + size])
        assert out.shape[-1] == vad.shape[-1] * HOP
        outs.append(out)
        vads.append(vad)
        start += size
    out, vad = streamer.process(x[:, start:])
    outs.append(out)
    vads.append(vad)
    wav = torch.cat(outs, dim=-1)
    assert wav.shape == x.shape
    assert (wav - ref.wav).abs().max().item() < 1e-5
    assert (torch.cat(vads, dim=-1) - ref.vad).abs().max().item() < 1e-5
    streamer.reset()
    again, _ = streamer.process(x[:, : 5 * HOP])
    assert (again - ref.wav[:, : 5 * HOP]).abs().max().item() < 1e-5
