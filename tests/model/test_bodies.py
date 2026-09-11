"""GRU and S4D-Lin SSM bodies: sequence mode equals the step recurrence."""

from __future__ import annotations

import pytest
import torch

from earmark.model.bodies import GRUBody, RecurrentBody, S4DLinCore, SSMBody


def _noise(shape: tuple[int, ...], seed: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _step_all(body: RecurrentBody, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    state = body.init_state(x.shape[0], dtype=x.dtype)
    ys = []
    for t in range(x.shape[1]):
        y, state = body.step(x[:, t], state)
        ys.append(y)
    return torch.stack(ys, dim=1), state


def _bodies(dtype: torch.dtype) -> list[RecurrentBody]:
    torch.manual_seed(0)
    return [GRUBody(24, layers=2).to(dtype), SSMBody(24, layers=2, state_dim=16, expansion=32).to(dtype)]


@pytest.mark.parametrize(("dtype", "tol"), [(torch.float32, 1e-5), (torch.float64, 1e-10)])
@pytest.mark.parametrize("kind", [0, 1], ids=["gru", "ssm"])
def test_forward_equals_step(kind: int, dtype: torch.dtype, tol: float) -> None:
    body = _bodies(dtype)[kind]
    x = _noise((2, 300, body.dim), seed=1, dtype=dtype)
    with torch.no_grad():
        y_seq, s_seq = body(x)
        y_step, s_step = _step_all(body, x)
    assert y_seq.shape == x.shape
    assert s_seq.shape == (2, *body.state_shape())
    assert (y_seq - y_step).abs().max().item() < tol
    assert (s_seq - s_step).abs().max().item() < tol


@pytest.mark.parametrize("kind", [0, 1], ids=["gru", "ssm"])
def test_forward_resumes_from_state(kind: int) -> None:
    body = _bodies(torch.float64)[kind]
    x = _noise((2, 120, body.dim), seed=2, dtype=torch.float64)
    with torch.no_grad():
        whole, s_whole = body(x)
        first, carry = body(x[:, :47])
        second, s_second = body(x[:, 47:], carry)
    torch.testing.assert_close(torch.cat([first, second], dim=1), whole, atol=1e-10, rtol=0)
    torch.testing.assert_close(s_second, s_whole, atol=1e-10, rtol=0)


@pytest.mark.parametrize("kind", [0, 1], ids=["gru", "ssm"])
def test_body_is_causal(kind: int) -> None:
    body = _bodies(torch.float64)[kind]
    x = _noise((1, 80, body.dim), seed=3, dtype=torch.float64)
    x2 = x.clone()
    # A random vector, not a constant: LayerNorm (SSM blocks) removes a constant shift.
    x2[:, 50] += _noise((body.dim,), seed=4, dtype=torch.float64)
    with torch.no_grad():
        diff = (body(x2)[0] - body(x)[0]).abs()
    assert diff[:, :50].max().item() < 1e-12
    assert diff[:, 50].max().item() > 1e-6


def test_ssm_kernel_is_the_impulse_response() -> None:
    torch.manual_seed(1)
    core = S4DLinCore(6, state_dim=16).double()
    length = 64
    impulse = torch.zeros(1, 6, dtype=torch.float64)
    state = torch.zeros(1, 6, 8, dtype=torch.complex128)
    response = []
    with torch.no_grad():
        for t in range(length):
            u = impulse + (1.0 if t == 0 else 0.0)
            y, state = core.step(u, state)
            response.append(y[0] - (core.d if t == 0 else 0.0))
        kernel = core.kernel(length)
    torch.testing.assert_close(torch.stack(response, dim=1), kernel, atol=1e-12, rtol=0)


def test_ssm_init_is_s4d_lin() -> None:
    core = S4DLinCore(4, state_dim=8)
    da, db, c = core.discretise()
    assert da.shape == db.shape == c.shape == (4, 4)
    assert torch.allclose(core.log_a_real.exp(), torch.full((4, 4), 0.5))
    assert torch.allclose(core.a_imag[0], torch.pi * torch.arange(4.0))
    assert (da.abs() < 1).all()  # stable
    with pytest.raises(ValueError):
        S4DLinCore(4, state_dim=7)


def test_ssm_gradients_flow_through_fft_mode() -> None:
    torch.manual_seed(2)
    body = SSMBody(8, layers=1, state_dim=8, expansion=16)
    y, _ = body(torch.randn(2, 40, 8))
    y.square().mean().backward()
    for name, param in body.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
