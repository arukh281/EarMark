"""Recurrent bodies for the Earmark network: a stacked GRU and a diagonal S4D-Lin SSM.

Both bodies map ``[B, T, dim] -> [B, T, dim]`` and expose the same interface:

* ``forward(x, state=None) -> (y, new_state)``: whole-sequence (training) mode.
* ``step(x, state) -> (y, new_state)``: one frame, ``[B, dim] -> [B, dim]``.
* ``state_shape()``: per-stream state shape (the batch dimension is prepended).
* ``init_state(batch, device, dtype)``: the all-zero initial state.

States are real, batch-first tensors so the streaming state can be flattened into
one fixed-size C struct. The SSM keeps its complex modes as trailing ``(re, im)``
pairs.

**GRU** follows PyTorch's ``nn.GRU`` gate convention (the C++ step must copy it)::

    r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
    h' = (1 - z) * n + z * h

**S4D-Lin** (Gu et al., 2022, "On the parameterization and initialization of diagonal
state space models"): per channel, ``M = state_dim // 2`` complex modes with
``A_m = -exp(log_a_real) + i * a_imag`` initialised to ``-1/2 + i pi m``, ``B = 1``, a
learned complex ``C`` and a learned step ``dt``. The zero-order-hold discretisation is::

    dA = exp(dt A),   dB = (dA - 1) / A
    h_t = dA * h_{t-1} + dB * u_t,     y_t = 2 Re(sum_m C_m h_t[m]) + D u_t

Training mode computes the same map as a causal convolution with kernel
``K[l] = 2 Re(sum_m C_m dB_m dA_m^l)`` via FFT. The kernel is built in float64 from the
very dA, dB and C that ``step`` uses, so the two modes agree to float rounding. The
FFT always runs outside autocast because half-precision cuFFT rejects non-power-of-two
sizes.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from earmark.model.dsp import complex_dtype, fp32_region


class RecurrentBody(nn.Module):
    """Interface shared by :class:`GRUBody` and :class:`SSMBody`."""

    dim: int
    layers: int

    def state_shape(self) -> tuple[int, ...]:
        """Per-stream state shape, without the leading batch dimension."""
        raise NotImplementedError

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """All-zero initial state ``[batch, *state_shape()]``."""
        return torch.zeros(batch, *self.state_shape(), device=device, dtype=dtype)

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Sequence mode: ``x`` ``[B, T, dim]`` -> ``(y [B, T, dim], final state)``."""
        raise NotImplementedError

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Streaming mode: ``x`` ``[B, dim]`` -> ``(y [B, dim], new state)``."""
        raise NotImplementedError


# -------------------------------------------------------------------------------- GRU


class GRUBody(RecurrentBody):
    """Stacked unidirectional GRU with ``hidden == dim``; state ``[B, layers, dim]``."""

    def __init__(self, dim: int, layers: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.layers = layers
        self.gru = nn.GRU(dim, dim, num_layers=layers, batch_first=True)

    def state_shape(self) -> tuple[int, ...]:
        return (self.layers, self.dim)

    def _h0(self, state: Tensor | None) -> Tensor | None:
        if state is None:
            return None
        return state.transpose(0, 1).to(self.gru.weight_ih_l0.dtype).contiguous()

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        y, h_n = self.gru(x, self._h0(state))
        return y, h_n.transpose(0, 1).to(self.gru.weight_ih_l0.dtype)

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        y, h_n = self.gru(x.unsqueeze(1), self._h0(state))
        return y[:, 0], h_n.transpose(0, 1).to(self.gru.weight_ih_l0.dtype)


# -------------------------------------------------------------------------------- SSM


class S4DLinCore(nn.Module):
    """Diagonal S4D-Lin SSM: one independent single-input single-output system per channel.

    Args:
        channels: number of channels ``H``.
        state_dim: real state size ``N``; the layer keeps ``N // 2`` complex modes per
            channel (the conjugate halves are implied by the ``2 Re(.)`` readout).
        dt_min, dt_max: range of the log-uniform initial step.
    """

    def __init__(
        self, channels: int, state_dim: int = 64, dt_min: float = 1e-3, dt_max: float = 1e-1
    ) -> None:
        super().__init__()
        if state_dim % 2:
            raise ValueError(f"state_dim must be even, got {state_dim}")
        self.channels = channels
        self.modes = state_dim // 2
        log_dt = torch.rand(channels) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.log_a_real = nn.Parameter(torch.full((channels, self.modes), math.log(0.5)))
        self.a_imag = nn.Parameter(
            math.pi * torch.arange(self.modes, dtype=torch.float32).repeat(channels, 1)
        )
        self.c = nn.Parameter(torch.randn(channels, self.modes, 2) * math.sqrt(0.5))
        self.d = nn.Parameter(torch.randn(channels))

    def _real_dtype(self) -> torch.dtype:
        return torch.float64 if self.log_dt.dtype == torch.float64 else torch.float32

    def discretise(self) -> tuple[Tensor, Tensor, Tensor]:
        """Zero-order-hold ``(dA, dB, C)``, each complex ``[channels, modes]``."""
        real = self._real_dtype()
        dt = self.log_dt.to(real).exp()
        a = torch.complex(-self.log_a_real.to(real).exp(), self.a_imag.to(real))
        da = torch.exp(a * dt[:, None])
        db = (da - 1.0) / a
        c = torch.view_as_complex(self.c.to(real).contiguous())
        return da, db, c

    def _kernel_dtype(self, device: torch.device) -> torch.dtype:
        # float64 kernels remove the drift between dA**l and l repeated multiplications;
        # MPS has no float64, so it falls back to complex64.
        return torch.complex64 if device.type == "mps" else torch.complex128

    def kernel(self, length: int) -> Tensor:
        """Convolution kernel ``K[h, l] = 2 Re(sum_m C dB dA^l)``, ``[channels, length]``."""
        da, db, c = self.discretise()
        kern = self._kernel_dtype(da.device)
        powers = self._powers(da.to(kern), length)
        return 2.0 * torch.einsum("hm,hml->hl", (c * db).to(kern), powers).real.to(da.real.dtype)

    @staticmethod
    def _powers(da: Tensor, length: int) -> Tensor:
        """``dA ** l`` for ``l = 0 .. length - 1`` as ``[channels, modes, length]``."""
        lags = torch.arange(length, device=da.device, dtype=da.real.dtype)
        return torch.exp(torch.log(da)[..., None] * lags)

    def forward(self, u: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Convolution mode.

        Args:
            u: real ``[B, T, channels]``.
            state: complex ``[B, channels, modes]`` state before frame 0, or ``None`` for zeros.

        Returns:
            ``(y [B, T, channels], final complex state [B, channels, modes])``.
        """
        with fp32_region(u.device):
            real = self._real_dtype()
            n_frames = u.shape[1]
            ut = u.to(real).transpose(1, 2)  # [B, H, T]
            da, db, c = self.discretise()
            kern = self._kernel_dtype(u.device)
            da_k = da.to(kern)
            powers = self._powers(da_k, n_frames)  # [H, M, T]
            kernel = 2.0 * torch.einsum("hm,hml->hl", (c * db).to(kern), powers).real.to(real)
            size = 2 * n_frames
            spectrum = torch.fft.rfft(ut, n=size) * torch.fft.rfft(kernel, n=size)
            y = torch.fft.irfft(spectrum, n=size)[..., :n_frames]
            y = y + self.d.to(real)[:, None] * ut
            new_state = torch.einsum("hml,bhl->bhm", powers.flip(-1), ut.to(kern)) * db.to(kern)
            if state is not None:
                start = state.to(kern)
                carry = powers * da_k[..., None]  # dA ** (t + 1)
                y = y + 2.0 * torch.einsum("hm,bhm,hml->bhl", c.to(kern), start, carry).real.to(real)
                new_state = new_state + carry[..., -1] * start
            return y.transpose(1, 2), new_state.to(complex_dtype(real))

    def step(self, u: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Exact recurrence for one frame: ``u`` ``[B, channels]``, complex ``state``."""
        with fp32_region(u.device):
            real = self._real_dtype()
            u = u.to(real)
            da, db, c = self.discretise()
            new_state = da * state.to(da.dtype) + db * u[..., None]
            y = 2.0 * (c * new_state).real.sum(-1) + self.d.to(real) * u
            return y, new_state

    def step_macs(self) -> int:
        """Real MACs per frame: 6 for the complex state update, 2 for the readout, plus D."""
        return self.channels * (8 * self.modes + 1)


class SSMBlock(nn.Module):
    """Pre-norm residual block: ``x + W_out(v * sigmoid(g))``, ``(v, g) = W_in(silu(SSM(LN(x))))``."""

    def __init__(self, dim: int, state_dim: int, expansion: int) -> None:
        super().__init__()
        self.expansion = expansion
        self.norm = nn.LayerNorm(dim)
        self.ssm = S4DLinCore(dim, state_dim)
        self.w_in = nn.Linear(dim, 2 * expansion)
        self.w_out = nn.Linear(expansion, dim)

    def _mix(self, x: Tensor, s: Tensor) -> Tensor:
        value, gate = self.w_in(F.silu(s)).chunk(2, dim=-1)
        return x + self.w_out(value * torch.sigmoid(gate))

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        s, new_state = self.ssm(self.norm(x), state)
        return self._mix(x, s), new_state

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        s, new_state = self.ssm.step(self.norm(x), state)
        return self._mix(x, s), new_state

    def step_macs(self) -> int:
        """MACs done by the block itself (the gating product); children count their own."""
        return self.expansion


class SSMBody(RecurrentBody):
    """Stack of :class:`SSMBlock` plus a final LayerNorm; state ``[B, layers, dim, modes, 2]``."""

    def __init__(self, dim: int, layers: int = 2, state_dim: int = 64, expansion: int = 176) -> None:
        super().__init__()
        self.dim = dim
        self.layers = layers
        self.modes = state_dim // 2
        self.blocks = nn.ModuleList(SSMBlock(dim, state_dim, expansion) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)

    def state_shape(self) -> tuple[int, ...]:
        return (self.layers, self.dim, self.modes, 2)

    @staticmethod
    def _layer_state(state: Tensor, index: int) -> Tensor:
        return torch.view_as_complex(state[:, index].contiguous())

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        new_states: list[Tensor] = []
        for index, block in enumerate(self.blocks):
            layer_state = None if state is None else self._layer_state(state, index)
            x, new_state = block(x, layer_state)
            new_states.append(torch.view_as_real(new_state))
        return self.norm(x), torch.stack(new_states, dim=1)

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        new_states: list[Tensor] = []
        for index, block in enumerate(self.blocks):
            x, new_state = block.step(x, self._layer_state(state, index))
            new_states.append(torch.view_as_real(new_state))
        return self.norm(x), torch.stack(new_states, dim=1)


__all__ = ["GRUBody", "RecurrentBody", "S4DLinCore", "SSMBlock", "SSMBody"]
