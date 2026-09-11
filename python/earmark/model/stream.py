"""Streaming state and helpers for running :class:`~earmark.model.earmark_net.EarmarkNet` hop by hop.

The streaming state is one named, fixed-size struct. Its shapes depend only on the
model config, never on how much audio has been processed, and its fields are listed
in :data:`STATE_FIELDS` in the order the C++ ``state.h`` struct should use. Every
field is batch-first float32 (float64 when the model runs in double precision).

======================  ==================================  ==============================
Field                   Per-stream shape                    Meaning
======================  ==================================  ==============================
``in_buf``              ``[HOP_LENGTH]``                    last input hop (first half of the next analysis window)
``ola_buf``             ``[HOP_LENGTH]``                    second half of the last synthesis frame
``erb_norm``            ``[ERB_BANDS]``                     running mean of ERB log-power, dB
``spec_norm``           ``[DF_BINS]``                       running mean of ``|X|`` on the low band
``enc_erb_prev``        ``[1, ERB_BANDS]``                  previous ERB feature frame (first conv, tap ``t - 1``)
``enc_df_prev``         ``[2, DF_BINS]``                    previous low-band feature frame, ``(re, im)`` channels
``df_hist``             ``[DF_ORDER - 1, DF_BINS, 2]``      gained low band at ``t - 2``, ``t - 1`` as ``(re, im)``
``body``                GRU ``[layers, hidden]``            GRU hidden state per layer
                        SSM ``[layers, dim, modes, 2]``     S4D complex modes as ``(re, im)``
======================  ==================================  ==============================

All fields start at zero except ``erb_norm`` and ``spec_norm``, which start at
:func:`earmark.model.dsp.erb_norm_init` and :func:`earmark.model.dsp.unit_norm_init`.

One call of ``EarmarkNet.step`` consumes ``HOP_LENGTH`` new samples and emits
``HOP_LENGTH`` output samples that lag the input by ``HOP_LENGTH`` (plus one hop of
block buffering, ``ALGORITHMIC_LATENCY_SAMPLES`` in total).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from earmark import constants as C
from earmark.model.dsp import erb_norm_init, unit_norm_init

if TYPE_CHECKING:
    from earmark.model.earmark_net import Conditioning, EarmarkNet

#: State fields in C-struct order.
STATE_FIELDS: tuple[str, ...] = (
    "in_buf",
    "ola_buf",
    "erb_norm",
    "spec_norm",
    "enc_erb_prev",
    "enc_df_prev",
    "df_hist",
    "body",
)

#: Per-stream shapes of every field except ``body`` (which depends on the body type).
FIXED_LAYOUT: dict[str, tuple[int, ...]] = {
    "in_buf": (C.HOP_LENGTH,),
    "ola_buf": (C.HOP_LENGTH,),
    "erb_norm": (C.ERB_BANDS,),
    "spec_norm": (C.DF_BINS,),
    "enc_erb_prev": (1, C.ERB_BANDS),
    "enc_df_prev": (2, C.DF_BINS),
    "df_hist": (C.DF_ORDER - 1, C.DF_BINS, 2),
}


@dataclass(frozen=True)
class StreamState:
    """Fixed-size streaming state of one or more (batched) streams. See the module docstring."""

    in_buf: Tensor
    ola_buf: Tensor
    erb_norm: Tensor
    spec_norm: Tensor
    enc_erb_prev: Tensor
    enc_df_prev: Tensor
    df_hist: Tensor
    body: Tensor

    @classmethod
    def initial(
        cls,
        layout: Mapping[str, tuple[int, ...]],
        batch: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> StreamState:
        """Initial state for ``batch`` streams with the given per-stream ``layout``."""
        missing = set(STATE_FIELDS) - set(layout)
        if missing:
            raise ValueError(f"layout is missing fields {sorted(missing)}")
        dev = torch.device(device) if device is not None else None
        tensors = {
            name: torch.zeros(batch, *layout[name], device=dev, dtype=dtype) for name in STATE_FIELDS
        }
        tensors["erb_norm"] = erb_norm_init(dev, dtype).expand(batch, -1).clone()
        tensors["spec_norm"] = unit_norm_init(dev, dtype).expand(batch, -1).clone()
        return cls(**tensors)

    def tensors(self) -> dict[str, Tensor]:
        """Fields as an ordered ``{name: tensor}`` dict (C-struct order)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @property
    def batch_size(self) -> int:
        return self.in_buf.shape[0]

    def layout(self) -> dict[str, tuple[int, ...]]:
        """Per-stream shape of each field."""
        return {name: tuple(t.shape[1:]) for name, t in self.tensors().items()}

    def nbytes(self) -> int:
        """Bytes held by all fields for the whole batch."""
        return sum(t.numel() * t.element_size() for t in self.tensors().values())

    def per_stream_bytes(self) -> int:
        """Bytes per stream (``nbytes() / batch_size``)."""
        return self.nbytes() // self.batch_size

    def map(self, fn: Callable[[Tensor], Tensor]) -> StreamState:
        """New state with ``fn`` applied to every field."""
        return StreamState(**{name: fn(t) for name, t in self.tensors().items()})

    def detach(self) -> StreamState:
        """State detached from the autograd graph (for truncated BPTT or streaming)."""
        return self.map(lambda t: t.detach())

    def clone(self) -> StreamState:
        return self.map(lambda t: t.clone())

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> StreamState:
        return self.map(lambda t: t.to(device=device, dtype=dtype))

    def select(self, index: int | slice) -> StreamState:
        """Sub-batch of streams (a single ``int`` keeps the batch dimension)."""
        idx = slice(index, index + 1) if isinstance(index, int) else index
        return self.map(lambda t: t[idx])

    def flatten(self) -> Tensor:
        """``[B, state_numel]`` with the fields concatenated in C-struct order."""
        return torch.cat([t.reshape(self.batch_size, -1) for t in self.tensors().values()], dim=1)

    @classmethod
    def unflatten(cls, flat: Tensor, layout: Mapping[str, tuple[int, ...]]) -> StreamState:
        """Inverse of :meth:`flatten` for a given per-stream ``layout``."""
        batch = flat.shape[0]
        needed = sum(math.prod(layout[name]) for name in STATE_FIELDS)
        if flat.dim() != 2 or flat.shape[1] != needed:
            raise ValueError(f"flat state has shape {tuple(flat.shape)}, layout needs [B, {needed}]")
        out: dict[str, Tensor] = {}
        offset = 0
        for name in STATE_FIELDS:
            size = math.prod(layout[name])
            out[name] = flat[:, offset : offset + size].reshape(batch, *layout[name])
            offset += size
        return cls(**out)


def state_size_bytes(net: EarmarkNet, dtype: torch.dtype = torch.float32) -> int:
    """Bytes of streaming state per stream for ``net`` (float32 by default)."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    return itemsize * sum(math.prod(shape) for shape in net.state_layout().values())


def step(
    net: EarmarkNet,
    frame: Tensor,
    emb: Tensor | Conditioning | None,
    state: StreamState,
) -> tuple[Tensor, Tensor, StreamState]:
    """Process one hop. Functional alias for :meth:`EarmarkNet.step`."""
    return net.step(frame, emb, state)


@torch.no_grad()
def stream_signal(
    net: EarmarkNet,
    x: Tensor,
    emb: Tensor | Conditioning | None = None,
    state: StreamState | None = None,
) -> tuple[Tensor, Tensor, StreamState]:
    """Run ``x`` ``[B, N]`` (``N`` a multiple of ``HOP_LENGTH``) through ``net.step`` hop by hop.

    Returns ``(wav [B, N], vad [B, N // HOP_LENGTH], final state)``, which must match
    ``net(x, emb)`` to float rounding.
    """
    batch, n = x.shape
    if n % C.HOP_LENGTH:
        raise ValueError(f"signal length {n} is not a multiple of HOP_LENGTH={C.HOP_LENGTH}")
    cond = net.condition(emb, batch=batch) if not _is_conditioning(emb) else emb
    if state is None:
        state = net.init_state(batch, x.device, _real(x.dtype))
    outs: list[Tensor] = []
    vads: list[Tensor] = []
    for start in range(0, n, C.HOP_LENGTH):
        out, vad, state = net.step(x[:, start : start + C.HOP_LENGTH], cond, state)
        outs.append(out)
        vads.append(vad)
    return torch.cat(outs, dim=-1), torch.stack(vads, dim=-1), state


class Streamer:
    """Stateful wrapper that accepts audio in blocks of any size, like the engine's C API.

    ``set_embedding`` mirrors ``em_set_embedding`` (``None`` selects the learned NULL
    embedding, i.e. Denoise mode) and precomputes the FiLM conditioning once; ``process``
    buffers input into hops and calls ``EarmarkNet.step`` for each complete hop.
    """

    def __init__(
        self,
        net: EarmarkNet,
        emb: Tensor | None = None,
        *,
        batch: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.net = net
        self.batch = batch
        self.device = next(net.parameters()).device
        self.dtype = dtype
        self.set_embedding(emb)
        self.reset()

    @torch.no_grad()
    def set_embedding(self, emb: Tensor | None) -> None:
        """Set (or clear, with ``None``) the enrolment embedding without resetting state."""
        if emb is not None:
            emb = emb.to(self.device)
        self.conditioning = self.net.condition(emb, batch=self.batch)

    def reset(self) -> None:
        """Return to the initial state and drop buffered samples."""
        self.state = self.net.init_state(self.batch, self.device, self.dtype)
        self._pending = torch.zeros(self.batch, 0, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def process(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Feed ``x`` ``[B, n]``; returns ``(out [B, 160 k], vad [B, k])`` for the ``k`` completed hops."""
        buf = torch.cat([self._pending, x.to(device=self.device, dtype=self.dtype)], dim=-1)
        hops = buf.shape[-1] // C.HOP_LENGTH
        outs: list[Tensor] = []
        vads: list[Tensor] = []
        for index in range(hops):
            frame = buf[:, index * C.HOP_LENGTH : (index + 1) * C.HOP_LENGTH]
            out, vad, self.state = self.net.step(frame, self.conditioning, self.state)
            outs.append(out)
            vads.append(vad)
        self._pending = buf[:, hops * C.HOP_LENGTH :]
        if not outs:
            empty = buf.new_zeros(self.batch, 0)
            return empty, empty
        return torch.cat(outs, dim=-1), torch.stack(vads, dim=-1)


def _real(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _is_conditioning(obj: object) -> bool:
    from earmark.model.earmark_net import Conditioning

    return isinstance(obj, Conditioning)


__all__ = [
    "FIXED_LAYOUT",
    "STATE_FIELDS",
    "StreamState",
    "Streamer",
    "state_size_bytes",
    "step",
    "stream_signal",
]
