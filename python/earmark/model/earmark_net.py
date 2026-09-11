"""The Earmark network: personal voice isolation plus a personal-VAD head.

Signal flow per 10 ms hop (all shapes per stream)::

    x hop (160) ─► WOLA analysis (320-sample sqrt-Hann window, 161 bins)
      ├─► ERB log-power (32), causal mean-normalised ──► grouped freq conv branch ─┐
      └─► low band bins 0..63, unit-normalised (64 x re/im) ► grouped freq conv ───┤
                                                                                   ▼
               GroupedLinear + ReLU ► FiLM(emb) ► body (GRU | S4D-Lin SSM) ► FiLM(emb)
                                                     │                          │
                                     VAD head: sigmoid(linear) ◄┘   ┌───────────┴───────────┐
                                                           32 sigmoid ERB gains   DF coefs (3 taps x 64 bins)
    X ─► X * expand(gains) ─► order-3 deep filter on bins 0..63 ─► inverse DFT, window, overlap-add ─► y hop (160)

Conventions the engine and trainer must follow:

* Output ``wav`` lags the input by :data:`OUTPUT_DELAY_SAMPLES` (one hop): the model's
  first frame is the previous (initially zero) hop plus the current hop. Model frame
  ``t`` is contract frame ``t - MODEL_FRAME_OFFSET`` of the input, so ``vad[:, 1:]``
  aligns with contract-frame labels ``labels[:, :T - 1]``.
* The first encoder conv has a 2-frame causal time kernel (frames ``t - 1`` and ``t``);
  all later layers are frame-local. The encoder frequency convs pad one bin each side.
* The embedding is L2-normalised; the learned NULL embedding (``conditioner.null_embedding``)
  replaces it in Denoise mode. FiLM is ``x * (1 + dg) + b`` with
  ``(dg, b) = Linear(tanh(Linear(e)))``, computed once per embedding change.
* DF head output (384 values) is laid out ``[bin, tap, re/im]``. The coefficients are
  ``tanh(raw)`` plus 1 on tap 0 (residual parameterisation, so zero output is the
  identity filter).
* Deep filter input is the ERB-gained spectrum (``gains, then deep filter``); bins
  ``>= DF_BINS`` keep the gained spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from earmark import constants as C
from earmark.model import dsp
from earmark.model.bodies import GRUBody, RecurrentBody, SSMBody
from earmark.model.stream import FIXED_LAYOUT, StreamState

#: Output samples lag input samples by this much (one hop).
OUTPUT_DELAY_SAMPLES: Final[int] = C.HOP_LENGTH

#: Model frame ``t`` analyses contract frame ``t - MODEL_FRAME_OFFSET`` of the input.
MODEL_FRAME_OFFSET: Final[int] = 1

#: Deep-filter head outputs: DF_BINS x DF_ORDER x (re, im).
DF_OUTPUTS: Final[int] = C.DF_BINS * C.DF_ORDER * 2


# ----------------------------------------------------------------------------- config


@dataclass(frozen=True)
class EarmarkConfig:
    """Architecture hyper-parameters of one Earmark variant."""

    name: str
    body: Literal["gru", "ssm"]
    hidden: int
    body_layers: int = 2
    enc_channels: int = 16
    enc_groups: int = 2
    enc_proj_groups: int = 1
    film_rank: int = 32
    df_head_groups: int = 2
    ssm_state: int = 64
    ssm_expansion: int = 176


@dataclass(frozen=True)
class PlanTarget:
    """Size targets from the plan (``None`` when the plan does not fix a number)."""

    params: int | None
    mmacs_per_s: float | None


CONFIGS: Final[dict[str, EarmarkConfig]] = {
    "S-GRU": EarmarkConfig(name="S-GRU", body="gru", hidden=128, df_head_groups=2),
    "S-SSM": EarmarkConfig(name="S-SSM", body="ssm", hidden=128, df_head_groups=2),
    "M": EarmarkConfig(name="M", body="gru", hidden=384, df_head_groups=4),
    # Fallback if week-1 throughput makes ~20 GPU-h insufficient for M.
    "M-256": EarmarkConfig(name="M-256", body="gru", hidden=256, df_head_groups=4),
}

PLAN_TARGETS: Final[dict[str, PlanTarget]] = {
    "S-GRU": PlanTarget(params=300_000, mmacs_per_s=30.0),
    "S-SSM": PlanTarget(params=None, mmacs_per_s=30.0),  # "matched MACs" with S-GRU
    "M": PlanTarget(params=2_000_000, mmacs_per_s=200.0),
    "M-256": PlanTarget(params=1_000_000, mmacs_per_s=None),
}


def config_for(name: str) -> EarmarkConfig:
    """Look up a config by name; case and ``_``/``-`` are ignored (``s_gru`` == ``S-GRU``)."""
    key = name.strip().upper().replace("_", "-")
    if key not in CONFIGS:
        raise KeyError(f"unknown Earmark config {name!r}; choose from {sorted(CONFIGS)}")
    return CONFIGS[key]


def build(config: str | EarmarkConfig, **overrides: object) -> EarmarkNet:
    """Build a randomly initialised :class:`EarmarkNet` from a config name or object.

    ``overrides`` replace config fields, e.g. ``build("M", hidden=256)``.
    """
    cfg = config_for(config) if isinstance(config, str) else config
    if overrides:
        cfg = replace(cfg, **overrides)  # type: ignore[arg-type]
    return EarmarkNet(cfg)


# ----------------------------------------------------------------------------- layers


class GroupedLinear(nn.Module):
    """Block-diagonal linear layer: ``groups`` independent ``in/g -> out/g`` matvecs.

    ``weight`` is ``[groups, out/g, in/g]`` (row-major matvec per group); input feature
    ``g * in/g + i`` feeds group ``g`` and output ``g * out/g + o`` comes from it.
    """

    def __init__(self, in_features: int, out_features: int, groups: int = 1) -> None:
        super().__init__()
        if in_features % groups or out_features % groups:
            raise ValueError(f"{in_features} -> {out_features} is not divisible into {groups} groups")
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        bound = 1.0 / (in_features // groups) ** 0.5
        weight = torch.empty(groups, out_features // groups, in_features // groups)
        self.weight = nn.Parameter(weight.uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))

    def forward(self, x: Tensor) -> Tensor:
        grouped = x.unflatten(-1, (self.groups, self.in_features // self.groups))
        return torch.einsum("...gi,goi->...go", grouped, self.weight).flatten(-2) + self.bias

    def step_macs(self) -> int:
        return self.in_features * self.out_features // self.groups


class FiLM(nn.Module):
    """Feature-wise affine modulation ``x * scale + shift`` with per-stream ``[B, dim]`` params."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        if x.dim() == 3:
            scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
        return x * scale + shift

    def step_macs(self) -> int:
        return self.dim


class FreqConvBranch(nn.Module):
    """Grouped frequency convolutions over one feature map ``[B, C_in, T, F]``.

    Layer 0: ``(2 x 3)`` kernel (frames ``t - 1, t``; 3 bins), stride 1, ungrouped.
    Layers 1..n: ``(1 x 3)`` kernel, frequency stride 2, ``groups`` channel groups.
    ReLU after every layer. Output ``[B, T, channels * F / 2**n]`` (channel-major).
    """

    def __init__(self, in_channels: int, channels: int, in_bins: int, downsamples: int, groups: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_bins = in_bins
        self.first = nn.Conv2d(in_channels, channels, kernel_size=(2, 3), padding=(0, 1))
        self.down = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), groups=groups)
            for _ in range(downsamples)
        )
        self.out_bins = in_bins >> downsamples
        self.out_features = channels * self.out_bins

    def forward(self, x: Tensor, prev: Tensor) -> Tensor:
        """``x`` ``[B, C_in, T, F]``, ``prev`` ``[B, C_in, F]`` (the frame before ``x[:, :, 0]``)."""
        y = F.relu(self.first(torch.cat([prev.unsqueeze(2), x], dim=2)))
        for conv in self.down:
            y = F.relu(conv(y))
        batch, channels, frames, bins = y.shape
        return y.permute(0, 2, 1, 3).reshape(batch, frames, channels * bins)


@dataclass(frozen=True)
class Conditioning:
    """FiLM parameters derived from one embedding per stream (all ``[B, ...]``)."""

    embedding: Tensor
    pre_scale: Tensor
    pre_shift: Tensor
    post_scale: Tensor
    post_shift: Tensor

    @property
    def batch_size(self) -> int:
        return self.embedding.shape[0]


class Conditioner(nn.Module):
    """Maps a speaker embedding (or the learned NULL embedding) to FiLM parameters."""

    def __init__(self, emb_dim: int, rank: int, hidden: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.null_embedding = nn.Parameter(F.normalize(torch.randn(emb_dim), dim=0))
        self.proj = nn.Linear(emb_dim, rank)
        self.pre = nn.Linear(rank, 2 * hidden)
        self.post = nn.Linear(rank, 2 * hidden)

    def resolve(self, emb: Tensor | None, batch: int | None, null_mask: Tensor | None) -> Tensor:
        """Unit-norm ``[B, emb_dim]`` embedding; NULL where ``emb`` is ``None`` or ``null_mask``."""
        null = self.null_embedding
        if emb is None:
            resolved = null.expand(batch or 1, -1)
        else:
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            if emb.shape[-1] != self.emb_dim:
                raise ValueError(f"embedding must have {self.emb_dim} values, got {emb.shape[-1]}")
            if batch is not None and emb.shape[0] != batch:
                if emb.shape[0] != 1:
                    raise ValueError(f"embedding batch {emb.shape[0]} does not match {batch}")
                emb = emb.expand(batch, -1)
            resolved = emb.to(null.dtype)
            if null_mask is not None:
                mask = null_mask.to(device=resolved.device, dtype=torch.bool).reshape(-1, 1)
                resolved = torch.where(mask, null.expand_as(resolved), resolved)
        return F.normalize(resolved, dim=-1, eps=1e-8)

    def forward(self, embedding: Tensor) -> Conditioning:
        code = torch.tanh(self.proj(embedding))
        pre_scale, pre_shift = self.pre(code).chunk(2, dim=-1)
        post_scale, post_shift = self.post(code).chunk(2, dim=-1)
        return Conditioning(embedding, 1.0 + pre_scale, pre_shift, 1.0 + post_scale, post_shift)


# ----------------------------------------------------------------------------- output


@dataclass(frozen=True)
class EarmarkOutput:
    """Result of :meth:`EarmarkNet.forward` for ``T = N / HOP_LENGTH`` frames."""

    wav: Tensor  #: ``[B, N]`` enhanced audio, lagging the input by OUTPUT_DELAY_SAMPLES
    vad: Tensor  #: ``[B, T]`` personal-VAD probability
    vad_logit: Tensor  #: ``[B, T]`` pre-sigmoid VAD logit (use with BCE-with-logits)
    gains: Tensor  #: ``[B, T, ERB_BANDS]`` sigmoid ERB gains
    df_coefs: Tensor  #: complex ``[B, T, DF_ORDER, DF_BINS]`` effective deep-filter taps
    spec: Tensor  #: complex ``[B, T, N_BINS]`` enhanced spectrum
    noisy_spec: Tensor  #: complex ``[B, T, N_BINS]`` input spectrum (model framing)
    state: StreamState  #: state after the last frame; pass it back to continue


class _NetOut(NamedTuple):
    vad_logit: Tensor
    gains: Tensor
    df_coefs: Tensor
    enc_erb_prev: Tensor
    enc_df_prev: Tensor
    body_state: Tensor


def _record(trace: dict[str, Tensor] | None, name: str, value: Tensor) -> None:
    if trace is not None:
        trace[name] = value.detach()


# ------------------------------------------------------------------------------ model


class EarmarkNet(nn.Module):
    """Earmark enhancer and personal-VAD network.

    Entry points:

    * ``forward(x, emb=None, state=None, *, null_mask=None, trace=None) -> EarmarkOutput``
      for whole signals ``[B, N]`` (``N`` a multiple of ``HOP_LENGTH``).
    * ``step(frame, emb, state, *, trace=None) -> (out_hop [B, 160], vad [B], new_state)``
      for one hop, using the exact recurrences.

    ``emb`` is a ``[B, 256]`` (or ``[256]``) embedding, ``None`` for the NULL embedding,
    or a precomputed :class:`Conditioning` from :meth:`condition`. Pass ``trace={}`` to
    collect named intermediate tensors (for golden files).
    """

    def __init__(self, config: EarmarkConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        channels = config.enc_channels
        self.erb_enc = FreqConvBranch(1, channels, C.ERB_BANDS, 2, config.enc_groups)
        self.df_enc = FreqConvBranch(2, channels, C.DF_BINS, 3, config.enc_groups)
        self.enc_proj = GroupedLinear(
            self.erb_enc.out_features + self.df_enc.out_features, hidden, config.enc_proj_groups
        )
        self.conditioner = Conditioner(C.EMBEDDING_DIM, config.film_rank, hidden)
        self.film_pre = FiLM(hidden)
        self.body: RecurrentBody
        if config.body == "gru":
            self.body = GRUBody(hidden, config.body_layers)
        elif config.body == "ssm":
            self.body = SSMBody(hidden, config.body_layers, config.ssm_state, config.ssm_expansion)
        else:
            raise ValueError(f"unknown body {config.body!r}")
        self.film_post = FiLM(hidden)
        self.vad_head = nn.Linear(hidden, 1)
        self.gain_head = nn.Linear(hidden, C.ERB_BANDS)
        self.df_head = GroupedLinear(hidden, DF_OUTPUTS, config.df_head_groups)

        identity = torch.zeros(C.DF_ORDER, C.DF_BINS, 2)
        identity[0, :, 0] = 1.0
        self.register_buffer("df_identity", identity, persistent=False)
        self.register_buffer("window", dsp.sqrt_hann_window(), persistent=False)
        self.register_buffer("erb_fb", dsp.erb_matrix(), persistent=False)
        self.register_buffer("erb_index", dsp.erb_band_index(), persistent=False)

    # ---------------------------------------------------------------- helpers

    @property
    def param_dtype(self) -> torch.dtype:
        return self.enc_proj.weight.dtype

    def state_layout(self) -> dict[str, tuple[int, ...]]:
        """Per-stream shape of every :class:`StreamState` field, in C-struct order."""
        return {**FIXED_LAYOUT, "body": self.body.state_shape()}

    def init_state(
        self,
        batch: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> StreamState:
        """Initial streaming state for ``batch`` streams."""
        if device is None:
            device = self.window.device
        return StreamState.initial(self.state_layout(), batch, device, dtype)

    def condition(
        self,
        emb: Tensor | None = None,
        *,
        batch: int | None = None,
        null_mask: Tensor | None = None,
    ) -> Conditioning:
        """FiLM conditioning for an embedding (``None`` = NULL); ``null_mask[b]`` forces NULL."""
        return self.conditioner(self.conditioner.resolve(emb, batch, null_mask))

    def _conditioning(
        self, emb: Tensor | Conditioning | None, batch: int, null_mask: Tensor | None
    ) -> Conditioning:
        if isinstance(emb, Conditioning):
            if emb.batch_size != batch:
                raise ValueError(f"conditioning batch {emb.batch_size} does not match {batch}")
            return emb
        return self.condition(emb, batch=batch, null_mask=null_mask)

    def df_coefficients(self, raw: Tensor) -> Tensor:
        """Head output ``[..., DF_OUTPUTS]`` -> complex taps ``[..., DF_ORDER, DF_BINS]``."""
        real = dsp.dsp_dtype(raw.dtype)
        taps = torch.tanh(raw.to(real)).unflatten(-1, (C.DF_BINS, C.DF_ORDER, 2)).transpose(-3, -2)
        return torch.view_as_complex((taps + self.df_identity.to(real)).contiguous())

    def enhance(
        self, spec: Tensor, gains: Tensor, df_coefs: Tensor, df_hist: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Apply ERB gains, then the deep filter on bins ``< DF_BINS``.

        ``spec`` complex ``[B, T, N_BINS]``, ``gains`` ``[B, T, ERB_BANDS]``, ``df_coefs``
        complex ``[B, T, DF_ORDER, DF_BINS]``, ``df_hist`` complex ``[B, DF_ORDER-1, DF_BINS]``.
        Returns ``(enhanced spectrum, new df history)``.
        """
        gained = spec * dsp.erb_expand(gains.to(spec.real.dtype), self.erb_index)
        low, history = dsp.deep_filter(gained[..., : C.DF_BINS], df_coefs, df_hist)
        return torch.cat([low, gained[..., C.DF_BINS :]], dim=-1), history

    def _network(
        self,
        erb_feat: Tensor,
        spec_feat: Tensor,
        cond: Conditioning,
        state: StreamState,
        body_state: Tensor | None,
        *,
        streaming: bool,
        trace: dict[str, Tensor] | None,
    ) -> _NetOut:
        """Encoder, FiLM, body and heads for features ``[B, T, 32]`` and complex ``[B, T, 64]``."""
        dtype = self.param_dtype
        erb_in = erb_feat.to(dtype).unsqueeze(1)
        df_in = torch.view_as_real(spec_feat).to(dtype).permute(0, 3, 1, 2)
        enc_erb = self.erb_enc(erb_in, state.enc_erb_prev.to(dtype))
        enc_df = self.df_enc(df_in, state.enc_df_prev.to(dtype))
        enc = F.relu(self.enc_proj(torch.cat([enc_erb, enc_df], dim=-1)))
        h = self.film_pre(enc, cond.pre_scale, cond.pre_shift)
        if streaming:
            assert body_state is not None
            body_out, new_body = self.body.step(h[:, 0], body_state.to(dtype))
            body_out = body_out.unsqueeze(1)
        else:
            body_out, new_body = self.body(h, None if body_state is None else body_state.to(dtype))
        vad_logit = self.vad_head(body_out).squeeze(-1)
        z = self.film_post(body_out, cond.post_scale, cond.post_shift)
        gains = torch.sigmoid(self.gain_head(z))
        raw_df = self.df_head(z)
        for name, value in (
            ("enc_erb", enc_erb),
            ("enc_df", enc_df),
            ("enc", enc),
            ("film_pre", h),
            ("body_out", body_out),
            ("vad_logit", vad_logit),
            ("film_post", z),
            ("gains", gains),
            ("df_raw", raw_df),
        ):
            _record(trace, name, value)
        return _NetOut(
            vad_logit=vad_logit,
            gains=gains,
            df_coefs=self.df_coefficients(raw_df),
            enc_erb_prev=erb_in[:, :, -1],
            enc_df_prev=df_in[:, :, -1],
            body_state=new_body,
        )

    # ------------------------------------------------------------ entry points

    def forward(
        self,
        x: Tensor,
        emb: Tensor | Conditioning | None = None,
        state: StreamState | None = None,
        *,
        null_mask: Tensor | None = None,
        trace: dict[str, Tensor] | None = None,
    ) -> EarmarkOutput:
        """Offline (whole-sequence) pass over ``x`` ``[B, N]``, ``N`` a multiple of ``HOP_LENGTH``.

        Equivalent to ``N / HOP_LENGTH`` calls of :meth:`step` starting from ``state``
        (the initial state if ``None``). Runs the FFTs, features, masking and synthesis in
        fp32 even under autocast.
        """
        if x.dim() != 2:
            raise ValueError(f"expected [batch, samples], got shape {tuple(x.shape)}")
        batch, n = x.shape
        if n == 0 or n % C.HOP_LENGTH:
            raise ValueError(
                f"length {n} must be a positive multiple of HOP_LENGTH={C.HOP_LENGTH}; "
                "use earmark.model.dsp.pad_to_hop"
            )
        real = dsp.dsp_dtype(x.dtype)
        cond = self._conditioning(emb, batch, null_mask)
        fresh = state is None
        if state is None:
            state = self.init_state(batch, x.device, real)
        with dsp.fp32_region(x.device):
            window = self.window.to(real)
            samples = torch.cat([state.in_buf.to(real), x.to(real)], dim=-1)
            spec = dsp.analysis_frame(dsp.frame_signal(samples), window)
            erb_feat, erb_norm = dsp.erb_features(spec, state.erb_norm.to(real), self.erb_fb.to(real))
            spec_feat, spec_norm = dsp.unit_norm_features(spec[..., : C.DF_BINS], state.spec_norm.to(real))
        _record(trace, "spec_in", spec)
        _record(trace, "erb_feat", erb_feat)
        _record(trace, "spec_feat", spec_feat)
        net = self._network(
            erb_feat, spec_feat, cond, state, None if fresh else state.body, streaming=False, trace=trace
        )
        with dsp.fp32_region(x.device):
            df_hist = torch.view_as_complex(state.df_hist.to(real).contiguous())
            spec_out, df_hist = self.enhance(spec, net.gains, net.df_coefs, df_hist)
            wav, ola = dsp.overlap_add(dsp.synthesis_frame(spec_out, window), state.ola_buf.to(real))
            vad_logit = net.vad_logit.to(real)
        _record(trace, "spec_out", spec_out)
        _record(trace, "wav", wav)
        new_state = StreamState(
            in_buf=samples[:, -C.HOP_LENGTH :],
            ola_buf=ola,
            erb_norm=erb_norm,
            spec_norm=spec_norm,
            enc_erb_prev=net.enc_erb_prev,
            enc_df_prev=net.enc_df_prev,
            df_hist=torch.view_as_real(df_hist),
            body=net.body_state,
        )
        return EarmarkOutput(
            wav=wav,
            vad=torch.sigmoid(vad_logit),
            vad_logit=vad_logit,
            gains=net.gains.to(real),
            df_coefs=net.df_coefs,
            spec=spec_out,
            noisy_spec=spec,
            state=new_state,
        )

    def step(
        self,
        frame: Tensor,
        emb: Tensor | Conditioning | None,
        state: StreamState,
        *,
        trace: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, StreamState]:
        """Streaming pass over one hop ``frame`` ``[B, HOP_LENGTH]``.

        Returns ``(out [B, HOP_LENGTH], vad [B], new_state)``; ``state`` is not modified.
        """
        if frame.dim() != 2 or frame.shape[-1] != C.HOP_LENGTH:
            raise ValueError(f"expected [batch, {C.HOP_LENGTH}], got shape {tuple(frame.shape)}")
        batch = frame.shape[0]
        real = dsp.dsp_dtype(frame.dtype)
        cond = self._conditioning(emb, batch, None)
        with dsp.fp32_region(frame.device):
            window = self.window.to(real)
            samples = torch.cat([state.in_buf.to(real), frame.to(real)], dim=-1)
            spec = dsp.analysis_frame(samples, window)
            erb_feat, erb_norm = dsp.erb_features_step(spec, state.erb_norm.to(real), self.erb_fb.to(real))
            spec_feat, spec_norm = dsp.unit_norm_features_step(
                spec[..., : C.DF_BINS], state.spec_norm.to(real)
            )
            spec, erb_feat, spec_feat = spec[:, None], erb_feat[:, None], spec_feat[:, None]
        _record(trace, "spec_in", spec)
        _record(trace, "erb_feat", erb_feat)
        _record(trace, "spec_feat", spec_feat)
        net = self._network(erb_feat, spec_feat, cond, state, state.body, streaming=True, trace=trace)
        with dsp.fp32_region(frame.device):
            df_hist = torch.view_as_complex(state.df_hist.to(real).contiguous())
            spec_out, df_hist = self.enhance(spec, net.gains, net.df_coefs, df_hist)
            out, ola = dsp.overlap_add(dsp.synthesis_frame(spec_out, window), state.ola_buf.to(real))
            vad = torch.sigmoid(net.vad_logit.to(real))[:, 0]
        _record(trace, "spec_out", spec_out)
        _record(trace, "wav", out)
        new_state = StreamState(
            in_buf=samples[:, C.HOP_LENGTH :],
            ola_buf=ola,
            erb_norm=erb_norm,
            spec_norm=spec_norm,
            enc_erb_prev=net.enc_erb_prev,
            enc_df_prev=net.enc_df_prev,
            df_hist=torch.view_as_real(df_hist),
            body=net.body_state,
        )
        return out, vad, new_state


__all__ = [
    "CONFIGS",
    "DF_OUTPUTS",
    "MODEL_FRAME_OFFSET",
    "OUTPUT_DELAY_SAMPLES",
    "PLAN_TARGETS",
    "Conditioner",
    "Conditioning",
    "EarmarkConfig",
    "EarmarkNet",
    "EarmarkOutput",
    "FiLM",
    "FreqConvBranch",
    "GroupedLinear",
    "PlanTarget",
    "build",
    "config_for",
]
