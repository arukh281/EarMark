"""Parameter and MAC counting for Earmark networks.

MACs are measured on the streaming path: one call of ``EarmarkNet.step`` for a single
stream, with the FiLM conditioning precomputed (the engine computes it once in
``em_set_embedding``, so it is reported separately as ``conditioning_macs``).

Counting conventions (real multiply-accumulates per frame):

* ``nn.Linear``, ``nn.Conv2d``, :class:`GroupedLinear`: one MAC per weight use; biases
  and activations are free.
* ``nn.GRU``: ``3 H (in + H)`` for the two matvecs plus ``3 H`` for the gate products
  (``r * (W_hn h)``, ``z * h``, ``(1 - z) * n``), per layer.
* ``nn.LayerNorm``: 4 per element (mean, variance, normalise, affine).
* FiLM: 1 per element. The SSM recurrence: 8 per complex mode (6 for the state
  update, 2 for the readout) plus 1 for ``D``. The SSM gate product: 1 per element.
* DSP (WOLA, ERB features, normalisation, gains, deep filter) is an analytic estimate
  reported separately as ``dsp_macs_per_frame``; the plan's MMAC/s targets cover the
  network only.

Built-in layers are counted with forward hooks while ``step`` actually runs. Custom
modules declare ``step_macs()``, meaning the MACs they perform themselves per frame,
excluding their children.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from earmark import constants as C
from earmark.model.earmark_net import CONFIGS, EarmarkNet, build
from earmark.model.stream import state_size_bytes


@dataclass(frozen=True)
class ComplexityReport:
    """Measured size of one network."""

    config: str
    params: int
    params_by_part: dict[str, int]
    macs_per_frame: int
    macs_by_part: dict[str, int]
    conditioning_macs: int
    dsp_macs_per_frame: int
    state_bytes: int
    frame_rate_hz: int = C.FRAME_RATE_HZ

    @property
    def mmacs_per_s(self) -> float:
        """Network MMAC/s at the contract frame rate (the plan's metric)."""
        return self.macs_per_frame * self.frame_rate_hz / 1e6

    @property
    def total_mmacs_per_s(self) -> float:
        """Network plus DSP MMAC/s."""
        return (self.macs_per_frame + self.dsp_macs_per_frame) * self.frame_rate_hz / 1e6

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["mmacs_per_s"] = self.mmacs_per_s
        out["total_mmacs_per_s"] = self.total_mmacs_per_s
        return out


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters (all, or only those with ``requires_grad``)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def params_by_part(net: nn.Module) -> dict[str, int]:
    """Parameters per top-level child module."""
    parts: dict[str, int] = defaultdict(int)
    for name, param in net.named_parameters():
        parts[name.split(".", 1)[0]] += param.numel()
    return dict(parts)


def _builtin_macs(module: nn.Module, inputs: tuple[object, ...], output: object) -> int:
    """MACs of one call of a built-in layer, or 0 for anything else."""
    x = inputs[0] if inputs else None
    if isinstance(module, nn.Linear) and isinstance(x, Tensor):
        return x.numel() // module.in_features * module.in_features * module.out_features
    if isinstance(module, nn.Conv1d | nn.Conv2d) and isinstance(output, Tensor):
        per_output = module.in_channels // module.groups * math.prod(module.kernel_size)
        return output.numel() * per_output
    if isinstance(module, nn.GRU) and isinstance(x, Tensor):
        steps = x.shape[0] * x.shape[1]
        hidden = module.hidden_size
        total = 0
        for layer in range(module.num_layers):
            width = module.input_size if layer == 0 else hidden
            total += 3 * hidden * (width + hidden) + 3 * hidden
        return steps * total
    if isinstance(module, nn.LayerNorm) and isinstance(x, Tensor):
        return 4 * x.numel()
    return 0


def _part(name: str) -> str:
    return name.split(".", 1)[0] if name else "<root>"


def _hooked_macs(net: nn.Module, fn: Callable[[], object]) -> dict[str, int]:
    """Run ``fn`` with hooks on every module; MACs of built-in layers by top-level part."""
    parts: dict[str, int] = defaultdict(int)
    handles = []
    for name, module in net.named_modules():

        def hook(mod: nn.Module, inputs: tuple[object, ...], output: object, part: str = _part(name)) -> None:
            macs = _builtin_macs(mod, inputs, output)
            if macs:
                parts[part] += macs

        handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            fn()
    finally:
        for handle in handles:
            handle.remove()
    return dict(parts)


def count_step_macs(net: EarmarkNet) -> dict[str, int]:
    """Network MACs of one ``step`` for one stream, by top-level part."""
    device = net.window.device
    cond = net.condition(None, batch=1)
    state = net.init_state(1, device)
    frame = torch.zeros(1, C.HOP_LENGTH, device=device)
    parts = _hooked_macs(net, lambda: net.step(frame, cond, state))
    for name, module in net.named_modules():
        own = getattr(module, "step_macs", None)
        if callable(own) and name.split(".", 1)[0] != "conditioner":
            parts[_part(name)] = parts.get(_part(name), 0) + int(own())
    return parts


def count_conditioning_macs(net: EarmarkNet) -> int:
    """MACs to turn one embedding into FiLM parameters (once per embedding change)."""
    emb = torch.zeros(1, C.EMBEDDING_DIM, device=net.window.device)
    return sum(_hooked_macs(net, lambda: net.condition(emb, batch=1)).values())


def dsp_macs_per_frame() -> int:
    """Analytic estimate of the per-frame DSP MACs around the network."""
    n = C.N_FFT
    rfft = round(2 * n * math.log2(n))  # rough real-FFT estimate, same for the inverse
    window = 2 * C.WINDOW_LENGTH  # analysis and synthesis windows
    erb = 3 * C.N_BINS + 3 * C.ERB_BANDS  # |X|^2, band means, running mean and scaling
    unit = 5 * C.DF_BINS  # |X|, running mean, complex divide
    gains = 2 * C.N_BINS  # complex spectrum times real gain
    deep_filter = 4 * C.DF_ORDER * C.DF_BINS  # complex MACs
    return 2 * rfft + window + erb + unit + gains + deep_filter


def complexity(net: EarmarkNet) -> ComplexityReport:
    """Measure parameters, per-frame MACs and state size of ``net``."""
    macs = count_step_macs(net)
    return ComplexityReport(
        config=net.config.name,
        params=count_params(net),
        params_by_part=params_by_part(net),
        macs_per_frame=sum(macs.values()),
        macs_by_part=macs,
        conditioning_macs=count_conditioning_macs(net),
        dsp_macs_per_frame=dsp_macs_per_frame(),
        state_bytes=state_size_bytes(net),
    )


def report_all(names: list[str] | None = None) -> dict[str, ComplexityReport]:
    """Complexity reports for the named configs (all configs by default)."""
    return {name: complexity(build(name)) for name in (names or list(CONFIGS))}


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m earmark.model.macs [--json] [CONFIG ...]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="*", help="config names (default: all)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args(argv)
    reports = report_all(args.configs or None)
    if args.json:
        print(json.dumps({k: v.as_dict() for k, v in reports.items()}, indent=2))
        return
    print(f"{'config':8} {'params':>10} {'MAC/frame':>10} {'MMAC/s':>8} {'+DSP':>7} {'state B':>8}")
    for name, rep in reports.items():
        print(
            f"{name:8} {rep.params:>10,} {rep.macs_per_frame:>10,} {rep.mmacs_per_s:>8.1f} "
            f"{rep.total_mmacs_per_s:>7.1f} {rep.state_bytes:>8,}"
        )


if __name__ == "__main__":
    main()


__all__ = [
    "ComplexityReport",
    "complexity",
    "count_conditioning_macs",
    "count_params",
    "count_step_macs",
    "dsp_macs_per_frame",
    "params_by_part",
    "report_all",
]
