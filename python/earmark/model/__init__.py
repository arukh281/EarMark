"""WOLA/ERB DSP, recurrent bodies, the Earmark network and its streaming step.

Typical use::

    from earmark.model import build, Streamer

    net = build("M")                      # or "S-GRU", "S-SSM", "M-256"
    out = net(x, emb)                     # offline: out.wav [B, N], out.vad [B, T]
    y, vad, state = net.step(hop, emb, net.init_state())  # one 160-sample hop

Modules: :mod:`~earmark.model.dsp`, :mod:`~earmark.model.bodies`,
:mod:`~earmark.model.earmark_net`, :mod:`~earmark.model.stream`,
:mod:`~earmark.model.macs`.
"""

from earmark.model.earmark_net import (
    CONFIGS,
    MODEL_FRAME_OFFSET,
    OUTPUT_DELAY_SAMPLES,
    PLAN_TARGETS,
    Conditioning,
    EarmarkConfig,
    EarmarkNet,
    EarmarkOutput,
    build,
    config_for,
)
from earmark.model.stream import StreamState, Streamer, state_size_bytes, stream_signal

__all__ = [
    "CONFIGS",
    "MODEL_FRAME_OFFSET",
    "OUTPUT_DELAY_SAMPLES",
    "PLAN_TARGETS",
    "Conditioning",
    "EarmarkConfig",
    "EarmarkNet",
    "EarmarkOutput",
    "StreamState",
    "Streamer",
    "build",
    "config_for",
    "state_size_bytes",
    "stream_signal",
]
