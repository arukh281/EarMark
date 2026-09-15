"""The training mixer: personal-enhancement examples rendered on any torch device.

Recipe (M1 plan; every number is a :class:`MixerConfig` field):

* **Target.** One or two utterances of an enrolled speaker, drawn from that speaker's
  ``target`` pool (chapters or sessions disjoint from the enrolment pool), with random
  pauses and a random active level ``U(-35, -15)`` dBFS. Reverberant with p=0.5; the
  training reference keeps the direct path plus the first 50 ms of the RIR.
* **Noise** at SNR ``U(-5, 20)`` dB.
* **Interferer** with p=0.6 at SIR ``U(-5, 10)`` dB: another human talker (1/2), a Kokoro
  agent voice from the training voices (1/3), or TV/podcast speech over a MUSAN music bed
  (1/6). All agent voices and half of the others go through a loudspeaker chain
  (band-limit, EQ, soft clipping, a small room, optional residual-echo gain dips), the
  way TTS or a TV leaks from laptop speakers after AEC.
* **Target absent** with p=0.15; **clipping** with p=0.1.
* **NULL embedding** with p=0.2: the embedding is zeros and ``null_embedding`` is set (the
  model substitutes its learned NULL vector), and the reference becomes all speech,
  denoised (target plus interferer speech, without music, noise or chain distortion).
* **VAD labels** from the direct-path target energy by the contract rule
  (:mod:`.labels`): utterance peak from the manifest, -40 dB, 50 ms hangover. With the
  NULL embedding they cover all reference speech.

Levels. SNR is ``10 log10(P_t / P_n)`` and SIR ``10 log10(P_t / P_i)``, where ``P_t`` and
``P_i`` are the active powers (:func:`.labels.active_power`) of the target and the
interferer as they appear in the mixture, after RIR and chain, and ``P_n`` is the noise's
mean power. In target-absent examples the drawn target level still sets the noise and
interferer levels. If the mixture would exceed ``max_peak`` every signal is scaled down
together, so the drawn SNR and SIR still hold.

Clipping (probability ``p_clip``) comes last and acts on ``mixture`` alone. In a clipped
example (``clipped`` True) the mixture no longer equals the sum of its components, so
``snr_db`` and ``sir_db`` are the levels *before* clipping. In every other example
``mixture`` is exactly ``target_mix + interferer_mix + noise_mix``.

Determinism. All random choices come from a NumPy generator seeded by
``(seed, batch_index)`` on the CPU; the device does only arithmetic. Batch ``k`` is a pure
function of the pools, the config, the seed and ``k``, so resuming needs only
:meth:`Mixer.state_dict` (``seed``, ``next_index``).

Batch (``B = batch_size``, ``T = example_seconds * 16000``, ``F = 1 + (T - 320) // 160``):

=====================  =========  =======  ==============================================
key                    shape      dtype    meaning
=====================  =========  =======  ==============================================
``mixture``            [B, T]     float32  model input
``target``             [B, T]     float32  training reference (zeros when absent)
``vad``                [B, F]     float32  personal-VAD frame labels in {0, 1}
``embedding``          [B, 256]   float32  enrolment embedding (zeros when NULL)
``null_embedding``     [B]        bool     use the learned NULL vector
``target_present``     [B]        bool     the enrolled speaker talks in this example
``interferer_kind``    [B]        int64    0 none, 1 human, 2 agent, 3 TV
``loudspeaker``        [B]        bool     interferer went through the loudspeaker chain
``reverberant``        [B]        bool     target convolved with an RIR
``clipped``            [B]        bool     mixture clipping applied
``snr_db``             [B]        float32  drawn SNR
``sir_db``             [B]        float32  drawn SIR (NaN without interferer)
``target_level_db``    [B]        float32  drawn target active level (dBFS, before scaling)
``speaker_index``      [B]        int64    row of the embedding table
=====================  =========  =======  ==============================================

Alignment with the network (:mod:`earmark.model.earmark_net`): ``target`` is sample-aligned
with ``mixture``, but the network's output lags its input by one hop
(``OUTPUT_DELAY_SAMPLES``), so compare ``out[:, 160:]`` with ``target[:, :-160]``. ``vad``
uses contract framing (frame ``t`` covers samples ``[160 t, 160 t + 320)``), and model
frame ``t`` analyses contract frame ``t - 1`` (``MODEL_FRAME_OFFSET``). On a whole
``T``-sample example the model gives ``T / 160`` frames against ``F = T / 160 - 1``
labels, so ``vad_logit[:, 1:]`` pairs with ``vad`` frame for frame.

With ``return_components=True`` the batch also holds ``target_mix``, ``interferer_mix``,
``noise_mix`` (as mixed, [B, T]), ``target_direct`` and ``interferer_reference`` ([B, T])
and ``output_gain`` ([B], the common down-scaling).
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from earmark import constants as C
from earmark.data.agent_voice import ENGLISH_VOICES, split_voices
from earmark.data.embeddings import SpeakerEmbeddings
from earmark.data.labels import (
    active_power,
    apply_hangover,
    db_to_power_ratio,
    frame_energy,
    num_frames,
)
from earmark.data.noise_filter import VB_HELDOUT_ENVIRONMENTS, mentions_excluded
from earmark.data.shards import POOL_TARGET, ShardedCorpus
from earmark.data.splits import VB_TEST_SPEAKERS, SplitLeakError

__all__ = [
    "INTERFERER_AGENT",
    "INTERFERER_HUMAN",
    "INTERFERER_NAMES",
    "INTERFERER_NONE",
    "INTERFERER_TV",
    "HeldOut",
    "Mixer",
    "MixerConfig",
    "MixerPools",
    "fft_convolve",
    "level_gain",
    "load_training_pools",
    "loudspeaker_response",
    "next_fft_size",
    "residual_echo_envelope",
    "rir_windows",
    "segment_peak_frames",
    "soft_clip",
    "synthetic_rir",
    "validate_training_pools",
]

INTERFERER_NONE = 0
INTERFERER_HUMAN = 1
INTERFERER_AGENT = 2
INTERFERER_TV = 3
INTERFERER_NAMES: tuple[str, ...] = ("none", "human", "agent", "tv")
_TINY = 1e-20


# --------------------------------------------------------------------------- DSP helpers


def next_fft_size(n: int) -> int:
    """Smallest power of two >= ``n``."""
    return 1 << max(0, int(n) - 1).bit_length()


def fft_convolve(x: torch.Tensor, h: torch.Tensor, n_fft: int | None = None) -> torch.Tensor:
    """Causal linear convolution of ``x [..., T]`` with ``h [..., L]``, truncated to ``T``."""
    total = x.shape[-1] + h.shape[-1] - 1
    n = n_fft or next_fft_size(total)
    if n < total:
        raise ValueError(f"n_fft={n} is shorter than the linear convolution ({total})")
    y = torch.fft.irfft(torch.fft.rfft(x, n=n) * torch.fft.rfft(h, n=n), n=n)
    return y[..., : x.shape[-1]]


def rir_windows(
    h: torch.Tensor, peak: torch.Tensor, *, direct_samples: int, early_samples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct-path and early parts of RIRs ``[B, L]``: taps up to peak + N samples."""
    idx = torch.arange(h.shape[-1], device=h.device)
    p = peak.to(h.device).long().unsqueeze(-1)
    return h * (idx <= p + direct_samples), h * (idx <= p + early_samples)


def synthetic_rir(
    rng: np.random.Generator,
    rt60_s: float,
    length: int,
    *,
    drr_db: tuple[float, float] = (0.0, 10.0),
    sample_rate: int = C.SAMPLE_RATE,
) -> np.ndarray:
    """Exponentially decaying noise RIR with a unit direct path at tap 0.

    The tail energy sits ``U(drr_db)`` dB below the direct path. Used when no RIR corpus is
    given, and for the loudspeaker chain's small room when there are no small-room RIRs.
    """
    n = np.arange(length)
    tail = rng.standard_normal(length) * np.exp(-6.907755 * n / (rt60_s * sample_rate))
    tail[0] = 0.0
    energy = float(np.sum(tail**2))
    drr = rng.uniform(*drr_db)
    if energy > 0:
        tail *= math.sqrt(10.0 ** (-drr / 10.0) / energy)
    tail[0] = 1.0
    return tail.astype(np.float32)


def loudspeaker_response(
    freqs: torch.Tensor,
    highpass_hz: torch.Tensor,
    lowpass_hz: torch.Tensor,
    eq_centre_hz: torch.Tensor,
    eq_gain_db: torch.Tensor,
    eq_width_oct: torch.Tensor,
) -> torch.Tensor:
    """Zero-phase magnitude ``[B, nF]`` of a small loudspeaker: 2nd-order high-pass,
    4th-order low-pass (Butterworth magnitudes) and Gaussian EQ bumps in log-frequency.
    """
    f = freqs.clamp_min(1.0).unsqueeze(0)
    mag = torch.rsqrt(1.0 + (highpass_hz.unsqueeze(-1) / f) ** 4)
    mag = mag * torch.rsqrt(1.0 + (f / lowpass_hz.unsqueeze(-1)) ** 8)
    lf = torch.log2(f).unsqueeze(-1)
    z = (lf - torch.log2(eq_centre_hz).unsqueeze(1)) / eq_width_oct.unsqueeze(1)
    bumps = (eq_gain_db.unsqueeze(1) * torch.exp(-0.5 * z.square())).sum(-1)
    mag = mag * torch.pow(10.0, bumps / 20.0)
    mag[..., 0] = 0.0
    return mag


def soft_clip(x: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
    """``(p / d) tanh(d x / p)`` with ``p`` the per-row peak: unit small-signal gain,
    peaks compressed harder as the drive ``d`` grows."""
    peak = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    d = drive.to(x.dtype).clamp_min(1e-3).unsqueeze(-1)
    return peak / d * torch.tanh(d * x / peak)


def level_gain(power: torch.Tensor, level_db: torch.Tensor) -> torch.Tensor:
    """Amplitude gain taking a signal of ``power`` to ``level_db``; 0 for silent signals."""
    target = torch.pow(10.0, level_db / 10.0)
    gain = torch.sqrt(target / power.clamp_min(_TINY))
    return torch.where(power > _TINY, gain, torch.zeros_like(gain))


def residual_echo_envelope(ctrl_db: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Linear interpolation of per-control-point gains in dB ``[B, K]`` to ``[B, T]``."""
    gain = torch.pow(10.0, ctrl_db / 20.0).unsqueeze(1)
    return F.interpolate(gain, size=num_samples, mode="linear", align_corners=True)[:, 0]


def segment_peak_frames(
    segments: Sequence[tuple[float, int, int]], n_frames: int, *, delay: int = 0
) -> np.ndarray:
    """Per-frame utterance peak energy for placed segments ``(peak_energy, dst, length)``.

    A frame gets the peak of every segment it overlaps (the largest if several); frames
    that overlap none get ``inf``, so they can never be active. ``delay`` shifts the
    segments (the RIR's direct-path delay).
    """
    out = np.full(n_frames, -np.inf)
    for peak, dst, length in segments:
        a = int(dst) + int(delay)
        b = a + int(length)
        t0 = max(0, (a - C.WINDOW_LENGTH) // C.HOP_LENGTH + 1)
        t1 = min(n_frames, -(-b // C.HOP_LENGTH))
        if t1 > t0:
            out[t0:t1] = np.maximum(out[t0:t1], float(peak))
    out[np.isneginf(out)] = np.inf
    return out


# --------------------------------------------------------------------------- config and pools


@dataclass(frozen=True)
class MixerConfig:
    """Every probability, range and constant of the training mix (defaults = the M1 plan)."""

    example_seconds: float = 4.0
    batch_size: int = 32
    # target
    p_target_absent: float = 0.15
    p_null: float = 0.2
    target_level_db: tuple[float, float] = (-35.0, -15.0)
    p_pause: float = 0.5
    lead_seconds: tuple[float, float] = (0.0, 1.0)
    pause_seconds: tuple[float, float] = (0.3, 1.5)
    min_segment_seconds: float = 0.5
    # acoustics
    p_reverb: float = 0.5
    direct_ms: float = 2.5
    early_ms: float = 50.0
    rir_max_seconds: float = 1.0
    p_same_room: float = 0.7
    synthetic_rt60_s: tuple[float, float] = (0.2, 0.8)
    # noise
    snr_db: tuple[float, float] = (-5.0, 20.0)
    # interferer
    p_interferer: float = 0.6
    sir_db: tuple[float, float] = (-5.0, 10.0)
    interferer_weights: tuple[float, float, float] = (1 / 2, 1 / 3, 1 / 6)  # human, agent, TV
    p_loudspeaker_other: float = 0.5
    music_bed_db: tuple[float, float] = (-20.0, -5.0)
    # loudspeaker chain
    highpass_hz: tuple[float, float] = (150.0, 500.0)
    lowpass_hz: tuple[float, float] = (3500.0, 7500.0)
    eq_bumps: int = 3
    eq_db: float = 6.0
    eq_octaves: tuple[float, float] = (0.3, 1.5)
    drive: tuple[float, float] = (1.0, 4.0)
    small_room_rt60_s: tuple[float, float] = (0.1, 0.35)
    p_residual_echo: float = 0.3
    residual_echo_db: tuple[float, float] = (6.0, 20.0)
    residual_echo_step_ms: float = 100.0
    # output
    p_clip: float = 0.1
    clip_fraction: tuple[float, float] = (0.3, 0.9)
    max_peak: float = 0.99

    def __post_init__(self) -> None:
        for name in (
            "p_target_absent", "p_null", "p_pause", "p_reverb", "p_same_room", "p_interferer",
            "p_loudspeaker_other", "p_residual_echo", "p_clip",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value} is not a probability")
        for name in (
            "target_level_db", "lead_seconds", "pause_seconds", "synthetic_rt60_s", "snr_db",
            "sir_db", "music_bed_db", "highpass_hz", "lowpass_hz", "eq_octaves", "drive",
            "small_room_rt60_s", "residual_echo_db", "clip_fraction",
        ):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name}={getattr(self, name)} has low > high")
        if self.num_samples < C.WINDOW_LENGTH:
            raise ValueError("examples must hold at least one frame")
        if self.batch_size < 1 or self.eq_bumps < 1:
            raise ValueError("batch_size and eq_bumps must be positive")
        if len(self.interferer_weights) != 3 or min(self.interferer_weights) < 0:
            raise ValueError("interferer_weights needs three non-negative weights")
        if not 0.0 < self.max_peak <= 1.0:
            raise ValueError("max_peak must be in (0, 1]")
        if self.min_segment_seconds * C.SAMPLE_RATE < C.WINDOW_LENGTH:
            raise ValueError("min_segment_seconds must cover at least one frame")

    @property
    def num_samples(self) -> int:
        return int(round(self.example_seconds * C.SAMPLE_RATE))

    @property
    def num_frames(self) -> int:
        return num_frames(self.num_samples)


def _test_agent_voices() -> frozenset[str]:
    """The held-out Kokoro voices of the default split (:func:`.agent_voice.split_voices`)."""
    return frozenset(v for v, s in split_voices(ENGLISH_VOICES).items() if s == "test")


@dataclass(frozen=True)
class HeldOut:
    """What training must never contain (checked by :func:`validate_training_pools`).

    The defaults cover everything known without data: the VoiceBank-DEMAND speakers, the
    DEMAND environments behind VB's test noise, the test Kokoro voices, real RIRs,
    held-out music and ESC-50. Add the dev/test speaker list stored with each prepared
    dataset (``heldout_speakers.json``) through :meth:`with_speakers`.

    ``rir_kinds``, ``music_splits`` and ``agent_splits`` list what training may use.
    """

    speakers: frozenset[str] = frozenset(VB_TEST_SPEAKERS)
    demand_environments: frozenset[str] = VB_HELDOUT_ENVIRONMENTS
    agent_voices: frozenset[str] = field(default_factory=_test_agent_voices)
    rir_kinds: frozenset[str] = frozenset({"simulated"})
    music_splits: frozenset[str] = frozenset({"train"})
    agent_splits: frozenset[str] = frozenset({"train"})
    noise_sources_blocked: frozenset[str] = frozenset({"esc50"})

    def with_speakers(self, speakers: Iterable[str]) -> HeldOut:
        """A copy that also holds out ``speakers`` (namespaced keys)."""
        return replace(self, speakers=self.speakers | frozenset(str(s) for s in speakers))


@dataclass
class MixerPools:
    """Prepared corpora the mixer draws from.

    ``speech`` needs ``speaker``, ``group``, ``pool`` and ``peak_energy`` (targets come
    from ``pool == "target"`` rows of speakers in ``embeddings``; interferers from any
    other speaker). ``rirs`` (simulated, ``kind`` column), ``music`` and ``agent`` (both
    with a ``split`` column) are optional: without RIRs the mixer synthesises them, and
    without music or agent audio those interferer types are skipped or bed-less.
    """

    speech: ShardedCorpus
    noise: ShardedCorpus
    embeddings: SpeakerEmbeddings
    rirs: ShardedCorpus | None = None
    music: ShardedCorpus | None = None
    agent: ShardedCorpus | None = None


def validate_training_pools(pools: MixerPools, held_out: HeldOut = HeldOut()) -> None:
    """Raise :class:`SplitLeakError` if training pools touch anything held out.

    Checks held-out speakers (speech and embeddings), held-out DEMAND environments and
    blocked noise sources, the noise metadata filter, real RIRs, held-out music and test
    agent voices.
    """
    problems: list[str] = []

    def report(what: str, values: Sequence[str] | set[str]) -> None:
        vals = sorted(str(v) for v in values)
        if vals:
            problems.append(f"{what}: {', '.join(vals[:8])}{' ...' if len(vals) > 8 else ''}")

    report("held-out speakers in speech", set(pools.speech.speaker.astype(str)) & held_out.speakers)
    report("held-out speakers in embeddings", set(pools.embeddings.speakers) & held_out.speakers)
    noise = pools.noise
    if noise.has_column("environment"):
        envs = {str(e).upper() for e in noise.column("environment") if e is not None}
        report("held-out DEMAND environments in noise", envs & {e.upper() for e in held_out.demand_environments})
    if noise.has_column("source"):
        report("blocked noise sources", {str(s) for s in noise.column("source")} & held_out.noise_sources_blocked)
    if noise.has_column("split"):
        report("non-training noise rows", {str(s) for s in noise.column("split") if s not in (None, "train")})
    text_cols = [c for c in ("utt_id", "speaker", "group", "description", "category", "label") if noise.has_column(c)]
    report(
        "noise rows matching the excluded-class filter",
        {
            str(noise.column("utt_id")[i])
            for i in range(len(noise))
            if mentions_excluded(*(noise.column(c)[i] for c in text_cols))
        },
    )
    if pools.rirs is not None:
        if not pools.rirs.has_column("kind"):
            problems.append("RIR corpus has no kind column")
        else:
            report("non-simulated RIRs", {str(k) for k in pools.rirs.column("kind")} - held_out.rir_kinds)
    for name, corpus, allowed in (
        ("music", pools.music, held_out.music_splits),
        ("agent", pools.agent, held_out.agent_splits),
    ):
        if corpus is None:
            continue
        if not corpus.has_column("split"):
            problems.append(f"{name} corpus has no split column")
        else:
            report(f"held-out {name} rows", {str(s) for s in corpus.column("split")} - allowed)
    if pools.agent is not None and pools.agent.has_column("voice_id"):
        report("test agent voices", {str(v) for v in pools.agent.column("voice_id")} & held_out.agent_voices)
    if pools.music is not None:
        cols = [c for c in ("utt_id", "genre", "artist") if pools.music.has_column(c)]
        report(
            "music rows matching the excluded-class filter",
            {
                str(pools.music.column("utt_id")[i])
                for i in range(len(pools.music))
                if mentions_excluded(*(pools.music.column(c)[i] for c in cols))
            },
        )
    if problems:
        raise SplitLeakError("training pools leak held-out data: " + "; ".join(problems))


def _load(roots: str | Path | Sequence[str | Path] | None) -> ShardedCorpus | None:
    if roots is None:
        return None
    items = [roots] if isinstance(roots, (str, Path)) else list(roots)
    if not items:
        return None
    corpora = [ShardedCorpus(r) for r in items]
    return corpora[0] if len(corpora) == 1 else ShardedCorpus.concat(corpora)


def load_training_pools(
    *,
    speech: str | Path | Sequence[str | Path],
    noise: str | Path | Sequence[str | Path],
    embeddings: str | Path | Sequence[str | Path],
    rirs: str | Path | Sequence[str | Path] | None = None,
    music: str | Path | Sequence[str | Path] | None = None,
    agent: str | Path | Sequence[str | Path] | None = None,
    music_splits: Iterable[str] | None = ("train",),
    agent_splits: Iterable[str] | None = ("train",),
) -> MixerPools:
    """Open prepared dataset directories (several per role are concatenated).

    Music and agent rows are kept only when their ``split`` column is in ``music_splits``
    or ``agent_splits`` (``None`` keeps every row), so pointing at a dataset that also holds
    the held-out tracks or test voices is safe. :func:`validate_training_pools` still runs
    when the :class:`Mixer` is built.
    """
    emb_paths = [embeddings] if isinstance(embeddings, (str, Path)) else list(embeddings)
    tables = [SpeakerEmbeddings.load(p) for p in emb_paths]
    speech_corpus = _load(speech)
    noise_corpus = _load(noise)
    if speech_corpus is None or noise_corpus is None:
        raise ValueError("speech and noise are required")
    return MixerPools(
        speech=speech_corpus,
        noise=noise_corpus,
        embeddings=tables[0] if len(tables) == 1 else SpeakerEmbeddings.merge(tables),
        rirs=_load(rirs),
        music=_only_splits(_load(music), music_splits),
        agent=_only_splits(_load(agent), agent_splits),
    )


def _only_splits(corpus: ShardedCorpus | None, splits: Iterable[str] | None) -> ShardedCorpus | None:
    if corpus is None or splits is None or not corpus.has_column("split"):
        return corpus
    return corpus.where("split", set(splits))


# --------------------------------------------------------------------------- the mixer


@dataclass(frozen=True)
class _Target:
    code: int
    emb_row: int
    rows: np.ndarray


@dataclass
class _Draw:
    """Everything random about one batch, drawn on the CPU (arrays over the batch)."""

    target_dry: np.ndarray
    target_peaks: np.ndarray
    target_rir: np.ndarray
    target_rir_peak: np.ndarray
    reverb: np.ndarray
    speech_i: np.ndarray
    interferer_peaks: np.ndarray
    music: np.ndarray
    music_db: np.ndarray
    kind: np.ndarray
    loudspeaker: np.ndarray
    interferer_path: np.ndarray
    interferer_rir: np.ndarray
    interferer_rir_peak: np.ndarray
    highpass: np.ndarray
    lowpass: np.ndarray
    eq_centre: np.ndarray
    eq_gain: np.ndarray
    eq_width: np.ndarray
    drive: np.ndarray
    echo_db: np.ndarray
    noise: np.ndarray
    speaker_row: np.ndarray
    emb_k: np.ndarray
    null: np.ndarray
    present: np.ndarray
    clip: np.ndarray
    level_db: np.ndarray
    snr_db: np.ndarray
    sir_db: np.ndarray
    clip_fraction: np.ndarray
    provenance: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def empty(cls, b: int, t: int, f: int, taps: int, bumps: int, ctrl: int) -> _Draw:
        return cls(
            target_dry=np.zeros((b, t), np.int16),
            target_peaks=np.full((b, f), np.inf),
            target_rir=np.zeros((b, taps), np.float32),
            target_rir_peak=np.zeros(b, np.int64),
            reverb=np.zeros(b, bool),
            speech_i=np.zeros((b, t), np.int16),
            interferer_peaks=np.full((b, f), np.inf),
            music=np.zeros((b, t), np.int16),
            music_db=np.zeros(b),
            kind=np.zeros(b, np.int64),
            loudspeaker=np.zeros(b, bool),
            interferer_path=np.zeros(b, bool),
            interferer_rir=np.zeros((b, taps), np.float32),
            interferer_rir_peak=np.zeros(b, np.int64),
            highpass=np.full(b, 20.0),
            lowpass=np.full(b, 20000.0),
            eq_centre=np.full((b, bumps), 1000.0),
            eq_gain=np.zeros((b, bumps)),
            eq_width=np.ones((b, bumps)),
            drive=np.ones(b),
            echo_db=np.zeros((b, ctrl)),
            noise=np.zeros((b, t), np.int16),
            speaker_row=np.zeros(b, np.int64),
            emb_k=np.zeros(b, np.int64),
            null=np.zeros(b, bool),
            present=np.zeros(b, bool),
            clip=np.zeros(b, bool),
            level_db=np.zeros(b),
            snr_db=np.zeros(b),
            sir_db=np.full(b, np.nan),
            clip_fraction=np.ones(b),
        )


class Mixer:
    """Seeded batch generator over :class:`MixerPools`; see the module docstring.

    ``batch(k)`` renders batch ``k`` without changing state; iterating (or
    :meth:`sample`) renders ``next_index`` and advances it.
    """

    def __init__(
        self,
        pools: MixerPools,
        config: MixerConfig = MixerConfig(),
        *,
        seed: int = 0,
        device: str | torch.device = "cpu",
        held_out: HeldOut | None = HeldOut(),
        return_components: bool = False,
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if held_out is not None:
            validate_training_pools(pools, held_out)
        if pools.embeddings.dim != C.EMBEDDING_DIM:
            raise ValueError(f"embeddings are {pools.embeddings.dim}-d; the contract says {C.EMBEDDING_DIM}")
        self.pools = pools
        self.config = config
        self.seed = int(seed)
        self.device = torch.device(device)
        self.return_components = return_components
        self.next_index = 0
        self.num_samples = config.num_samples
        self.n_frames = config.num_frames
        self._taps = int(round(config.rir_max_seconds * C.SAMPLE_RATE))
        self._n_fft = next_fft_size(self.num_samples + self._taps - 1)
        self._direct = int(round(config.direct_ms * C.SAMPLE_RATE / 1000))
        self._early = int(round(config.early_ms * C.SAMPLE_RATE / 1000))
        self._min_seg = int(round(config.min_segment_seconds * C.SAMPLE_RATE))
        step = int(round(config.residual_echo_step_ms * C.SAMPLE_RATE / 1000))
        self._echo_ctrl = max(2, -(-self.num_samples // max(1, step)) + 1)
        self._freqs = torch.fft.rfftfreq(self._n_fft, d=1.0 / C.SAMPLE_RATE).to(self.device)

        speech = pools.speech
        names, code = np.unique(speech.speaker.astype(str), return_inverse=True)
        self._speech_code = code
        self._speech_names = names
        self._speech_len = speech.num_samples
        self._speech_peak = speech.peak_energy
        pool = (
            speech.column("pool").astype(str)
            if speech.has_column("pool")
            else np.full(len(speech), POOL_TARGET)
        )
        emb_index = pools.embeddings.index()
        candidates = np.flatnonzero((pool == POOL_TARGET) & (self._speech_len >= self._min_seg))
        candidates = candidates[np.argsort(code[candidates], kind="stable")]
        self._targets: list[_Target] = []
        if candidates.size:
            for rows in np.split(candidates, np.flatnonzero(np.diff(code[candidates])) + 1):
                name = str(names[code[rows[0]]])
                if name in emb_index:
                    self._targets.append(_Target(int(code[rows[0]]), emb_index[name], rows))
        if not self._targets:
            raise ValueError("no target speakers: need target-pool speech from speakers with embeddings")
        self._interferer_rows = np.flatnonzero(self._speech_len >= self._min_seg)
        if len(np.unique(code[self._interferer_rows])) < 2:
            raise ValueError("speech pool needs at least two speakers")
        self._noise_rows = np.flatnonzero(pools.noise.num_samples >= C.WINDOW_LENGTH)
        if not self._noise_rows.size:
            raise ValueError("noise pool is empty")

        self._rir_rows: np.ndarray | None = None
        self._rir_room: np.ndarray | None = None
        self._rir_by_room: dict[str, np.ndarray] = {}
        self._small_rows: np.ndarray | None = None
        if pools.rirs is not None and len(pools.rirs):
            self._rir_rows = np.flatnonzero(pools.rirs.num_samples > 0)
            if pools.rirs.has_column("room"):
                self._rir_room = pools.rirs.column("room").astype(str)
                for room in np.unique(self._rir_room[self._rir_rows]):
                    self._rir_by_room[str(room)] = self._rir_rows[self._rir_room[self._rir_rows] == room]
            if pools.rirs.has_column("room_size"):
                small = self._rir_rows[pools.rirs.column("room_size").astype(str)[self._rir_rows] == "smallroom"]
                self._small_rows = small if small.size else None
        self._music_rows = (
            np.flatnonzero(pools.music.num_samples >= C.WINDOW_LENGTH)
            if pools.music is not None and len(pools.music)
            else None
        )
        self._agent_rows = (
            np.flatnonzero(pools.agent.num_samples >= self._min_seg)
            if pools.agent is not None and len(pools.agent)
            else None
        )
        if self._agent_rows is not None and not self._agent_rows.size:
            self._agent_rows = None
        weights = np.asarray(config.interferer_weights, dtype=np.float64)
        if self._agent_rows is None:
            weights[INTERFERER_AGENT - 1] = 0.0
        if weights.sum() <= 0:
            raise ValueError("no interferer type is available")
        self._kind_p = weights / weights.sum()
        self._emb_table = torch.from_numpy(pools.embeddings.table).to(self.device)
        self._emb_k = pools.embeddings.per_speaker

    # ------------------------------------------------------------------ public API

    def batch(self, index: int) -> dict[str, torch.Tensor]:
        """Render batch ``index`` (pure: same pools, config, seed and index -> same batch)."""
        if index < 0:
            raise ValueError("batch index must be non-negative")
        return self._render(self._draw(int(index)))

    def sample(self) -> dict[str, torch.Tensor]:
        """Render ``next_index`` and advance it."""
        out = self.batch(self.next_index)
        self.next_index += 1
        return out

    def describe(self, index: int) -> list[dict[str, Any]]:
        """Where each example of batch ``index`` comes from (utterance ids, kinds), without
        rendering it. Useful for listening passes and for provenance checks."""
        return self._draw(int(index)).provenance

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            yield self.sample()

    def iterate(self, *, prefetch: int = 2) -> Iterator[dict[str, torch.Tensor]]:
        """Yield batches from ``next_index`` on, drawing the next ``prefetch`` on a thread.

        The CPU draw (index sampling and memmap reads) of upcoming batches overlaps the
        device work of the current one. The batches are exactly those of :meth:`sample`,
        and ``next_index`` advances only as batches are yielded, so :meth:`state_dict`
        stays exact.
        """
        if prefetch < 1:
            yield from self
            return
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="earmark-mixer")
        pending: deque[Future[_Draw]] = deque()
        submitted = self.next_index
        try:
            while True:
                while len(pending) < prefetch:
                    pending.append(pool.submit(self._draw, submitted))
                    submitted += 1
                draw = pending.popleft().result()
                out = self._render(draw)
                self.next_index += 1
                yield out
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to resume the batch sequence."""
        return {"seed": self.seed, "next_index": self.next_index, "config": asdict(self.config)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Resume from :meth:`state_dict`; the seed and config must match."""
        same_config = json.dumps(dict(state["config"]), sort_keys=True) == json.dumps(
            asdict(self.config), sort_keys=True
        )
        if int(state["seed"]) != self.seed or not same_config:
            raise ValueError("mixer state was saved with a different seed or config")
        self.next_index = int(state["next_index"])

    @property
    def num_target_speakers(self) -> int:
        return len(self._targets)

    # ------------------------------------------------------------------ drawing (CPU)

    def _layout_single(self, rng: np.random.Generator, n: int) -> tuple[int, int, int]:
        t = self.num_samples
        if n >= t:
            return int(rng.integers(n - t + 1)), 0, t
        return 0, int(rng.integers(t - n + 1)), int(n)

    def _layout_target(self, rng: np.random.Generator, rows: np.ndarray) -> list[tuple[int, int, int, int]]:
        cfg = self.config
        t = self.num_samples
        u1 = int(rows[rng.integers(len(rows))])
        if rng.random() >= cfg.p_pause:
            src, dst, length = self._layout_single(rng, int(self._speech_len[u1]))
            return [(u1, src, dst, length)]
        lead = min(int(rng.uniform(*cfg.lead_seconds) * C.SAMPLE_RATE), max(0, t - self._min_seg))
        pause = int(rng.uniform(*cfg.pause_seconds) * C.SAMPLE_RATE)
        avail = t - lead
        n1 = int(self._speech_len[u1])
        len1 = int(min(n1, max(self._min_seg, avail - pause - self._min_seg), avail))
        segs = [(u1, int(rng.integers(n1 - len1 + 1)), lead, len1)]
        start2 = lead + len1 + pause
        if t - start2 >= self._min_seg:
            u2 = int(rows[rng.integers(len(rows))])
            n2 = int(self._speech_len[u2])
            len2 = int(min(n2, t - start2))
            segs.append((u2, int(rng.integers(n2 - len2 + 1)), start2, len2))
        return segs

    def _gather_tiled(self, corpus: ShardedCorpus, row: int, rng: np.random.Generator, out: np.ndarray) -> None:
        src = corpus.audio_int16(row)
        n, t = src.size, out.size
        if n >= t:
            start = int(rng.integers(n - t + 1))
            out[:] = src[start : start + t]
        else:
            out[:] = src[(int(rng.integers(n)) + np.arange(t)) % n]

    def _normalised(self, h: np.ndarray) -> np.ndarray:
        h = h[: self._taps].astype(np.float32)
        peak = float(np.max(np.abs(h))) if h.size else 0.0
        return h / peak if peak > 0 else h

    def _draw_rir(self, rng: np.random.Generator, room: str | None = None) -> tuple[np.ndarray, str | None]:
        if self._rir_rows is None or self.pools.rirs is None:
            return synthetic_rir(rng, rng.uniform(*self.config.synthetic_rt60_s), self._taps), None
        rows = self._rir_by_room.get(room, self._rir_rows) if room is not None else self._rir_rows
        row = int(rows[rng.integers(len(rows))])
        h = self._normalised(self.pools.rirs.audio(row))
        return h, (str(self._rir_room[row]) if self._rir_room is not None else None)

    def _draw_small_room(self, rng: np.random.Generator) -> np.ndarray:
        if self._small_rows is not None and self.pools.rirs is not None:
            row = int(self._small_rows[rng.integers(len(self._small_rows))])
            return self._normalised(self.pools.rirs.audio(row))
        return synthetic_rir(rng, rng.uniform(*self.config.small_room_rt60_s), self._taps)

    def _draw_other_speaker(self, rng: np.random.Generator, code: int) -> int:
        rows = self._interferer_rows
        for _ in range(64):
            row = int(rows[rng.integers(len(rows))])
            if self._speech_code[row] != code:
                return row
        others = rows[self._speech_code[rows] != code]
        return int(others[rng.integers(len(others))])

    def _draw(self, index: int) -> _Draw:
        cfg = self.config
        rng = np.random.default_rng([self.seed, index])
        b_total, t, f = cfg.batch_size, self.num_samples, self.n_frames
        d = _Draw.empty(b_total, t, f, self._taps, cfg.eq_bumps, self._echo_ctrl)
        speech = self.pools.speech
        for b in range(b_total):
            tgt = self._targets[int(rng.integers(len(self._targets)))]
            d.speaker_row[b] = tgt.emb_row
            d.emb_k[b] = int(rng.integers(self._emb_k))
            d.null[b] = rng.random() < cfg.p_null
            d.present[b] = rng.random() >= cfg.p_target_absent
            d.level_db[b] = rng.uniform(*cfg.target_level_db)
            d.snr_db[b] = rng.uniform(*cfg.snr_db)
            record: dict[str, Any] = {
                "target_speaker": str(self._speech_names[tgt.code]),
                "target_present": bool(d.present[b]),
                "null_embedding": bool(d.null[b]),
                "target_utts": [],
                "target_segments": [],
                "reverberant": False,
                "interferer_kind": INTERFERER_NONE,
                "interferer_utt": None,
                "interferer_speaker": None,
                "loudspeaker": False,
                "noise_utt": None,
            }

            room: str | None = None
            delay = 0
            if rng.random() < cfg.p_reverb:
                h, room = self._draw_rir(rng)
                d.target_rir[b, : h.size] = h
                delay = int(np.argmax(np.abs(h)))
                d.target_rir_peak[b] = delay
                d.reverb[b] = True
                record["reverberant"] = True
            if d.present[b]:
                segs = self._layout_target(rng, tgt.rows)
                for row, src, dst, length in segs:
                    d.target_dry[b, dst : dst + length] = speech.audio_int16(row, src, length)
                d.target_peaks[b] = segment_peak_frames(
                    [(self._speech_peak[row], dst, length) for row, _, dst, length in segs], f, delay=delay
                )
                record["target_utts"] = [str(speech.column("utt_id")[row]) for row, _, _, _ in segs]
                record["target_segments"] = [
                    (str(speech.column("utt_id")[row]), int(src), int(dst), int(length))
                    for row, src, dst, length in segs
                ]

            if rng.random() < cfg.p_interferer:
                kind = int(rng.choice(3, p=self._kind_p)) + 1
                if kind == INTERFERER_AGENT and self._agent_rows is not None and self.pools.agent is not None:
                    corpus = self.pools.agent
                    row = int(self._agent_rows[rng.integers(len(self._agent_rows))])
                else:
                    corpus = speech
                    row = self._draw_other_speaker(rng, tgt.code)
                src, dst, length = self._layout_single(rng, int(corpus.num_samples[row]))
                d.speech_i[b, dst : dst + length] = corpus.audio_int16(row, src, length)
                loud = kind == INTERFERER_AGENT or rng.random() < cfg.p_loudspeaker_other
                if kind == INTERFERER_TV and self._music_rows is not None and self.pools.music is not None:
                    mrow = int(self._music_rows[rng.integers(len(self._music_rows))])
                    self._gather_tiled(self.pools.music, mrow, rng, d.music[b])
                    d.music_db[b] = rng.uniform(*cfg.music_bed_db)
                h_i: np.ndarray | None = None
                if loud:
                    d.highpass[b] = rng.uniform(*cfg.highpass_hz)
                    d.lowpass[b] = rng.uniform(*cfg.lowpass_hz)
                    d.eq_centre[b] = np.exp2(rng.uniform(np.log2(150.0), np.log2(6000.0), cfg.eq_bumps))
                    d.eq_gain[b] = rng.uniform(-cfg.eq_db, cfg.eq_db, cfg.eq_bumps)
                    d.eq_width[b] = rng.uniform(*cfg.eq_octaves, cfg.eq_bumps)
                    d.drive[b] = rng.uniform(*cfg.drive)
                    h_i = self._draw_small_room(rng)
                    if rng.random() < cfg.p_residual_echo:
                        depth = rng.uniform(*cfg.residual_echo_db)
                        d.echo_db[b] = -rng.uniform(0.0, depth, self._echo_ctrl)
                elif rng.random() < cfg.p_reverb:
                    same = room is not None and rng.random() < cfg.p_same_room
                    h_i, _ = self._draw_rir(rng, room if same else None)
                idelay = 0
                if h_i is not None:
                    d.interferer_rir[b, : h_i.size] = h_i
                    idelay = int(np.argmax(np.abs(h_i)))
                    d.interferer_rir_peak[b] = idelay
                    d.interferer_path[b] = True
                peaks = corpus.peak_energy
                d.interferer_peaks[b] = segment_peak_frames([(peaks[row], dst, length)], f, delay=idelay)
                d.kind[b] = kind
                d.loudspeaker[b] = loud
                d.sir_db[b] = rng.uniform(*cfg.sir_db)
                record.update(
                    interferer_kind=kind,
                    interferer_utt=str(corpus.column("utt_id")[row]),
                    interferer_speaker=str(corpus.speaker[row]),
                    loudspeaker=bool(loud),
                )

            nrow = int(self._noise_rows[rng.integers(len(self._noise_rows))])
            self._gather_tiled(self.pools.noise, nrow, rng, d.noise[b])
            record["noise_utt"] = str(self.pools.noise.column("utt_id")[nrow])
            d.provenance.append(record)
            d.clip[b] = rng.random() < cfg.p_clip
            d.clip_fraction[b] = rng.uniform(*cfg.clip_fraction)
        return d

    # ------------------------------------------------------------------ rendering (device)

    def _tensor(self, array: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Host array to a device tensor, converting on the host first (MPS has no float64)."""
        t = torch.from_numpy(np.ascontiguousarray(array))
        if dtype is not None:
            t = t.to(dtype)
        if self.device.type == "cuda":
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def _render(self, d: _Draw) -> dict[str, torch.Tensor]:
        cfg = self.config
        t_len, n = self.num_samples, self._n_fft
        f32 = torch.float32
        tens = self._tensor
        inv16 = 1.0 / 32768.0
        present = tens(d.present)
        null = tens(d.null)
        kind = tens(d.kind)
        has_i = kind > 0
        level = tens(d.level_db, f32)
        snr = tens(d.snr_db, f32)
        sir = torch.nan_to_num(tens(d.sir_db, f32), nan=0.0)

        # Target through its (optional) room: full, early (reference) and direct (labels).
        dry = tens(d.target_dry, f32) * inv16
        h_t = tens(d.target_rir)
        hd_t, he_t = rir_windows(h_t, tens(d.target_rir_peak), direct_samples=self._direct, early_samples=self._early)
        conv = torch.fft.irfft(
            torch.fft.rfft(dry, n=n).unsqueeze(1) * torch.fft.rfft(torch.stack([h_t, he_t, hd_t], 1), n=n), n=n
        )[..., :t_len]
        rv = tens(d.reverb).unsqueeze(-1)
        t_rev = torch.where(rv, conv[:, 0], dry)
        t_early = torch.where(rv, conv[:, 1], dry)
        t_direct = torch.where(rv, conv[:, 2], dry)
        del conv

        # Interferer: speech (+ music bed) -> optional loudspeaker chain -> room.
        s = tens(d.speech_i, f32) * inv16
        m = tens(d.music, f32) * inv16
        p_s = active_power(s)
        g_m = level_gain(m.square().mean(-1), 10.0 * torch.log10(p_s.clamp_min(_TINY)) + tens(d.music_db, f32))
        u = s + g_m.unsqueeze(-1) * m
        resp = loudspeaker_response(
            self._freqs, tens(d.highpass, f32), tens(d.lowpass, f32),
            tens(d.eq_centre, f32), tens(d.eq_gain, f32), tens(d.eq_width, f32),
        )
        lin = torch.fft.irfft(torch.fft.rfft(torch.stack([u, s], 1), n=n) * resp.unsqueeze(1), n=n)[..., :t_len]
        lsp = tens(d.loudspeaker).unsqueeze(-1)
        src_mix = torch.where(lsp, soft_clip(lin[:, 0], tens(d.drive, f32)), u)
        src_ref = torch.where(lsp, lin[:, 1], s)
        h_i = tens(d.interferer_rir)
        hd_i, he_i = rir_windows(h_i, tens(d.interferer_rir_peak), direct_samples=self._direct, early_samples=self._early)
        conv_i = torch.fft.irfft(
            torch.fft.rfft(torch.stack([src_mix, src_ref, src_ref], 1), n=n)
            * torch.fft.rfft(torch.stack([h_i, he_i, hd_i], 1), n=n),
            n=n,
        )[..., :t_len]
        ip = tens(d.interferer_path).unsqueeze(-1)
        env = residual_echo_envelope(tens(d.echo_db, f32), t_len)
        i_out = torch.where(ip, conv_i[:, 0], src_mix) * env
        i_ref = torch.where(ip, conv_i[:, 1], src_ref) * env
        i_direct = torch.where(ip, conv_i[:, 2], src_ref) * env
        del conv_i, lin

        # Levels: active target level, then noise and interferer relative to it.
        noise = tens(d.noise, f32) * inv16
        zeros = torch.zeros_like(level)
        g_t = torch.where(present, level_gain(active_power(t_rev), level), zeros)
        g_n = level_gain(noise.square().mean(-1), level - snr)
        g_i = torch.where(has_i, level_gain(active_power(i_out), level - sir), zeros)
        target_mix = g_t.unsqueeze(-1) * t_rev
        interferer_mix = g_i.unsqueeze(-1) * i_out
        noise_mix = g_n.unsqueeze(-1) * noise
        mixture = target_mix + interferer_mix + noise_mix
        peak = mixture.abs().amax(-1)
        out_gain = torch.where(peak > cfg.max_peak, cfg.max_peak / peak.clamp_min(_TINY), torch.ones_like(peak))
        og = out_gain.unsqueeze(-1)
        mixture, target_mix, interferer_mix, noise_mix = (x * og for x in (mixture, target_mix, interferer_mix, noise_mix))
        limit = (tens(d.clip_fraction, f32) * mixture.abs().amax(-1)).unsqueeze(-1)
        clip = tens(d.clip)
        clipped = torch.maximum(torch.minimum(mixture, limit), -limit)
        mixture = torch.where(clip.unsqueeze(-1), clipped, mixture)

        # Reference and labels at output scale.
        gt = (g_t * out_gain).unsqueeze(-1)
        gi = (g_i * out_gain).unsqueeze(-1)
        interferer_reference = gi * i_ref
        target = gt * t_early + torch.where(null.unsqueeze(-1), interferer_reference, torch.zeros_like(t_early))
        ratio = db_to_power_ratio(C.VAD_THRESHOLD_DB)
        rho_t = t_direct.square().sum(-1) / dry.square().sum(-1).clamp_min(_TINY)
        thr_t = tens(d.target_peaks, f32) * (rho_t.unsqueeze(-1) * gt.square() * ratio)
        target_direct = gt * t_direct
        act = (frame_energy(target_direct) > thr_t) & present.unsqueeze(-1)
        rho_i = i_direct.square().sum(-1) / s.square().sum(-1).clamp_min(_TINY)
        thr_i = tens(d.interferer_peaks, f32) * (rho_i.unsqueeze(-1) * gi.square() * ratio)
        act_i = (frame_energy(gi * i_direct) > thr_i) & (null & has_i).unsqueeze(-1)
        vad = apply_hangover(act | act_i).to(f32)

        emb = self._emb_table[tens(d.speaker_row), tens(d.emb_k)]
        emb = torch.where(null.unsqueeze(-1), torch.zeros_like(emb), emb)
        out = {
            "mixture": mixture,
            "target": target,
            "vad": vad,
            "embedding": emb,
            "null_embedding": null,
            "target_present": present,
            "interferer_kind": kind,
            "loudspeaker": tens(d.loudspeaker),
            "reverberant": tens(d.reverb) & present,
            "clipped": clip,
            "snr_db": snr,
            "sir_db": tens(d.sir_db, f32),
            "target_level_db": level,
            "speaker_index": tens(d.speaker_row),
        }
        if self.return_components:
            out.update(
                target_mix=target_mix,
                interferer_mix=interferer_mix,
                noise_mix=noise_mix,
                target_direct=target_direct,
                interferer_reference=interferer_reference,
                output_gain=out_gain,
            )
        return out
