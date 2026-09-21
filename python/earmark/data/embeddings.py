"""Frozen speaker encoder interface, enrolment clips and the per-speaker embedding table.

The model is conditioned on a speaker embedding from a frozen WeSpeaker ResNet34-LM
(CC BY 4.0, 256-d). Training uses eight embeddings per speaker, precomputed on Kaggle
(``notebooks/kaggle_embeddings.py``) from augmented 5-10 s clips of that speaker's
enrolment pool (noise, RIR, EQ, Opus round trip). The mixer samples one per example.

* :class:`SpeakerEncoder` is the interface: 16 kHz float audio ``[B, T]`` in, unit-norm
  ``[B, EMBEDDING_DIM]`` out. Embeddings are always L2-normalised, so the model sees the
  same scale from Kaggle, from the dev runner and from the browser's enrolment.
* :class:`StubSpeakerEncoder` is a deterministic, weight-free stand-in for tests.
* :class:`WeSpeakerOnnxEncoder` wraps the real model: the ONNX release on the Hub behind
  a Kaldi fbank front end. It imports onnxruntime lazily, and uses torchaudio for the
  front end when it is installed (Kaggle) or :mod:`earmark.data.fbank` when it is not
  (the demo server on a torch release torchaudio has no wheel for).
* :class:`SpeakerEmbeddings` is the ``.npz`` table the mixer reads.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from earmark import constants as C
from earmark.data.corpora import stable_hash
from earmark.data.labels import active_power_np
from earmark.data.shards import POOL_ENROL, ShardedCorpus, to_int16, trim_silence

__all__ = [
    "KALDI_FBANK",
    "WAVE_SCALE",
    "WESPEAKER_ONNX",
    "WESPEAKER_ONNX_SHA256",
    "WESPEAKER_REPO",
    "CodecUnavailable",
    "EnrolAugment",
    "EnrolAugmentConfig",
    "SpeakerEmbeddings",
    "SpeakerEncoder",
    "StubSpeakerEncoder",
    "WeSpeakerOnnxEncoder",
    "compute_speaker_embeddings",
    "enrolment_clip",
    "l2_normalise",
    "opus_available",
    "opus_roundtrip",
    "random_eq",
]

WESPEAKER_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
WESPEAKER_ONNX = "voxceleb_resnet34_LM.onnx"
#: sha256 and size of the ONNX file on the Hub (checked 2026-09-11).
WESPEAKER_ONNX_SHA256 = "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"
WESPEAKER_ONNX_BYTES = 26_530_309

#: Kaldi fbank settings of WeSpeaker's front end (80 mel bins, 25/10 ms, no dither at
#: inference, Hamming window, no energy term), followed by per-utterance mean removal.
KALDI_FBANK: Mapping[str, Any] = MappingProxyType(
    {
        "num_mel_bins": 80,
        "frame_length": 25.0,
        "frame_shift": 10.0,
        "dither": 0.0,
        "window_type": "hamming",
        "use_energy": False,
        "sample_frequency": float(C.SAMPLE_RATE),
    }
)
#: WeSpeaker scales float audio to the int16 range before the fbank.
WAVE_SCALE = 32768.0


def l2_normalise(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Unit-normalise along the last dimension."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@runtime_checkable
class SpeakerEncoder(Protocol):
    """A frozen speaker encoder: ``[B, T]`` 16 kHz float audio -> unit-norm ``[B, D]``."""

    name: str
    embedding_dim: int

    def embed(self, wave: torch.Tensor) -> torch.Tensor: ...


class StubSpeakerEncoder:
    """Deterministic, weight-free encoder for tests and smoke runs.

    Features are the per-bin mean and standard deviation of the log power spectrum
    (contract STFT), with the mean over bins removed so a gain change leaves them alone.
    A fixed seeded projection maps them to ``embedding_dim`` and the result is
    L2-normalised. Different spectral envelopes and pitches land far apart; that is all
    tests need.
    """

    name = "stub-logspec-projection"

    def __init__(self, embedding_dim: int = C.EMBEDDING_DIM, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.embedding_dim = embedding_dim
        self._proj = torch.randn(2 * C.N_BINS, embedding_dim, generator=g) / math.sqrt(2 * C.N_BINS)
        n = torch.arange(C.WINDOW_LENGTH, dtype=torch.float64)
        self._window = torch.sin(math.pi * n / C.WINDOW_LENGTH).float()

    @torch.no_grad()
    def embed(self, wave: torch.Tensor) -> torch.Tensor:
        x = wave.detach().float().cpu()
        if x.ndim == 1:
            x = x[None]
        min_len = 2 * C.WINDOW_LENGTH
        if x.shape[-1] < min_len:
            x = torch.nn.functional.pad(x, (0, min_len - x.shape[-1]))
        spec = torch.stft(
            x, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH, win_length=C.WINDOW_LENGTH,
            window=self._window, center=False, return_complex=True,
        )
        power = spec.abs().square()
        # A floor relative to the clip's mean power keeps the features gain-invariant.
        floor = 1e-6 * power.mean(dim=(-2, -1), keepdim=True) + 1e-20
        logp = torch.log(power + floor)
        feats = torch.cat([logp.mean(-1), logp.std(-1)], dim=-1)
        feats[:, : C.N_BINS] -= feats[:, : C.N_BINS].mean(-1, keepdim=True)
        return l2_normalise(feats @ self._proj)


class WeSpeakerOnnxEncoder:
    """WeSpeaker ResNet34-LM (ONNX) with its Kaldi fbank front end. Kaggle/Colab only.

    Needs ``onnxruntime`` (or ``onnxruntime-gpu``) and ``torchaudio``. The model input is
    ``[B, frames, 80]`` mean-normalised fbank; the output is normalised here.
    """

    name = "wespeaker-voxceleb-resnet34-LM"
    embedding_dim = 256

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        intra_op_threads: int | None = None,
        verify_sha256: bool = True,
    ) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if verify_sha256:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != WESPEAKER_ONNX_SHA256:
                raise ValueError(f"{path} sha256 {digest} != pinned {WESPEAKER_ONNX_SHA256}")
        opts = ort.SessionOptions()
        if intra_op_threads:
            opts.intra_op_num_threads = int(intra_op_threads)
        available = ort.get_available_providers()
        chosen = list(providers) if providers else [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
        ]
        self._session = ort.InferenceSession(str(path), sess_options=opts, providers=chosen)
        self._input = self._session.get_inputs()[0].name
        self._output = self._session.get_outputs()[0].name
        self.providers = tuple(self._session.get_providers())

    @classmethod
    def from_hub(cls, *, cache_dir: str | Path | None = None, **kwargs: Any) -> WeSpeakerOnnxEncoder:
        """Download the pinned ONNX file from the Hub (public; no token needed) and load it."""
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(WESPEAKER_REPO, WESPEAKER_ONNX, cache_dir=cache_dir)
        return cls(path, **kwargs)

    @staticmethod
    def fbank(wave: torch.Tensor) -> torch.Tensor:
        """Mean-normalised Kaldi fbank ``[frames, 80]`` of one 16 kHz waveform ``[T]``.

        Uses torchaudio when it is installed (bit-identical to the training-time
        embeddings) and otherwise :mod:`earmark.data.fbank`, which reproduces it to
        float32 rounding (``tests/data/test_fbank.py``).
        """
        scaled = wave.float()[None] * WAVE_SCALE
        try:
            from torchaudio.compliance import kaldi

            feats = kaldi.fbank(scaled, **KALDI_FBANK)
        except ImportError:
            from earmark.data.fbank import kaldi_fbank

            feats = kaldi_fbank(scaled, **KALDI_FBANK)
        return feats - feats.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def embed(self, wave: torch.Tensor) -> torch.Tensor:
        x = wave.detach().float().cpu()
        if x.ndim == 1:
            x = x[None]
        feats = torch.stack([self.fbank(w) for w in x])
        out = self._session.run([self._output], {self._input: feats.numpy()})[0]
        return l2_normalise(torch.from_numpy(np.asarray(out, dtype=np.float32)))


# --------------------------------------------------------------------------- table


@dataclass(frozen=True)
class SpeakerEmbeddings:
    """``table[s, k]`` is embedding ``k`` of ``speakers[s]`` (float32 ``[S, K, D]``, unit norm)."""

    speakers: tuple[str, ...]
    table: np.ndarray
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = np.asarray(self.table, dtype=np.float32)
        object.__setattr__(self, "table", t)
        object.__setattr__(self, "speakers", tuple(str(s) for s in self.speakers))
        if t.ndim != 3 or t.shape[0] != len(self.speakers):
            raise ValueError(f"table shape {t.shape} does not match {len(self.speakers)} speakers")
        if len(set(self.speakers)) != len(self.speakers):
            raise ValueError("duplicate speakers in the embedding table")
        if not np.isfinite(t).all():
            raise ValueError("embedding table has non-finite values")
        norms = np.linalg.norm(t, axis=-1)
        if t.size and not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("embeddings must be L2-normalised")

    @property
    def per_speaker(self) -> int:
        return int(self.table.shape[1])

    @property
    def dim(self) -> int:
        return int(self.table.shape[2])

    def index(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.speakers)}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            np.savez(
                fh,
                speakers=np.array(self.speakers, dtype=str),
                table=self.table,
                meta=np.array(json.dumps(dict(self.meta), sort_keys=True, default=str)),
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> SpeakerEmbeddings:
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
            return cls(tuple(z["speakers"].tolist()), z["table"], meta)

    @staticmethod
    def merge(parts: Sequence[SpeakerEmbeddings]) -> SpeakerEmbeddings:
        """Concatenate tables with disjoint speakers and equal ``K`` and ``D``."""
        if not parts:
            raise ValueError("nothing to merge")
        speakers = tuple(s for p in parts for s in p.speakers)
        table = np.concatenate([p.table for p in parts], axis=0)
        meta = {"merged_from": [dict(p.meta) for p in parts]}
        return SpeakerEmbeddings(speakers, table, meta)


# --------------------------------------------------------------------------- enrolment


def enrolment_clip(
    corpus: ShardedCorpus,
    rows: Sequence[int] | np.ndarray,
    rng: np.random.Generator,
    *,
    seconds: tuple[float, float] = (5.0, 10.0),
    gap_seconds: tuple[float, float] = (0.05, 0.3),
) -> np.ndarray:
    """A 5-10 s enrolment clip built from ``rows`` (a speaker's enrolment-pool utterances).

    Utterances are silence-trimmed and joined in a seeded order with short gaps (cycling
    if the pool is short), then cropped at a random point to the drawn length.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        raise ValueError("no enrolment utterances")
    target = int(rng.uniform(*seconds) * C.SAMPLE_RATE)
    order = rng.permutation(rows)
    pieces: list[np.ndarray] = []
    total = 0
    visited = 0
    while total < target:
        row = int(order[visited % len(order)])
        visited += 1
        audio = corpus.audio(row)
        start, end = trim_silence(audio)
        if end > start:
            if pieces:
                gap = np.zeros(int(rng.uniform(*gap_seconds) * C.SAMPLE_RATE), np.float32)
                pieces.append(gap)
                total += gap.size
            pieces.append(audio[start:end])
            total += end - start
        elif visited > 4 * len(order) and total == 0:
            raise ValueError("enrolment utterances are silent")
    clip = np.concatenate(pieces)
    if clip.size > target:
        begin = int(rng.integers(0, clip.size - target + 1))
        clip = clip[begin : begin + target]
    return clip.astype(np.float32)


class CodecUnavailable(RuntimeError):
    """Raised when no ffmpeg with libopus is available."""


def opus_available(ffmpeg: str | None = None) -> bool:
    """Whether ``ffmpeg`` (default: on PATH) has the libopus encoder."""
    exe = ffmpeg or shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        out = subprocess.run(
            [exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "libopus" in out.stdout


def opus_roundtrip(
    x: np.ndarray,
    *,
    bitrate: int = 16000,
    sample_rate: int = C.SAMPLE_RATE,
    ffmpeg: str | None = None,
) -> np.ndarray:
    """Encode to Opus at ``bitrate`` b/s and decode back (via ffmpeg + libopus).

    The output has the input's length (trailing padding is cropped; a short decode is
    zero-padded). Raises :class:`CodecUnavailable` without ffmpeg/libopus.
    """
    exe = ffmpeg or shutil.which("ffmpeg")
    if not exe:
        raise CodecUnavailable("ffmpeg not found on PATH")
    pcm, _ = to_int16(np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0))
    raw = ["-f", "s16le", "-ar", str(sample_rate), "-ac", "1"]
    quiet = [exe, "-hide_banner", "-loglevel", "error"]
    try:
        enc = subprocess.run(
            [*quiet, *raw, "-i", "pipe:0", "-c:a", "libopus", "-b:a", str(bitrate),
             "-application", "voip", "-f", "ogg", "pipe:1"],
            input=pcm.tobytes(), capture_output=True, check=True,
        )
        dec = subprocess.run(
            [*quiet, "-f", "ogg", "-i", "pipe:0", *raw, "pipe:1"],
            input=enc.stdout, capture_output=True, check=True,
        )
    except subprocess.CalledProcessError as err:
        raise CodecUnavailable(err.stderr.decode("utf-8", "replace").strip() or str(err)) from err
    y = np.frombuffer(dec.stdout, dtype="<i2").astype(np.float32) / 32768.0
    out = np.zeros(len(pcm), dtype=np.float32)
    n = min(len(out), len(y))
    out[:n] = y[:n]
    return out


def random_eq(
    x: np.ndarray, rng: np.random.Generator, *, max_db: float = 6.0, bumps: int = 3
) -> np.ndarray:
    """Zero-phase random EQ: a sum of Gaussian bumps in log-frequency, applied by FFT."""
    n = len(x)
    nfft = 1 << max(1, (2 * n - 1).bit_length())
    freqs = np.fft.rfftfreq(nfft, 1.0 / C.SAMPLE_RATE)
    lf = np.log2(np.maximum(freqs, 20.0))
    gain_db = np.zeros_like(freqs)
    for _ in range(bumps):
        centre = rng.uniform(np.log2(150.0), np.log2(6000.0))
        width = rng.uniform(0.3, 1.5)
        gain_db += rng.uniform(-max_db, max_db) * np.exp(-0.5 * ((lf - centre) / width) ** 2)
    y = np.fft.irfft(np.fft.rfft(x, nfft) * 10.0 ** (gain_db / 20.0), nfft)[:n]
    return y.astype(np.float32)


@dataclass(frozen=True)
class EnrolAugmentConfig:
    """Augmentation of enrolment clips before embedding (plan: noise, RIR, EQ, Opus)."""

    p_noise: float = 0.7
    snr_db: tuple[float, float] = (5.0, 30.0)
    p_reverb: float = 0.5
    p_eq: float = 0.5
    eq_db: float = 6.0
    p_opus: float = 0.5
    opus_bitrate: int = 16000
    level_db: tuple[float, float] = (-35.0, -15.0)


class EnrolAugment:
    """Apply :class:`EnrolAugmentConfig` to a clip with a caller-supplied generator."""

    def __init__(
        self,
        config: EnrolAugmentConfig = EnrolAugmentConfig(),
        *,
        noise: ShardedCorpus | None = None,
        rirs: ShardedCorpus | None = None,
        opus: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.config = config
        self.noise = noise
        self.rirs = rirs
        self.opus = opus

    def __call__(self, clip: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.config
        y = np.asarray(clip, dtype=np.float64)
        n = y.size
        applied: dict[str, Any] = {}
        if self.rirs is not None and len(self.rirs) and rng.random() < cfg.p_reverb:
            row = int(rng.integers(len(self.rirs)))
            h = self.rirs.audio(row).astype(np.float64)
            peak = np.max(np.abs(h)) if h.size else 0.0
            if peak > 0:
                nfft = 1 << (n + h.size - 1).bit_length()
                y = np.fft.irfft(np.fft.rfft(y, nfft) * np.fft.rfft(h / peak, nfft), nfft)[:n]
                applied["rir"] = str(self.rirs.column("utt_id")[row])
        if rng.random() < cfg.p_eq:
            y = random_eq(y, rng, max_db=cfg.eq_db).astype(np.float64)
            applied["eq"] = True
        speech_power = float(active_power_np(y))
        if speech_power > 0:
            y = y * math.sqrt(10.0 ** (rng.uniform(*cfg.level_db) / 10.0) / speech_power)
            speech_power = float(active_power_np(y))
        if self.noise is not None and len(self.noise) and speech_power > 0 and rng.random() < cfg.p_noise:
            row = int(rng.integers(len(self.noise)))
            src = self.noise.audio(row).astype(np.float64)
            if src.size:
                idx = (int(rng.integers(src.size)) + np.arange(n)) % src.size
                noise = src[idx]
                p_noise = float(np.mean(noise**2))
                if p_noise > 0:
                    snr = rng.uniform(*cfg.snr_db)
                    y = y + noise * math.sqrt(speech_power / (10.0 ** (snr / 10.0)) / p_noise)
                    applied["noise"] = str(self.noise.column("utt_id")[row])
                    applied["snr_db"] = float(snr)
        peak = np.max(np.abs(y)) if y.size else 0.0
        if peak > 0.99:
            y = y * (0.99 / peak)
        if self.opus is not None and rng.random() < cfg.p_opus:
            y = self.opus(y.astype(np.float32)).astype(np.float64)
            applied["opus"] = True
        return y.astype(np.float32), applied


def compute_speaker_embeddings(
    corpus: ShardedCorpus,
    encoder: SpeakerEncoder,
    *,
    per_speaker: int = 8,
    seed: int = 0,
    clip_seconds: tuple[float, float] = (5.0, 10.0),
    augment: EnrolAugment | None = None,
    speakers: Iterable[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SpeakerEmbeddings:
    """Embed ``per_speaker`` augmented enrolment clips for every speaker with an enrolment pool.

    Each speaker's clips come from a generator seeded by ``(seed, speaker)``, so a
    speaker's embeddings do not depend on which other speakers are processed.
    """
    if not corpus.has_column("pool"):
        raise ValueError("corpus has no pool column; run shards.add_pools first")
    pool = corpus.column("pool")
    spk = corpus.speaker.astype(str)
    enrol_rows: dict[str, list[int]] = {}
    for i in np.flatnonzero(pool == POOL_ENROL):
        enrol_rows.setdefault(spk[i], []).append(int(i))
    names = sorted(enrol_rows if speakers is None else set(speakers) & set(enrol_rows))
    table = np.zeros((len(names), per_speaker, encoder.embedding_dim), dtype=np.float32)
    for s, name in enumerate(names):
        rng = np.random.default_rng([int(seed), stable_hash("enrol", name)])
        for k in range(per_speaker):
            clip = enrolment_clip(corpus, enrol_rows[name], rng, seconds=clip_seconds)
            if augment is not None:
                clip, _ = augment(clip, rng)
            emb = encoder.embed(torch.from_numpy(clip)[None])[0]
            table[s, k] = emb.numpy()
        if progress is not None:
            progress(s + 1, len(names))
    meta = {
        "encoder": getattr(encoder, "name", type(encoder).__name__),
        "per_speaker": per_speaker,
        "seed": seed,
        "clip_seconds": list(clip_seconds),
        "augment": None if augment is None else augment.config.__dict__,
        "contract_hash": C.CONTRACT_HASH,
    }
    return SpeakerEmbeddings(tuple(names), table, meta)
