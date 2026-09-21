"""PyTorch-stream dev runner: Earmark's ``step()`` path over an Earmark-Synth manifest.

The plan takes dev numbers from the PyTorch ``step()`` path from week 1 on. This module runs
a model hop by hop, as the engine will, over the mixtures an Earmark-Synth manifest describes
(:mod:`earmark.data.synth_bench`), and scores the output while it streams out:

* **SI-SDRi over unprocessed**: a :class:`~earmark.eval.stream_score.StreamingScorer` is fed
  every ``chunk_hops`` hops, so at most that much enhanced audio exists at any moment and
  none is ever written. The clean control (mixture == reference) has no SI-SDRi; its output
  SI-SDR is reported as the do-no-harm number instead.
* **VAD AUC**: frame-level ROC AUC of the personal-VAD head against the manifest's contract
  labels, pooled over all mixtures, with a speaker-cluster bootstrap interval.
* **False barge-ins per minute** (paired with onset recall, frame recall and median onset
  delay) at a threshold matched on this run at 95% target frame recall, or at a threshold
  given with ``--vad-threshold`` (frozen on dev; required for test).

Alignment. Output hop ``k`` of ``step()`` reconstructs input hop ``k - 1``
(:data:`~earmark.model.OUTPUT_DELAY_SAMPLES`), so the first output hop is dropped and one
zero hop is fed after the mixture to flush the last one: the scored output is time-aligned
with the mixture and has its length. Model frame ``t`` analyses contract frame ``t - 1``
(:data:`~earmark.model.MODEL_FRAME_OFFSET`), so VAD values ``1 .. F`` pair with the ``F``
contract labels.

Modes. ``personal`` conditions on the embedding of each mixture's ``enrol_utts`` (a frozen
speaker encoder, or a table precomputed where the encoder runs); ``denoise`` uses the learned
NULL embedding. Gate mode's audio is the raw mixture, so Gate's numbers are the VAD metrics of
the personal run.

Results go to a JSON file under ``results/dev/`` and one record is appended to
``results/runs.jsonl`` with ``inference_path="pytorch-stream"``. The pools file (YAML or
JSON) maps corpus names to prepared dataset folders, relative to the file::

    targets: ../devtest_16k/dev_clean
    interferers: ../devtest_16k/dev_other
    noise: [../noise_rir_16k/demand_eval, ../esc50_16k]
    rirs: ../noise_rir_16k/rir_real_eval
    agent: ../kokoro_agent_16k
    music: ../musan_music_16k/heldout

CLI::

    python -m earmark.eval.dev_runner --config M --checkpoint m_v1.pt \\
        --manifest data/earmark_synth/dev/manifest.parquet   # pools.yaml next to it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from earmark import constants as C
from earmark.data.embeddings import CodecUnavailable, SpeakerEncoder, StubSpeakerEncoder
from earmark.data.labels import num_frames
from earmark.data.shards import trim_silence
from earmark.data.synth_bench import (
    COND_CLEAN,
    CONDITIONS,
    BenchPools,
    MixtureSpec,
    load_bench_pools,
    read_manifest,
    render_mixture,
)
from earmark.eval.bargein import (
    BargeInCounts,
    GateConfig,
    binarize,
    gate_frames,
    matched_threshold,
    pool_counts,
    score_bargein,
)
from earmark.eval.bootstrap import DEFAULT_RESAMPLES, bootstrap_mean, cluster_bootstrap
from earmark.eval.runs_log import DEFAULT_RUNS_PATH, append_run, make_record, to_jsonable
from earmark.eval.stream_score import StreamingScorer, summarize_rows
from earmark.eval.systems import EnhancerOutput, SystemInfo
from earmark.model import (
    MODEL_FRAME_OFFSET,
    OUTPUT_DELAY_SAMPLES,
    EarmarkNet,
    build,
    config_for,
    state_size_bytes,
)
from earmark.model.macs import complexity

__all__ = [
    "ACCEPT_VAD_AUC",
    "DEFAULT_BATCH",
    "DEFAULT_CHUNK_HOPS",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_TARGET_RECALL",
    "INFERENCE_PATH",
    "MODES",
    "NO_SI_SDRI_CONDITIONS",
    "POOLS_FILE_NAME",
    "SUITE",
    "SUMMARY_KEYS",
    "DevRun",
    "EarmarkStreamSystem",
    "EnrolmentEmbedder",
    "ItemResult",
    "PrecomputedEmbeddings",
    "StreamingModel",
    "acceptance",
    "apply_threshold",
    "check_pools_cover",
    "choose_threshold",
    "default_out_path",
    "enrolment_key",
    "extract_state_dict",
    "file_sha256",
    "load_enrolment_embeddings",
    "load_manifest",
    "load_model",
    "load_pools_file",
    "main",
    "roc_auc",
    "run_dev",
    "save_enrolment_embeddings",
    "score_run",
    "select_specs",
    "stream_batch",
    "summarize_run",
    "system_info",
    "vad_auc_summary",
]

Mode = Literal["personal", "denoise"]
ThresholdLevel = Literal["frame", "onset"]

INFERENCE_PATH: Final[str] = "pytorch-stream"
SUITE: Final[str] = "earmark-synth"
MODES: Final[tuple[str, ...]] = ("personal", "denoise")
DEFAULT_BATCH: Final[int] = 8
#: Output reaches the scorers every 50 hops (0.5 s): the most enhanced audio held per stream.
DEFAULT_CHUNK_HOPS: Final[int] = 50
DEFAULT_TARGET_RECALL: Final[float] = 0.95
#: Week-1 acceptance: the mini-full run's VAD head reaches this pooled frame-level AUC on dev.
ACCEPT_VAD_AUC: Final[float] = 0.9
DEFAULT_RESULTS_DIR: Final[Path] = DEFAULT_RUNS_PATH.parent / "dev"
POOLS_FILE_NAME: Final[str] = "pools.yaml"
#: Conditions whose mixture equals the reference: SI-SDRi over unprocessed is undefined there
#: (the unprocessed SI-SDR is float rounding noise), so they are judged by output SI-SDR (H5).
NO_SI_SDRI_CONDITIONS: Final[frozenset[str]] = frozenset({COND_CLEAN})
#: H5 do-no-harm target for the clean control's output SI-SDR, reported for context.
DO_NO_HARM_SI_SDR_DB: Final[float] = 20.0
#: Per-mixture metrics summarised with speaker-cluster bootstrap intervals.
SUMMARY_KEYS: Final[tuple[str, ...]] = (
    "si_sdri",
    "si_sdr",
    "si_sdr_mixture",
    "interferer_suppression_db",
)

_POOL_KEYS: Final[tuple[str, ...]] = ("targets", "interferers", "noise", "rirs", "agent", "music")
_POOL_SPLIT_KEYS: Final[tuple[str, ...]] = ("agent_splits", "music_splits")
_STATE_DICT_KEYS: Final[tuple[str, ...]] = ("model", "model_state_dict", "state_dict", "net")
_WRAPPER_PREFIXES: Final[tuple[str, ...]] = ("module.", "_orig_mod.")

_HOP: Final[int] = C.HOP_LENGTH
if OUTPUT_DELAY_SAMPLES % _HOP:
    raise RuntimeError("the dev runner needs a model output delay of whole hops")
_DELAY_HOPS: Final[int] = OUTPUT_DELAY_SAMPLES // _HOP

FloatArray = NDArray[np.float64]
Sink = Callable[[int, int, NDArray[np.float32]], None]
EmbeddingSource = Callable[[MixtureSpec], NDArray[np.float32]]


# --------------------------------------------------------------------------------------------
# Streaming core


class StreamingModel(Protocol):
    """What the runner needs from a model: :class:`~earmark.model.EarmarkNet` satisfies it."""

    def condition(self, emb: Tensor | None = None, *, batch: int | None = None) -> Any: ...

    def init_state(
        self,
        batch: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Any: ...

    def step(self, frame: Tensor, emb: Any, state: Any) -> tuple[Tensor, Tensor, Any]: ...


@torch.inference_mode()
def stream_batch(
    net: StreamingModel,
    mixtures: Sequence[ArrayLike],
    embeddings: Tensor | None = None,
    *,
    sink: Sink | None = None,
    chunk_hops: int = DEFAULT_CHUNK_HOPS,
    device: torch.device | str = "cpu",
) -> list[NDArray[np.float32]]:
    """Run a batch of mono 16 kHz mixtures through ``net.step`` one hop at a time.

    Mixtures may differ in length: each is zero-padded to the longest one plus the flush
    hop, which cannot change its own output because the model is causal. ``sink(b, start,
    chunk)`` receives item ``b``'s output samples ``[start, start + len(chunk))``, aligned
    with its mixture, as soon as ``chunk_hops`` hops are ready; samples past an item's end
    are never emitted. ``embeddings`` is ``[B, D]`` (``None`` selects the NULL embedding).

    Returns each item's VAD probabilities, one per contract frame of its mixture.
    """
    if chunk_hops < 1:
        raise ValueError("chunk_hops must be >= 1")
    if not mixtures:
        return []
    arrays = [np.asarray(m, dtype=np.float32) for m in mixtures]
    if any(a.ndim != 1 or a.size == 0 for a in arrays):
        raise ValueError("every mixture must be a non-empty 1-D array")
    dev = torch.device(device)
    lengths = [a.size for a in arrays]
    batch = len(arrays)
    steps = -(-max(lengths) // _HOP) + _DELAY_HOPS
    x = torch.zeros(batch, steps * _HOP, dtype=torch.float32)
    for b, a in enumerate(arrays):
        x[b, : a.size] = torch.from_numpy(a)
    x = x.to(dev)
    emb = None if embeddings is None else embeddings.to(device=dev, dtype=torch.float32)
    cond = net.condition(emb, batch=batch)
    state = net.init_state(batch, dev, torch.float32)
    vads: list[Tensor] = []
    pending: list[Tensor] = []
    start = 0  # aligned sample index of the first pending output sample
    for k in range(steps):
        out, vad, state = net.step(x[:, k * _HOP : (k + 1) * _HOP], cond, state)
        vads.append(vad)
        if k < _DELAY_HOPS:
            continue  # reconstructs samples from before the input starts
        pending.append(out)
        if len(pending) == chunk_hops or k == steps - 1:
            chunk = torch.cat(pending, dim=-1).float().cpu().numpy()
            pending.clear()
            if sink is not None:
                for b, n in enumerate(lengths):
                    stop = min(n, start + chunk.shape[1])
                    if stop > start:
                        sink(b, start, chunk[b, : stop - start])
            start += chunk.shape[1]
    vad_all = torch.stack(vads, dim=-1).float().cpu().numpy()
    return [
        vad_all[b, MODEL_FRAME_OFFSET : MODEL_FRAME_OFFSET + num_frames(n)].copy()
        for b, n in enumerate(lengths)
    ]


class EarmarkStreamSystem:
    """:class:`~earmark.eval.systems.Enhancer` adapter over the same ``step()`` path.

    For suites that go through :func:`~earmark.eval.stream_score.score_items` (Suite B in
    Denoise mode, ...). ``enhance`` streams one mixture and returns its latency-compensated
    output (the mixture's length) and one VAD probability per contract frame.
    """

    sample_rate: int = C.SAMPLE_RATE

    def __init__(
        self,
        net: StreamingModel,
        *,
        mode: Mode = "personal",
        info: SystemInfo | None = None,
        device: torch.device | str = "cpu",
        chunk_hops: int = DEFAULT_CHUNK_HOPS,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.net = net
        self.mode = mode
        self.device = device
        self.chunk_hops = chunk_hops
        self.info = info or SystemInfo(name=f"earmark-{mode}", inference_path=INFERENCE_PATH)
        if self.info.inference_path != INFERENCE_PATH:
            raise ValueError(f"the step() path is labelled {INFERENCE_PATH!r}")

    def enhance(
        self, mixture: NDArray[np.float32], embedding: NDArray[np.float32] | None = None
    ) -> EnhancerOutput:
        x = np.asarray(mixture, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(f"mixture must be 1-D, got shape {x.shape}")
        emb = None
        if self.mode == "personal":
            if embedding is None:
                raise ValueError("personal mode needs the enrolment embedding")
            emb = torch.as_tensor(np.asarray(embedding, dtype=np.float32)).reshape(1, -1)
        out = np.zeros(x.size, dtype=np.float32)

        def sink(_b: int, start: int, chunk: NDArray[np.float32]) -> None:
            out[start : start + chunk.size] = chunk

        vad = stream_batch(
            self.net, [x], emb, sink=sink, chunk_hops=self.chunk_hops, device=self.device
        )
        return EnhancerOutput(out, vad[0])


# --------------------------------------------------------------------------------------------
# Enrolment embeddings


def enrolment_key(spec: MixtureSpec) -> str:
    """Digest of a mixture's enrolment utterance list (guards precomputed tables)."""
    return hashlib.sha256("\n".join(spec.enrol_utts).encode("utf-8")).hexdigest()[:16]


class EnrolmentEmbedder:
    """Embeds each mixture's enrolment utterances with a frozen speaker encoder.

    The clip is every utterance in ``spec.enrol_utts`` (from the target corpus, a different
    chapter than the target), silence-trimmed and joined with ``gap_seconds`` of silence in
    manifest order, so it is deterministic. Embeddings are L2-normalised and cached by
    utterance list.
    """

    def __init__(
        self, pools: BenchPools, encoder: SpeakerEncoder, *, gap_seconds: float = 0.1
    ) -> None:
        self.pools = pools
        self.encoder = encoder
        self.gap = np.zeros(int(round(gap_seconds * C.SAMPLE_RATE)), dtype=np.float32)
        self._cache: dict[tuple[str, ...], NDArray[np.float32]] = {}

    @property
    def name(self) -> str:
        return str(getattr(self.encoder, "name", type(self.encoder).__name__))

    def clip(self, spec: MixtureSpec) -> NDArray[np.float32]:
        """The enrolment clip of one mixture."""
        pieces: list[NDArray[np.float32]] = []
        for utt in spec.enrol_utts:
            audio = self.pools.targets.audio(self.pools.row("targets", utt))
            begin, end = trim_silence(audio)
            if end > begin:
                if pieces:
                    pieces.append(self.gap)
                pieces.append(audio[begin:end])
        if not pieces:
            raise ValueError(f"{spec.mixture_id}: enrolment utterances are missing or silent")
        return np.concatenate(pieces).astype(np.float32)

    def __call__(self, spec: MixtureSpec) -> NDArray[np.float32]:
        key = tuple(spec.enrol_utts)
        if key not in self._cache:
            emb = self.encoder.embed(torch.from_numpy(self.clip(spec))[None])[0]
            vec = emb.detach().cpu().numpy().astype(np.float64)
            self._cache[key] = (vec / max(float(np.linalg.norm(vec)), 1e-12)).astype(np.float32)
        return self._cache[key]


class PrecomputedEmbeddings:
    """Per-mixture enrolment embeddings saved by :func:`save_enrolment_embeddings`.

    Lets a machine without the speaker encoder (the Mac has no onnxruntime) run Personal
    mode with embeddings computed where the encoder runs. Each lookup checks that the
    mixture's enrolment list is the one the embedding was computed from.
    """

    def __init__(
        self,
        mixture_ids: Sequence[str],
        keys: Sequence[str],
        table: ArrayLike,
        *,
        encoder: str,
        path: str = "",
    ) -> None:
        vectors = np.asarray(table, dtype=np.float32)
        n = len(mixture_ids)
        if vectors.ndim != 2 or vectors.shape[0] != n or len(keys) != n:
            raise ValueError("embedding table does not match its mixture ids")
        self._rows = {
            str(m): (str(k), vectors[i])
            for i, (m, k) in enumerate(zip(mixture_ids, keys, strict=True))
        }
        self.encoder = encoder
        self.path = path

    @property
    def name(self) -> str:
        return f"precomputed:{self.encoder}"

    def __len__(self) -> int:
        return len(self._rows)

    def check(self, specs: Sequence[MixtureSpec]) -> None:
        """Raise ``ValueError`` unless every spec has a matching embedding (before a long run)."""
        missing = [s.mixture_id for s in specs if s.mixture_id not in self._rows]
        stale = [
            s.mixture_id
            for s in specs
            if s.mixture_id in self._rows and self._rows[s.mixture_id][0] != enrolment_key(s)
        ]
        if missing:
            raise ValueError(
                f"no embedding for {len(missing)} selected mixtures in {self.path or 'the table'} "
                f"(e.g. {missing[:5]})"
            )
        if stale:
            raise ValueError(
                f"{len(stale)} embeddings were computed from a different enrolment list "
                f"(e.g. {stale[:5]})"
            )

    def __call__(self, spec: MixtureSpec) -> NDArray[np.float32]:
        if spec.mixture_id not in self._rows:
            where = self.path or "the table"
            raise KeyError(f"no precomputed embedding for {spec.mixture_id} in {where}")
        key, vec = self._rows[spec.mixture_id]
        if key != enrolment_key(spec):
            raise ValueError(
                f"{spec.mixture_id}: the embedding was computed from a different enrolment list"
            )
        return vec


def save_enrolment_embeddings(
    path: str | Path,
    specs: Sequence[MixtureSpec],
    vectors: Mapping[str, ArrayLike],
    *,
    encoder: str,
) -> Path:
    """Write the embeddings of ``specs`` (``vectors[mixture_id]``) as an ``.npz`` table."""
    chosen = [s for s in specs if s.mixture_id in vectors]
    if not chosen:
        raise ValueError("no embeddings to save")
    table = np.stack([np.asarray(vectors[s.mixture_id], dtype=np.float32) for s in chosen])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        np.savez(
            fh,
            mixture_ids=np.array([s.mixture_id for s in chosen], dtype=str),
            enrol_keys=np.array([enrolment_key(s) for s in chosen], dtype=str),
            embeddings=table,
            encoder=np.array(encoder),
            contract_hash=np.array(C.CONTRACT_HASH),
        )
    return path


def load_enrolment_embeddings(path: str | Path) -> PrecomputedEmbeddings:
    """Read a table written by :func:`save_enrolment_embeddings`."""
    with np.load(Path(path), allow_pickle=False) as z:
        return PrecomputedEmbeddings(
            z["mixture_ids"].tolist(),
            z["enrol_keys"].tolist(),
            z["embeddings"],
            encoder=str(z["encoder"]),
            path=str(path),
        )


# --------------------------------------------------------------------------------------------
# Running a manifest


@dataclass
class ItemResult:
    """Numbers for one mixture: the JSON row plus its VAD probabilities and contract labels."""

    spec: MixtureSpec
    row: dict[str, Any]
    vad: NDArray[np.float32]
    labels: NDArray[np.bool_]
    bargein: BargeInCounts | None = None


@dataclass
class DevRun:
    """Everything :func:`run_dev` produced (no audio)."""

    items: list[ItemResult]
    skipped: list[dict[str, str]]
    mode: str
    audio_seconds: float
    elapsed_seconds: float
    embeddings: dict[str, NDArray[np.float32]] = field(default_factory=dict)


def _check_unique_ids(specs: Sequence[MixtureSpec]) -> None:
    dupes = sorted(k for k, v in Counter(s.mixture_id for s in specs).items() if v > 1)
    if dupes:
        raise ValueError(f"duplicate mixture ids: {dupes[:5]}")


def _finite(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _item_result(
    spec: MixtureSpec,
    rendered: Mapping[str, Any],
    scores: Mapping[str, Any],
    vad: NDArray[np.float32],
) -> ItemResult:
    labels = np.asarray(rendered["vad"], dtype=bool)
    if labels.shape != vad.shape:
        raise RuntimeError(
            f"{spec.mixture_id}: {vad.size} VAD values for {labels.size} contract labels"
        )
    has_ref = spec.target_present and spec.reference_clean
    si_sdr = _finite(scores.get("si_sdr")) if has_ref else math.nan
    si_sdri = _finite(scores.get("si_sdri")) if has_ref else math.nan
    si_sdr_mix = si_sdr - si_sdri
    if spec.condition in NO_SI_SDRI_CONDITIONS:
        si_sdri = si_sdr_mix = math.nan
    row: dict[str, Any] = {
        "item_id": spec.mixture_id,
        "cluster": spec.target_speaker,
        "condition": spec.condition,
        "target_present": bool(spec.target_present),
        "interferer_kind": int(spec.interferer_kind),
        "sir_db": spec.sir_db,
        "snr_db": spec.snr_db,
        "reverberant": spec.target_rir is not None,
        "loudspeaker": spec.loudspeaker is not None,
        "codec": spec.codec,
        "duration_s": float(np.asarray(rendered["mixture"]).size / C.SAMPLE_RATE),
        "si_sdr": si_sdr,
        "si_sdr_mixture": si_sdr_mix,
        "si_sdri": si_sdri,
        "tsos_percent": _finite(scores.get("tsos_percent")) if has_ref else math.nan,
        "tsos_active_frames": int(scores.get("tsos_active_frames", 0)) if has_ref else 0,
        "tsos_os_frames": int(scores.get("tsos_os_frames", 0)) if has_ref else 0,
        "interferer_suppression_db": _finite(scores.get("interferer_suppression_db")),
        "vad_frames": int(labels.size),
        "vad_active_frames": int(labels.sum()),
        "vad_auc": roc_auc(vad, labels) if labels.size else math.nan,
    }
    return ItemResult(spec=spec, row=row, vad=vad, labels=labels)


def run_dev(
    net: StreamingModel,
    specs: Sequence[MixtureSpec],
    pools: BenchPools,
    *,
    mode: Mode = "personal",
    embed: EmbeddingSource | None = None,
    batch: int = DEFAULT_BATCH,
    chunk_hops: int = DEFAULT_CHUNK_HOPS,
    device: torch.device | str = "cpu",
    apply_codec: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> DevRun:
    """Render each mixture, stream it through ``net.step`` and score it as it streams.

    Mixtures are rendered from their specs in memory, ``batch`` at a time, and dropped once
    scored. A mixture whose codec cannot be applied here (no ffmpeg with libopus) is skipped
    and listed in :attr:`DevRun.skipped`, never scored uncoded.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "personal" and embed is None:
        raise ValueError("personal mode needs an embedding source")
    if batch < 1:
        raise ValueError("batch must be >= 1")
    _check_unique_ids(specs)
    started = time.perf_counter()
    items: list[ItemResult] = []
    skipped: list[dict[str, str]] = []
    embeddings: dict[str, NDArray[np.float32]] = {}
    pending: list[tuple[MixtureSpec, dict[str, Any], NDArray[np.float32] | None]] = []
    audio_seconds = 0.0

    def flush() -> None:
        nonlocal audio_seconds
        if not pending:
            return
        rendered = [r for _, r, _ in pending]
        scorers = [StreamingScorer() for _ in pending]

        def sink(b: int, start: int, chunk: NDArray[np.float32]) -> None:
            r, stop = rendered[b], start + chunk.size
            scorers[b].update(
                r["reference"][start:stop],
                chunk,
                r["mixture"][start:stop],
                r["interferer"][start:stop],
            )

        emb = None
        if mode == "personal":
            emb = torch.from_numpy(np.stack([e for _, _, e in pending if e is not None]))
        vads = stream_batch(
            net,
            [r["mixture"] for r in rendered],
            emb,
            sink=sink,
            chunk_hops=chunk_hops,
            device=device,
        )
        for (spec, r, _), scorer, vad in zip(pending, scorers, vads, strict=True):
            items.append(_item_result(spec, r, scorer.result(), vad))
            audio_seconds += np.asarray(r["mixture"]).size / C.SAMPLE_RATE
        pending.clear()
        if progress is not None:
            progress(len(items) + len(skipped), len(specs))

    for spec in specs:
        try:
            rendered = render_mixture(spec, pools, apply_codec=apply_codec)
        except CodecUnavailable as exc:
            reason = f"codec {spec.codec} unavailable here: {exc}"
            skipped.append(
                {"item_id": spec.mixture_id, "condition": spec.condition, "reason": reason}
            )
            continue
        vec = None
        if mode == "personal" and embed is not None:
            vec = np.asarray(embed(spec), dtype=np.float32)
            embeddings[spec.mixture_id] = vec
        pending.append((spec, rendered, vec))
        if len(pending) == batch:
            flush()
    flush()
    return DevRun(
        items=items,
        skipped=skipped,
        mode=mode,
        audio_seconds=audio_seconds,
        elapsed_seconds=time.perf_counter() - started,
        embeddings=embeddings,
    )


# --------------------------------------------------------------------------------------------
# VAD AUC


def _auc_from_counts(pos: FloatArray, neg: FloatArray) -> float:
    """AUC from positive/negative counts per ascending score bin (ties count one half)."""
    p_total, n_total = float(pos.sum()), float(neg.sum())
    if p_total <= 0.0 or n_total <= 0.0:
        return math.nan
    below = np.cumsum(neg) - neg
    return float(np.dot(pos, below + 0.5 * neg) / (p_total * n_total))


def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Exact ROC AUC (Mann-Whitney U / (P N), ties count one half); NaN with one class only."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).astype(bool).ravel()
    if s.shape != y.shape:
        raise ValueError(f"{s.size} scores for {y.size} labels")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores contain NaN or inf")
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return math.nan
    _, inverse = np.unique(s, return_inverse=True)
    k = int(inverse.max()) + 1
    pos = np.bincount(inverse[y], minlength=k).astype(np.float64)
    neg = np.bincount(inverse[~y], minlength=k).astype(np.float64)
    return _auc_from_counts(pos, neg)


def vad_auc_summary(
    scores: Sequence[ArrayLike],
    labels: Sequence[ArrayLike],
    clusters: Sequence[str] | None = None,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    max_bins: int = 4096,
) -> dict[str, Any]:
    """Pooled frame-level VAD AUC with a cluster (speaker) bootstrap interval.

    The point estimate is exact. For the interval, frames are grouped into at most
    ``max_bins`` score-quantile bins (tied scores always share a bin) and summed per
    cluster, so each of the ``n_resamples`` resamples costs one small matrix product.
    """
    s_list = [np.asarray(s, dtype=np.float64).ravel() for s in scores]
    y_list = [np.asarray(lab).astype(bool).ravel() for lab in labels]
    if len(s_list) != len(y_list):
        raise ValueError(f"{len(s_list)} score arrays for {len(y_list)} label arrays")
    if any(a.shape != b.shape for a, b in zip(s_list, y_list, strict=True)):
        raise ValueError("each score array must match its label array")
    lengths = np.array([a.size for a in s_list], dtype=np.int64)
    s = np.concatenate(s_list) if s_list else np.zeros(0)
    y = np.concatenate(y_list) if y_list else np.zeros(0, dtype=bool)
    n_pos = int(y.sum())
    out: dict[str, Any] = {
        "auc": roc_auc(s, y) if y.size else math.nan,
        "ci95": [math.nan, math.nan],
        "n_frames": int(y.size),
        "n_active_frames": n_pos,
        "n_items": len(s_list),
    }
    if n_pos == 0 or n_pos == y.size or n_resamples < 1:
        return out
    if clusters is None:
        names = [str(i) for i in range(len(s_list))]
    else:
        names = [str(c) for c in clusters]
    if len(names) != len(s_list):
        raise ValueError(f"{len(names)} clusters for {len(s_list)} items")
    _, unit = np.unique(np.asarray(names), return_inverse=True)
    n_units = int(unit.max()) + 1
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    min_rank = np.cumsum(counts) - counts
    bins = max(1, min(max_bins, counts.size, max(64, 4_000_000 // n_units)))
    score_bin = (min_rank[inverse] * bins) // s.size
    cell = np.repeat(unit.astype(np.int64), lengths) * bins + score_bin
    size = n_units * bins
    pos_h = np.bincount(cell[y], minlength=size).reshape(n_units, bins).astype(np.float64)
    neg_h = np.bincount(cell[~y], minlength=size).reshape(n_units, bins).astype(np.float64)

    def statistic(idx: NDArray[np.int64]) -> float:
        w = np.bincount(idx, minlength=n_units).astype(np.float64)
        return _auc_from_counts(w @ pos_h, w @ neg_h)

    boot = cluster_bootstrap(
        statistic, n_units, n_resamples=n_resamples, seed=seed, name="vad_auc"
    )
    out.update(
        ci95=[boot.low, boot.high],
        n_units=n_units,
        unit="cluster" if clusters is not None else "item",
        n_bins=bins,
        n_invalid=boot.n_invalid,
    )
    return out


# --------------------------------------------------------------------------------------------
# Thresholds, barge-ins and the summary


def choose_threshold(
    items: Sequence[ItemResult],
    *,
    fixed: float | None = None,
    target_recall: float = DEFAULT_TARGET_RECALL,
    level: ThresholdLevel = "frame",
) -> dict[str, Any]:
    """The VAD threshold for barge-in scoring: ``fixed`` if given, else matched on ``items``.

    Raises ``ValueError`` when no item has a target-active frame to match recall on.
    """
    if fixed is not None:
        if not 0.0 <= fixed <= 1.0:
            raise ValueError(f"a VAD threshold is a probability in [0, 1], got {fixed}")
        return {"threshold": float(fixed), "source": "fixed", "level": "fixed"}
    usable = [it for it in items if it.labels.size]
    frozen = matched_threshold(
        [it.vad for it in usable], [it.labels for it in usable], target_recall, level=level
    )
    return {"source": "matched-on-this-run", **frozen.as_dict()}


def _bargein_fields(counts: BargeInCounts) -> dict[str, Any]:
    """Row fields, named as :func:`~earmark.eval.stream_score.pooled_bargein` expects."""
    fields = {f"bargein_{k}": v for k, v in counts.summary().items()}
    fields["bargein_delays_frames"] = list(counts.delays_frames)
    fields["bargein_silent_frames"] = counts.silent_frames
    return fields


def apply_threshold(
    items: Sequence[ItemResult], threshold: float, gate: GateConfig | None = None
) -> None:
    """Score barge-ins of every item (adds ``bargein_*`` fields to its row).

    Frames are decided by ``scores >= threshold``, or by ``gate`` when one is given (the
    gate's own attack replaces ``threshold``). The reference onsets are unaffected either way.
    """
    config = gate if gate is not None else GateConfig(attack=float(threshold))
    for it in items:
        counts = score_bargein(gate_frames(it.vad, config), it.labels)
        it.bargein = counts
        it.row.update(_bargein_fields(counts))


def save_frames(items: Sequence[ItemResult], path: Path) -> Path:
    """Write every item's VAD scores and reference activity to one ``.npz``.

    Items are concatenated with an offsets array, because they differ in length. This is
    what :mod:`earmark.eval.tune_gate` reads, so gate settings can be searched without
    running the model again.
    """
    usable = [it for it in items if it.labels.size]
    scores = np.concatenate([it.vad for it in usable]) if usable else np.zeros(0, np.float32)
    labels = np.concatenate([it.labels for it in usable]) if usable else np.zeros(0, bool)
    lengths = [int(it.labels.size) for it in usable]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scores=scores.astype(np.float32),
        labels=labels.astype(bool),
        offsets=np.cumsum([0, *lengths], dtype=np.int64),
        item_ids=np.array([str(it.row.get("item_id", index)) for index, it in enumerate(usable)]),
        frame_rate_hz=np.int64(C.FRAME_RATE_HZ),
    )
    return path


def _rate_ci(
    values: Sequence[float],
    denominators: Sequence[float],
    clusters: Sequence[str],
    n_resamples: int,
    seed: int,
) -> dict[str, Any] | None:
    """Ratio of sums with a cluster bootstrap interval (``None`` without exposure)."""
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(denominators, dtype=np.float64)
    if v.size == 0 or float(d.sum()) <= 0.0 or n_resamples < 1:
        return None
    b = bootstrap_mean(
        v, clusters=list(clusters), denominators=d, n_resamples=n_resamples, seed=seed
    )
    return {
        "estimate": b.estimate,
        "ci95": [b.low, b.high],
        "n_units": b.n_units,
        "n_invalid": b.n_invalid,
    }


def _bargein_summary(
    items: Sequence[ItemResult],
    threshold: Mapping[str, Any] | None,
    n_resamples: int,
    seed: int,
) -> dict[str, Any] | None:
    scored = [it for it in items if it.bargein is not None]
    if threshold is None or not scored:
        return None
    counts = [it.bargein for it in scored if it.bargein is not None]
    clusters = [str(it.row["cluster"]) for it in scored]
    silent_minutes = [c.silent_frames / (60.0 * C.FRAME_RATE_HZ) for c in counts]
    out: dict[str, Any] = {"threshold": dict(threshold), **pool_counts(counts).summary()}
    out["ci95"] = {
        "false_barge_ins_per_min": _rate_ci(
            [c.false_barge_ins for c in counts], silent_minutes, clusters, n_resamples, seed
        ),
        "onset_recall": _rate_ci(
            [c.hits for c in counts],
            [c.reference_onsets for c in counts],
            clusters,
            n_resamples,
            seed,
        ),
        "frame_recall": _rate_ci(
            [c.detected_active_frames for c in counts],
            [c.active_frames for c in counts],
            clusters,
            n_resamples,
            seed,
        ),
    }
    return out


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else math.nan


def _by_condition(items: Sequence[ItemResult]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[ItemResult]] = {}
    for it in items:
        groups.setdefault(str(it.row["condition"]), []).append(it)
    table: dict[str, dict[str, Any]] = {}
    for cond in sorted(groups):
        sel = groups[cond]
        scores = np.concatenate([it.vad for it in sel])
        labels = np.concatenate([it.labels for it in sel])
        entry: dict[str, Any] = {
            "n": len(sel),
            "n_target_present": sum(1 for it in sel if it.row["target_present"]),
            "si_sdri": _mean([it.row["si_sdri"] for it in sel]),
            "si_sdr": _mean([it.row["si_sdr"] for it in sel]),
            "vad_auc": roc_auc(scores, labels) if labels.size else math.nan,
        }
        counts = [it.bargein for it in sel if it.bargein is not None]
        if counts:
            pooled = pool_counts(counts)
            entry.update(
                false_barge_ins_per_min=pooled.false_per_minute,
                onset_recall=pooled.onset_recall,
                frame_recall=pooled.frame_recall,
                median_onset_delay_ms=pooled.median_delay_ms,
            )
        table[cond] = entry
    return table


def summarize_run(
    run: DevRun,
    *,
    threshold: Mapping[str, Any] | None,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """Headline numbers with speaker-cluster bootstrap intervals, per-condition tables and
    the week-1 acceptance block. Call :func:`apply_threshold` first for barge-in numbers."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    items = run.items
    rows = [it.row for it in items]
    clusters = [str(r["cluster"]) for r in rows]
    tsos_rows = [r for r in rows if r["tsos_active_frames"] > 0]
    tsos = _rate_ci(
        [r["tsos_os_frames"] for r in tsos_rows],
        [r["tsos_active_frames"] for r in tsos_rows],
        [str(r["cluster"]) for r in tsos_rows],
        n_resamples,
        seed,
    )
    tsos_pooled = None
    if tsos is not None:
        tsos_pooled = {
            "estimate": 100.0 * tsos["estimate"],
            "ci95": [100.0 * v for v in tsos["ci95"]],
            "n_items": len(tsos_rows),
        }
    clean = [
        r["si_sdr"]
        for r in rows
        if r["condition"] in NO_SI_SDRI_CONDITIONS and math.isfinite(r["si_sdr"])
    ]
    metrics = {}
    if rows:
        metrics = summarize_rows(
            rows, SUMMARY_KEYS, cluster_level=True, n_resamples=n_resamples, seed=seed
        )
    vad = vad_auc_summary(
        [it.vad for it in items],
        [it.labels for it in items],
        clusters,
        n_resamples=n_resamples,
        seed=seed,
    )
    audio_s, elapsed_s = run.audio_seconds, run.elapsed_seconds
    summary: dict[str, Any] = {
        "n_items": len(items),
        "n_skipped": len(run.skipped),
        "skipped": list(run.skipped),
        "n_speakers": len(set(clusters)),
        "per_condition_counts": dict(sorted(Counter(r["condition"] for r in rows).items())),
        "metrics": metrics,
        "tsos_pooled_percent": tsos_pooled,
        "vad": vad,
        "bargein": _bargein_summary(items, threshold, n_resamples, seed),
        "by_condition": _by_condition(items),
        "do_no_harm": {
            "conditions": sorted(NO_SI_SDRI_CONDITIONS),
            "n": len(clean),
            "si_sdr_mean_db": _mean(clean),
            "si_sdr_min_db": float(min(clean)) if clean else math.nan,
            "h5_target_db": DO_NO_HARM_SI_SDR_DB,
        },
        "timing": {
            "audio_s": audio_s,
            "elapsed_s": elapsed_s,
            "rtf_end_to_end": elapsed_s / audio_s if audio_s else math.nan,
        },
    }
    summary["acceptance"] = acceptance(summary)
    return summary


def score_run(
    run: DevRun,
    *,
    vad_threshold: float | None = None,
    target_recall: float = DEFAULT_TARGET_RECALL,
    threshold_level: ThresholdLevel = "frame",
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    gate: GateConfig | None = None,
) -> dict[str, Any]:
    """Choose the threshold, score barge-ins and summarise (the CLI's scoring step).

    ``gate`` replaces plain thresholding with hysteresis and gap bridging; its attack is
    the chosen threshold, so the threshold is still matched the same way.
    """
    note = None
    threshold: dict[str, Any] | None
    if vad_threshold is not None:
        threshold = choose_threshold(run.items, fixed=vad_threshold)
    else:
        try:
            threshold = choose_threshold(
                run.items, target_recall=target_recall, level=threshold_level
            )
        except ValueError as exc:
            threshold, note = None, f"no barge-in scores: {exc}"
    if threshold is not None:
        chosen = float(threshold["threshold"])
        config = None if gate is None else replace(gate, attack=chosen)
        apply_threshold(run.items, chosen, config)
        if config is not None:
            threshold = {**threshold, "gate": config.as_dict()}
    summary = summarize_run(run, threshold=threshold, n_resamples=n_resamples, seed=seed)
    if note is not None:
        summary["bargein_note"] = note
    return summary


def acceptance(summary: Mapping[str, Any], *, min_auc: float = ACCEPT_VAD_AUC) -> dict[str, Any]:
    """Week-1 done-when on dev: SI-SDRi improves over unprocessed and VAD AUC >= ``min_auc``.

    "Improves" means the speaker-cluster 95% interval of the mean SI-SDRi lies above 0 dB,
    so a noisy pilot cannot pass on its point estimate alone.
    """
    si = summary.get("metrics", {}).get("si_sdri", {}) or {}
    ci = si.get("ci95") or [math.nan, math.nan]
    mean, low = _finite(si.get("mean")), _finite(ci[0])
    auc = _finite((summary.get("vad") or {}).get("auc"))
    si_ok = math.isfinite(low) and low > 0.0
    auc_ok = math.isfinite(auc) and auc >= min_auc
    return {
        "si_sdri_improves": {
            "criterion": "speaker-cluster 95% CI of mean SI-SDRi over unprocessed lies above 0 dB",
            "mean_db": mean,
            "ci95_low_db": low,
            "passed": si_ok,
        },
        "vad_auc": {
            "criterion": f"pooled frame-level VAD AUC >= {min_auc}",
            "value": auc,
            "passed": auc_ok,
        },
        "passed": si_ok and auc_ok,
    }


# --------------------------------------------------------------------------------------------
# Loading: model, manifest, pools


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_tensor_mapping(obj: Any) -> bool:
    if not isinstance(obj, Mapping) or not obj:
        return False
    return all(isinstance(v, Tensor) for v in obj.values())


def extract_state_dict(obj: Any) -> dict[str, Tensor]:
    """The model ``state_dict`` inside a loaded checkpoint.

    Accepts a bare ``state_dict`` or a training checkpoint holding it under one of
    ``model``, ``model_state_dict``, ``state_dict`` or ``net``; strips the ``module.``
    (DataParallel) and ``_orig_mod.`` (torch.compile) prefixes.
    """
    if _is_tensor_mapping(obj):
        state = dict(obj)
    elif isinstance(obj, Mapping) and any(_is_tensor_mapping(obj.get(k)) for k in _STATE_DICT_KEYS):
        state = dict(next(obj[k] for k in _STATE_DICT_KEYS if _is_tensor_mapping(obj.get(k))))
    else:
        raise ValueError(
            f"no model state_dict found (a tensor mapping, or one under {_STATE_DICT_KEYS})"
        )
    stripped = True
    while stripped:
        stripped = False
        for prefix in _WRAPPER_PREFIXES:
            if all(k.startswith(prefix) for k in state):
                state = {k[len(prefix) :]: v for k, v in state.items()}
                stripped = True
    return state


def load_model(
    config: str,
    checkpoint: str | Path | None = None,
    *,
    seed: int = 0,
    trust_checkpoint: bool = False,
) -> tuple[EarmarkNet, dict[str, Any]]:
    """Build ``config`` (seeded random init) and load ``checkpoint`` into it (strict).

    Checkpoints load with ``weights_only=True`` unless ``trust_checkpoint``; never unpickle a
    file you did not produce. Returns ``(net in eval mode on the CPU, provenance dict)``.
    """
    cfg = config_for(config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = build(cfg)
    info: dict[str, Any] = {
        "config": cfg.name,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "random_weights": checkpoint is None,
        "init_seed": seed if checkpoint is None else None,
    }
    if checkpoint is not None:
        path = Path(checkpoint)
        try:
            obj = torch.load(path, map_location="cpu", weights_only=not trust_checkpoint)
        except pickle.UnpicklingError as exc:
            raise ValueError(
                f"{path} does not load with weights_only=True ({exc}); pass --trust-checkpoint "
                "only if you produced this file yourself"
            ) from exc
        saved_for = obj.get("config") if isinstance(obj, Mapping) else None
        if isinstance(saved_for, str) and config_for(saved_for).name != cfg.name:
            raise ValueError(f"{path} was saved for config {saved_for!r}, not {cfg.name!r}")
        try:
            net.load_state_dict(extract_state_dict(obj), strict=True)
        except RuntimeError as exc:
            raise ValueError(f"{path} does not fit config {cfg.name}: {exc}") from exc
        info.update(checkpoint=str(path), checkpoint_sha256=file_sha256(path))
    net.eval()
    return net, info


def system_info(net: StreamingModel, model_info: Mapping[str, Any], mode: str) -> SystemInfo:
    """The results row header: analytic size and compute first, runtime labelled."""
    params = mmacs = None
    if isinstance(net, EarmarkNet):
        report = complexity(net)
        params, mmacs = int(report.params), float(report.mmacs_per_s)
    if model_info.get("random_weights", True):
        training = "none (random init)"
    else:
        training = "Earmark training mix (docs/DATA_AND_LICENSES.md)"
    return SystemInfo(
        name=f"earmark-{model_info.get('config', 'custom')}-{mode}",
        inference_path=INFERENCE_PATH,
        params=params,
        mmac_per_s=mmacs,
        training_data=training,
        notes="PyTorch step() hop by hop, fp32; output latency-compensated by one hop",
    )


def load_manifest(path: str | Path) -> list[MixtureSpec]:
    """Specs of an Earmark-Synth manifest, checked against the current signal contract."""
    path = Path(path)
    if "contract_hash" not in pq.read_schema(path).names:
        raise ValueError(f"{path} has no contract_hash column; is it a synth_bench manifest?")
    column = pq.read_table(path, columns=["contract_hash"]).column("contract_hash")
    hashes = sorted({str(h) for h in column.to_pylist()})
    if hashes != [C.CONTRACT_HASH]:
        raise ValueError(
            f"{path} was designed under contract {hashes}, but this code is "
            f"{C.CONTRACT_HASH}; regenerate the manifest"
        )
    specs = read_manifest(path)
    _check_unique_ids(specs)
    return specs


def select_specs(
    specs: Sequence[MixtureSpec],
    *,
    conditions: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[MixtureSpec]:
    """Keep the given conditions (manifest order), then the first ``limit`` mixtures."""
    chosen = list(specs)
    if conditions:
        unknown = sorted(set(conditions) - set(CONDITIONS))
        if unknown:
            raise ValueError(f"unknown conditions {unknown}; choose from {list(CONDITIONS)}")
        wanted = set(conditions)
        chosen = [s for s in chosen if s.condition in wanted]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        chosen = chosen[:limit]
    if not chosen:
        raise ValueError("no mixtures selected")
    return chosen


def load_pools_file(path: str | Path) -> BenchPools:
    """Open the pools named in a YAML/JSON file (paths relative to the file)."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must map corpus names to dataset folders")
    allowed = _POOL_KEYS + _POOL_SPLIT_KEYS
    unknown = sorted(set(map(str, data)) - set(allowed))
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}; allowed {list(allowed)}")

    def resolve(value: Any) -> list[Path] | None:
        if value is None:
            return None
        entries = [value] if isinstance(value, str) else list(value)
        return [(path.parent / Path(str(v)).expanduser()).resolve() for v in entries]

    kwargs: dict[str, Any] = {k: resolve(data.get(k)) for k in _POOL_KEYS}
    missing = [k for k in ("targets", "interferers", "noise") if not kwargs[k]]
    if missing:
        raise ValueError(f"{path}: needs {missing}")
    for key in _POOL_SPLIT_KEYS:
        if key in data:
            kwargs[key] = None if data[key] is None else [str(v) for v in data[key]]
    return load_bench_pools(**kwargs)


def check_pools_cover(specs: Sequence[MixtureSpec], pools: BenchPools) -> None:
    """Fail fast if any source a spec references is missing from the pools."""
    missing: dict[str, None] = {}

    def need(corpus: str, utt_id: str) -> None:
        try:
            pools.row(corpus, utt_id)
        except KeyError:
            missing[f"{corpus}:{utt_id}"] = None

    for s in specs:
        for ref in (*s.target_segments, *s.interferer, s.music, s.noise):
            if ref is not None:
                need(ref.corpus, ref.utt_id)
        for utt in s.enrol_utts:
            need("targets", utt)
        for rir in (s.target_rir, s.interferer_rir):
            if rir:
                need("rirs", rir)
    if missing:
        shown = list(missing)[:5]
        raise ValueError(
            f"{len(missing)} sources in the manifest are not in the pools (e.g. {shown})"
        )


def default_out_path(
    system_name: str, manifest: str | Path, *, now: datetime | None = None
) -> Path:
    """``results/dev/<system>_<manifest tag>_<UTC time>.json``."""
    manifest = Path(manifest)
    tag = manifest.parent.name if manifest.stem == "manifest" else manifest.stem
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RESULTS_DIR / f"{system_name}_{tag or 'manifest'}_{stamp}.json"


# --------------------------------------------------------------------------------------------
# CLI


def _headline(summary: Mapping[str, Any]) -> dict[str, Any]:
    si = summary.get("metrics", {}).get("si_sdri", {}) or {}
    bargein = summary.get("bargein") or {}
    return {
        "si_sdri": si.get("mean"),
        "si_sdri_ci95": si.get("ci95"),
        "vad_auc": summary["vad"]["auc"],
        "vad_auc_ci95": summary["vad"]["ci95"],
        "false_barge_ins_per_min": bargein.get("false_barge_ins_per_min"),
        "onset_recall": bargein.get("onset_recall"),
        "frame_recall": bargein.get("frame_recall"),
        "median_onset_delay_ms": bargein.get("median_onset_delay_ms"),
        "vad_threshold": (bargein.get("threshold") or {}).get("threshold"),
        "n_items": summary["n_items"],
        "n_skipped": summary["n_skipped"],
        "acceptance_passed": summary["acceptance"]["passed"],
    }


def _progress(every: int = 100) -> Callable[[int, int], None]:
    last = 0

    def report(done: int, total: int) -> None:
        nonlocal last
        if done - last >= every or done == total:
            last = done
            print(f"dev_runner: {done}/{total} mixtures", file=sys.stderr, flush=True)

    return report


def _fmt(x: Any, spec: str = ".3f") -> str:
    v = _finite(x)
    return format(v, spec) if math.isfinite(v) else "n/a"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m earmark.eval.dev_runner", description=(__doc__ or "").split("\n\n")[0]
    )
    model = parser.add_argument_group("model")
    model.add_argument("--config", required=True, help="model config: M, S-GRU, S-SSM or M-256")
    model.add_argument(
        "--checkpoint", type=Path, default=None, help="weights (.pt); omit for random init"
    )
    model.add_argument(
        "--trust-checkpoint", action="store_true", help="allow full unpickling (own files only)"
    )
    model.add_argument(
        "--seed", type=int, default=0, help="random-init seed when there is no checkpoint"
    )
    model.add_argument("--mode", choices=MODES, default="personal")
    model.add_argument("--device", default="cpu", help="torch device (cpu, mps, cuda)")

    data = parser.add_argument_group("data")
    data.add_argument(
        "--manifest", type=Path, required=True, help="Earmark-Synth manifest.parquet"
    )
    data.add_argument(
        "--pools",
        type=Path,
        default=None,
        help=f"pools file (default: {POOLS_FILE_NAME} next to the manifest)",
    )
    data.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=CONDITIONS,
        help="score only these conditions",
    )
    data.add_argument(
        "--limit", type=int, default=None, help="score only the first N selected mixtures"
    )
    data.add_argument(
        "--encoder",
        choices=("wespeaker", "stub"),
        default="wespeaker",
        help="speaker encoder for enrolment (stub: smoke runs only)",
    )
    data.add_argument(
        "--embeddings", type=Path, default=None, help="precomputed per-mixture embeddings (.npz)"
    )
    data.add_argument(
        "--embeddings-out", type=Path, default=None, help="save this run's embeddings (.npz)"
    )

    run = parser.add_argument_group("run")
    run.add_argument(
        "--batch", type=int, default=DEFAULT_BATCH, help="mixtures streamed side by side"
    )
    run.add_argument(
        "--chunk-hops",
        type=int,
        default=DEFAULT_CHUNK_HOPS,
        help="hops of output per scoring update (the most output audio held per stream)",
    )

    scoring = parser.add_argument_group("scoring")
    scoring.add_argument(
        "--vad-threshold", type=float, default=None, help="fixed VAD threshold (frozen on dev)"
    )
    scoring.add_argument(
        "--target-recall",
        type=float,
        default=DEFAULT_TARGET_RECALL,
        help="recall the threshold is matched to when --vad-threshold is not given",
    )
    scoring.add_argument("--threshold-level", choices=("frame", "onset"), default="frame")
    scoring.add_argument(
        "--gate-release",
        type=float,
        default=None,
        help="hysteresis: hold an open run while the score stays at or above this (<= the threshold)",
    )
    scoring.add_argument(
        "--gate-max-gap-frames",
        type=int,
        default=0,
        help="hysteresis: frames below the release threshold an open run survives (10 ms each)",
    )
    scoring.add_argument(
        "--resamples", type=int, default=DEFAULT_RESAMPLES, help="bootstrap resamples"
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--split", choices=("dev", "test", "smoke"), default="dev", help="run-log split label"
    )
    output.add_argument(
        "--out", type=Path, default=None, help="summary JSON (default under results/dev/)"
    )
    output.add_argument(
        "--rows-out", type=Path, default=None, help="per-mixture rows (JSONL, numbers only)"
    )
    output.add_argument(
        "--frames-out",
        type=Path,
        default=None,
        help="per-frame VAD scores and reference activity (.npz), for tuning the gate offline",
    )
    output.add_argument("--runs-log", type=Path, default=DEFAULT_RUNS_PATH)
    output.add_argument("--no-log", action="store_true", help="do not append to the run log")
    output.add_argument("--notes", default="", help="free text stored in the run log")
    output.add_argument(
        "--check", action="store_true", help="exit 1 unless the week-1 acceptance passes"
    )
    return parser


def _label_problems(args: argparse.Namespace) -> list[str]:
    """Refuse runs whose numbers would be mislabelled or meaningless."""
    problems = []
    if args.batch < 1 or args.chunk_hops < 1:
        problems.append("--batch and --chunk-hops must be >= 1")
    if args.resamples < 1:
        problems.append("--resamples must be >= 1")
    if args.vad_threshold is not None and not 0.0 <= args.vad_threshold <= 1.0:
        problems.append("--vad-threshold is a probability in [0, 1]")
    if args.split == "test" and args.vad_threshold is None:
        problems.append("test thresholds are frozen on dev: pass --vad-threshold")
    if args.checkpoint is None and args.split != "smoke":
        problems.append("no --checkpoint means random weights; label the run --split smoke")
    if args.mode == "denoise" and (args.embeddings is not None or args.embeddings_out is not None):
        problems.append("--embeddings and --embeddings-out only apply to personal mode")
    stub = args.mode == "personal" and args.embeddings is None and args.encoder == "stub"
    if stub and args.split != "smoke":
        problems.append("the stub encoder is not WeSpeaker; label the run --split smoke")
    return problems


def _make_embedder(
    args: argparse.Namespace, pools: BenchPools, specs: Sequence[MixtureSpec]
) -> EmbeddingSource | None:
    if args.mode == "denoise":
        return None
    if args.embeddings is not None:
        table = load_enrolment_embeddings(args.embeddings)
        table.check(specs)
        return table
    if args.encoder == "stub":
        return EnrolmentEmbedder(pools, StubSpeakerEncoder())
    from earmark.data.embeddings import WeSpeakerOnnxEncoder

    try:
        encoder = WeSpeakerOnnxEncoder.from_hub()
    except (ImportError, OSError, ValueError) as exc:
        raise ValueError(
            f"the WeSpeaker encoder is unavailable here ({type(exc).__name__}: {exc}); it needs "
            "onnxruntime and torchaudio. Compute embeddings where it runs (--embeddings-out) "
            "and pass them with --embeddings"
        ) from exc
    return EnrolmentEmbedder(pools, encoder)


def _report_line(name: str, summary: Mapping[str, Any], out_path: Path) -> str:
    head = _headline(summary)
    ci = head["si_sdri_ci95"] or [math.nan, math.nan]
    verdict = "PASS" if head["acceptance_passed"] else "FAIL"
    return (
        f"dev_runner: {name} [{INFERENCE_PATH}] {head['n_items']} mixtures "
        f"({head['n_skipped']} skipped) | SI-SDRi {_fmt(head['si_sdri'], '.2f')} dB "
        f"[{_fmt(ci[0], '.2f')}, {_fmt(ci[1], '.2f')}] | VAD AUC {_fmt(head['vad_auc'])} | "
        f"false barge-ins {_fmt(head['false_barge_ins_per_min'], '.2f')}/min at threshold "
        f"{_fmt(head['vad_threshold'])} | acceptance {verdict} | {out_path}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    problems = _label_problems(args)
    if problems:
        parser.error("; ".join(problems))
    try:
        all_specs = load_manifest(args.manifest)
        specs = select_specs(all_specs, conditions=args.conditions, limit=args.limit)
        manifest_splits = sorted({s.split for s in specs})
        if args.split != "smoke" and manifest_splits != [args.split]:
            raise ValueError(
                f"--split {args.split} but the manifest's mixtures are {manifest_splits}"
            )
        pools_path = args.pools or args.manifest.parent / POOLS_FILE_NAME
        if not pools_path.is_file():
            raise ValueError(f"pools file {pools_path} not found; pass --pools")
        pools = load_pools_file(pools_path)
        check_pools_cover(specs, pools)
        net, model_info = load_model(
            args.config, args.checkpoint, seed=args.seed, trust_checkpoint=args.trust_checkpoint
        )
        info = system_info(net, model_info, args.mode)
        model_info["state_bytes_per_stream"] = state_size_bytes(net)
        embed = _make_embedder(args, pools, specs)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    device = torch.device(args.device)
    net = net.to(device)

    run = run_dev(
        net,
        specs,
        pools,
        mode=args.mode,
        embed=embed,
        batch=args.batch,
        chunk_hops=args.chunk_hops,
        device=device,
        progress=_progress(),
    )
    gate = None
    if args.gate_release is not None or args.gate_max_gap_frames:
        # attack is filled in from the chosen threshold inside score_run.
        gate = GateConfig(attack=1.0, release=args.gate_release, max_gap_frames=args.gate_max_gap_frames)
    scored = score_run(
        run,
        vad_threshold=args.vad_threshold,
        target_recall=args.target_recall,
        threshold_level=args.threshold_level,
        n_resamples=args.resamples,
        gate=gate,
    )
    if embed is None:
        embeddings_source = "null (denoise mode)"
    else:
        embeddings_source = str(getattr(embed, "name", type(embed).__name__))
    out_path = args.out or default_out_path(info.name, args.manifest)
    summary: dict[str, Any] = {
        "suite": SUITE,
        "runner": "earmark.eval.dev_runner",
        "inference_path": INFERENCE_PATH,
        "split": args.split,
        "mode": args.mode,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "contract_hash": C.CONTRACT_HASH,
        "system": info.as_dict(),
        "model": model_info,
        "manifest": {
            "path": str(args.manifest),
            "sha256": file_sha256(args.manifest),
            "n_mixtures": len(all_specs),
            "n_selected": len(specs),
            "splits": manifest_splits,
            "conditions": args.conditions,
            "limit": args.limit,
        },
        "pools_file": str(pools_path),
        "embeddings": embeddings_source,
        "stream": {"batch": args.batch, "chunk_hops": args.chunk_hops, "device": str(device)},
        "storage": "numbers only: enhanced audio is scored as it streams and never written",
        **scored,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(summary), indent=2, allow_nan=False)
    out_path.write_text(text + "\n", encoding="utf-8")
    if args.rows_out is not None:
        args.rows_out.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_out.open("w", encoding="utf-8") as fh:
            for it in run.items:
                fh.write(json.dumps(to_jsonable(it.row), allow_nan=False) + "\n")
    if args.frames_out is not None:
        save_frames(run.items, args.frames_out)
    if args.embeddings_out is not None and run.embeddings:
        save_enrolment_embeddings(
            args.embeddings_out, specs, run.embeddings, encoder=embeddings_source
        )
    if not args.no_log:
        record = make_record(
            suite=SUITE,
            system=info.name,
            split=args.split,
            inference_path=INFERENCE_PATH,
            metrics=_headline(summary),
            config={
                "model": model_info,
                "mode": args.mode,
                "manifest": summary["manifest"],
                "embeddings": embeddings_source,
                "threshold": (summary.get("bargein") or {}).get("threshold"),
                "results_json": str(out_path),
            },
            notes=args.notes,
        )
        append_run(record, args.runs_log)
    print(_report_line(info.name, summary, out_path), file=sys.stderr)
    return 1 if args.check and not summary["acceptance"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
