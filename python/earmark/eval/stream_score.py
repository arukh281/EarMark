"""Score-as-you-go evaluation: run a system over items and keep only numbers, never audio.

The plan's storage rule is that enhanced audio is never written anywhere; only JSON, transcripts
and at most :data:`GALLERY_MAX_CLIPS` hand-picked gallery clips are kept. This module is how
suites honour it:

* :func:`score_items` loads one item at a time, runs the system in the main process, hands the
  reference/output pair to a worker pool for the whole-utterance metrics (PESQ, STOI, ...),
  and drops the audio as soon as the job is queued. At most ``max_in_flight`` items are alive
  at once, so memory stays bounded however large the suite is. Rows come back in item order.
* :class:`StreamingScorer` scores one long recording chunk by chunk (LibriCSS sessions, live
  engine output) keeping only running sums and a few numbers per 10 ms frame: SI-SDR/SI-SDRi,
  TSOS and interferer suppression match the offline functions in :mod:`earmark.eval.metrics`.
* :func:`summarize_rows` turns rows into means with bootstrap intervals (utterance level, or
  speaker-cluster level when rows carry a ``cluster``).
"""

from __future__ import annotations

import math
import multiprocessing
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from earmark import constants as C
from earmark.data.activity import frame_energy_from_frames
from earmark.eval import metrics as M
from earmark.eval.bargein import BargeInCounts, binarize, pool_counts, score_bargein
from earmark.eval.bootstrap import DEFAULT_RESAMPLES, bootstrap_mean
from earmark.eval.systems import Enhancer, as_output

__all__ = [
    "DEFAULT_METRICS",
    "GALLERY_MAX_CLIPS",
    "UTTERANCE_METRICS",
    "EvalItem",
    "ItemAudio",
    "StreamingScorer",
    "pooled_bargein",
    "score_items",
    "summarize_rows",
    "utterance_metrics",
]

#: Hard cap on audio clips a suite may keep (for the web gallery); everything else is numbers.
GALLERY_MAX_CLIPS: int = 10

#: Metric groups :func:`utterance_metrics` understands.
UTTERANCE_METRICS: tuple[str, ...] = (
    "pesq_wb",
    "stoi",
    "estoi",
    "si_sdr",
    "si_sdri",
    "composite",
    "tsos",
    "interferer_suppression",
)
DEFAULT_METRICS: tuple[str, ...] = ("pesq_wb", "stoi", "estoi", "si_sdr", "si_sdri")


@dataclass(frozen=True)
class ItemAudio:
    """Audio and labels for one item, loaded lazily by :attr:`EvalItem.load`.

    ``reference`` is the clean target (``None`` for target-absent items), ``interferer`` the
    interfering source as it appears in the mixture, ``reference_vad`` the target's frame labels.
    """

    mixture: NDArray[np.floating]
    reference: NDArray[np.floating] | None = None
    interferer: NDArray[np.floating] | None = None
    embedding: NDArray[np.float32] | None = None
    reference_vad: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class EvalItem:
    """One scoring unit. ``cluster`` (the speaker) drives the speaker-cluster bootstrap."""

    item_id: str
    cluster: str
    load: Callable[[], ItemAudio]
    meta: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Whole-utterance metrics (run in worker processes)


def utterance_metrics(
    reference: ArrayLike | None,
    estimate: ArrayLike,
    mixture: ArrayLike | None = None,
    interferer: ArrayLike | None = None,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    sample_rate: int = C.SAMPLE_RATE,
) -> dict[str, Any]:
    """Compute the requested metric groups for one item; failures become NaN plus a message.

    Output keys: ``pesq_wb``, ``stoi``, ``estoi``, ``si_sdr``, ``si_sdri``, ``csig``, ``cbak``,
    ``covl``, ``segsnr``, ``tsos_percent``, ``tsos_active_frames``, ``tsos_os_frames``,
    ``tsos_max_os_s``, ``interferer_suppression_db``, and ``errors`` (metric -> message) when any
    metric was undefined. Reference-based metrics are skipped when ``reference`` is ``None``.
    """
    unknown = set(metrics) - set(UTTERANCE_METRICS)
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}; choose from {UTTERANCE_METRICS}")
    est = np.asarray(estimate, dtype=np.float64)
    ref = None if reference is None else np.asarray(reference, dtype=np.float64)
    mix = None if mixture is None else np.asarray(mixture, dtype=np.float64)
    out: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def attempt(name: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except (M.MetricError, ValueError) as exc:
            errors[name] = str(exc)
            return None

    needs_ref = {"pesq_wb", "stoi", "estoi", "si_sdr", "si_sdri", "composite", "tsos"}
    for name in metrics:
        if name in needs_ref and ref is None:
            continue
        match name:
            case "pesq_wb":
                v = attempt(name, lambda: M.pesq_wb(ref, est, sample_rate))
                out["pesq_wb"] = math.nan if v is None else v
            case "stoi":
                v = attempt(name, lambda: M.stoi(ref, est, sample_rate))
                out["stoi"] = math.nan if v is None else v
            case "estoi":
                v = attempt(name, lambda: M.estoi(ref, est, sample_rate))
                out["estoi"] = math.nan if v is None else v
            case "si_sdr":
                v = attempt(name, lambda: M.si_sdr(ref, est))
                out["si_sdr"] = math.nan if v is None else v
            case "si_sdri":
                if mix is None:
                    errors[name] = "no mixture given"
                    out["si_sdri"] = math.nan
                else:
                    v = attempt(name, lambda: M.si_sdr_improvement(ref, est, mix))
                    out["si_sdri"] = math.nan if v is None else v
            case "composite":
                pesq_known = out.get("pesq_wb")
                reuse = pesq_known if pesq_known is not None and math.isfinite(pesq_known) else None
                v = attempt(name, lambda: M.composite(ref, est, sample_rate, pesq_score=reuse))
                for key in ("csig", "cbak", "covl", "segsnr"):
                    out[key] = math.nan if v is None else getattr(v, key)
            case "tsos":
                v = attempt(name, lambda: M.tsos(ref, est))
                out["tsos_percent"] = math.nan if v is None else v.percent
                out["tsos_active_frames"] = 0 if v is None else v.active_frames
                out["tsos_os_frames"] = 0 if v is None else v.os_frames
                out["tsos_max_os_s"] = math.nan if v is None else v.max_os_s
            case "interferer_suppression":
                if mix is None or interferer is None:
                    errors[name] = "needs mixture and interferer"
                    out["interferer_suppression_db"] = math.nan
                else:
                    region = M.interferer_region(ref, interferer)
                    v = attempt(name, lambda: M.interferer_suppression_db(mix, est, region))
                    out["interferer_suppression_db"] = math.nan if v is None else v
    if errors:
        out["errors"] = errors
    return out


def _score_job(job: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point (module level so it pickles under the spawn start method)."""
    return utterance_metrics(
        job["reference"],
        job["estimate"],
        job["mixture"],
        job["interferer"],
        metrics=job["metrics"],
        sample_rate=job["sample_rate"],
    )


# --------------------------------------------------------------------------------------------
# Running a system over items


def _bargein_fields(counts: BargeInCounts) -> dict[str, Any]:
    fields = {f"bargein_{k}": v for k, v in counts.summary().items()}
    fields["bargein_delays_frames"] = list(counts.delays_frames)
    fields["bargein_silent_frames"] = counts.silent_frames
    return fields


def score_items(
    items: Iterable[EvalItem],
    system: Enhancer,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    workers: int = 0,
    max_in_flight: int | None = None,
    vad_threshold: float | None = None,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run ``system`` over ``items`` and return one row of numbers per item, in item order.

    ``workers=0`` scores in this process (tests, tiny runs); ``workers>0`` uses a spawn-based
    process pool while the system keeps running here. When the system returns frame VAD
    probabilities, the item has ``reference_vad`` and ``vad_threshold`` is given (a threshold
    frozen on dev), barge-in tallies are added under ``bargein_*`` keys. ``on_row`` is called
    as each row completes (in order) - use it for progress or to stream rows to JSONL.
    """
    if workers < 0:
        raise ValueError("workers must be >= 0")
    limit = max_in_flight if max_in_flight is not None else max(2 * workers, 1)
    rows: dict[int, dict[str, Any]] = {}
    pending: dict[Future[dict[str, Any]], int] = {}
    emitted = 0

    def flush() -> None:
        nonlocal emitted
        while emitted in rows:
            if on_row is not None:
                on_row(rows[emitted])
            emitted += 1

    def harvest(done: Iterable[Future[dict[str, Any]]]) -> None:
        for fut in done:
            idx = pending.pop(fut)
            rows[idx].update(fut.result())
        flush()

    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 0
        else None
    )
    try:
        for idx, item in enumerate(items):
            audio = item.load()
            mixture = np.asarray(audio.mixture, dtype=np.float32)
            output = as_output(system.enhance(mixture, audio.embedding), mixture.size)
            row: dict[str, Any] = {
                "item_id": item.item_id,
                "cluster": item.cluster,
                "duration_s": mixture.size / system.sample_rate,
                **dict(item.meta),
            }
            if output.vad is not None and audio.reference_vad is not None and vad_threshold is not None:
                ref_vad = np.asarray(audio.reference_vad, dtype=bool)
                n = min(ref_vad.size, output.vad.size)
                row.update(_bargein_fields(score_bargein(binarize(output.vad[:n], vad_threshold), ref_vad[:n])))
            job = {
                "reference": audio.reference,
                "estimate": output.audio,
                "mixture": mixture,
                "interferer": audio.interferer,
                "metrics": tuple(metrics),
                "sample_rate": system.sample_rate,
            }
            rows[idx] = row
            if pool is None:
                row.update(_score_job(job))
                flush()
            else:
                pending[pool.submit(_score_job, job)] = idx
                if len(pending) >= limit:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    harvest(done)
            del audio, output, job, mixture  # nothing audio-sized outlives its item
        if pending:
            done, _ = wait(pending)
            harvest(done)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return [rows[i] for i in range(len(rows))]


# --------------------------------------------------------------------------------------------
# Summaries


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    *,
    cluster_level: bool = False,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Mean and bootstrap interval of each metric over the rows where it is finite.

    Each entry reports ``mean``, ``ci95`` (percentile bootstrap, utterance level unless
    ``cluster_level``), ``n`` (items used) and ``n_missing`` (NaN or absent). Barge-in tallies
    are pooled separately with :func:`pooled_bargein`.
    """
    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        vals, clusters = [], []
        for r in rows:
            v = r.get(key)
            if isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v)):
                vals.append(float(v))
                clusters.append(str(r.get("cluster", "")))
        entry: dict[str, Any] = {"n": len(vals), "n_missing": len(rows) - len(vals)}
        if vals:
            boot = bootstrap_mean(vals, clusters=clusters if cluster_level else None, n_resamples=n_resamples, seed=seed)
            entry.update(mean=boot.estimate, ci95=[boot.low, boot.high], unit=boot.unit, n_units=boot.n_units)
        else:
            entry.update(mean=math.nan, ci95=[math.nan, math.nan])
        out[key] = entry
    return out


def pooled_bargein(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Pool per-item barge-in tallies (ratio of sums) from rows produced by :func:`score_items`."""
    counts = []
    for r in rows:
        if "bargein_reference_onsets" not in r:
            continue
        counts.append(
            BargeInCounts(
                reference_onsets=int(r["bargein_reference_onsets"]),
                detected_onsets=int(r["bargein_detected_onsets"]),
                hits=int(r["bargein_hits"]),
                false_barge_ins=int(r["bargein_false_barge_ins"]),
                silent_frames=int(r["bargein_silent_frames"]),
                active_frames=int(r["bargein_active_frames"]),
                detected_active_frames=int(r["bargein_detected_active_frames"]),
                delays_frames=list(r["bargein_delays_frames"]),
            )
        )
    return pool_counts(counts).summary() if counts else None


# --------------------------------------------------------------------------------------------
# Streaming scorer for long recordings


class StreamingScorer:
    """Chunk-by-chunk scoring of one long, time-aligned stream without keeping its audio.

    Keeps running sums for SI-SDR and, per 10 ms contract frame, the reference/output/mixture/
    interferer frame energies and the TSOS over-suppression flag (a few numbers per frame).
    :meth:`result` matches :func:`metrics.si_sdr`, :func:`metrics.tsos` and
    :func:`metrics.interferer_suppression_db` on the concatenated signals. Activity masks are
    relative to the whole stream's peak, so they are resolved at :meth:`result` time.
    """

    def __init__(self, *, tsos_p: float = M.TSOS_COMPRESSION, tsos_gamma: float = M.TSOS_GAMMA) -> None:
        self._p = tsos_p
        self._gamma = tsos_gamma
        self._n = 0
        self._sums = dict.fromkeys(("r", "e", "rr", "ee", "re", "m", "mm", "rm"), 0.0)
        self._has_mix: bool | None = None
        self._has_int: bool | None = None
        self._carry: dict[str, NDArray[np.float64]] = {}
        self._frames: dict[str, list[NDArray[np.float64]]] = {k: [] for k in ("er", "ee", "em", "ei", "os")}

    def update(
        self,
        reference: ArrayLike,
        estimate: ArrayLike,
        mixture: ArrayLike | None = None,
        interferer: ArrayLike | None = None,
    ) -> None:
        """Add the next time-aligned chunk (all given signals must have the same length)."""
        chunks = {"r": np.asarray(reference, dtype=np.float64), "e": np.asarray(estimate, dtype=np.float64)}
        if mixture is not None:
            chunks["m"] = np.asarray(mixture, dtype=np.float64)
        if interferer is not None:
            chunks["i"] = np.asarray(interferer, dtype=np.float64)
        n = chunks["r"].size
        if any(c.ndim != 1 or c.size != n for c in chunks.values()):
            raise ValueError("all chunks must be 1-D with the same length")
        has_mix, has_int = "m" in chunks, "i" in chunks
        if self._has_mix is None:
            self._has_mix, self._has_int = has_mix, has_int
            self._carry = {k: np.zeros(0) for k in chunks}
        elif (has_mix, has_int) != (self._has_mix, self._has_int):
            raise ValueError("pass the same set of signals (mixture, interferer) with every chunk")

        r, e = chunks["r"], chunks["e"]
        s = self._sums
        s["r"] += float(r.sum())
        s["e"] += float(e.sum())
        s["rr"] += float(np.dot(r, r))
        s["ee"] += float(np.dot(e, e))
        s["re"] += float(np.dot(r, e))
        if has_mix:
            m = chunks["m"]
            s["m"] += float(m.sum())
            s["mm"] += float(np.dot(m, m))
            s["rm"] += float(np.dot(r, m))
        self._n += n

        bufs = {k: np.concatenate((self._carry[k], c)) for k, c in chunks.items()}
        t = M.num_frames(bufs["r"].size)
        if t:
            frames = {k: M.frame_signal(b)[:t] for k, b in bufs.items()}
            # The contract's windowed frame energy, the same code path as metrics.frame_energy.
            energy = {k: frame_energy_from_frames(f) for k, f in frames.items()}
            self._frames["er"].append(energy["r"])
            self._frames["ee"].append(energy["e"])
            if has_mix:
                self._frames["em"].append(energy["m"])
            if has_int:
                self._frames["ei"].append(energy["i"])
            self._frames["os"].append(
                M.os_flags_from_frames(frames["r"], frames["e"], p=self._p, gamma=self._gamma).astype(np.float64)
            )
        self._carry = {k: b[t * C.HOP_LENGTH :].copy() for k, b in bufs.items()}

    @staticmethod
    def _si_sdr(n: int, sr: float, se: float, srr: float, see: float, sre: float) -> float:
        mr, me = sr / n, se / n
        rr = srr - n * mr * mr
        ee = see - n * me * me
        re = sre - n * mr * me
        if rr <= 1e-12:
            raise M.MetricError("reference is silent; SI-SDR is undefined")
        target = re * re / rr
        noise = max(ee - target, 0.0)
        return 10.0 * math.log10((target + 1e-12) / (noise + 1e-12))

    def _cat(self, key: str) -> NDArray[np.float64]:
        parts = self._frames[key]
        return np.concatenate(parts) if parts else np.zeros(0)

    def result(self) -> dict[str, Any]:
        """Scores over everything seen so far (zero-mean SI-SDR, TSOS, interferer suppression)."""
        if self._n == 0:
            raise ValueError("no audio has been scored")
        s, n = self._sums, self._n
        out: dict[str, Any] = {"duration_s": n / C.SAMPLE_RATE, "frames": int(self._cat("er").size)}
        try:
            out["si_sdr"] = self._si_sdr(n, s["r"], s["e"], s["rr"], s["ee"], s["re"])
            if self._has_mix:
                out["si_sdri"] = out["si_sdr"] - self._si_sdr(n, s["r"], s["m"], s["rr"], s["mm"], s["rm"])
        except M.MetricError:
            out["si_sdr"] = math.nan
            if self._has_mix:
                out["si_sdri"] = math.nan
        er = self._cat("er")
        active = M.activity_from_energy(er)
        ts = M.tsos_from_flags(self._cat("os") > 0.5, active)
        out.update(
            tsos_percent=ts.percent,
            tsos_active_frames=ts.active_frames,
            tsos_os_frames=ts.os_frames,
            tsos_max_os_s=ts.max_os_s,
        )
        if self._has_mix and self._has_int:
            region = M.activity_from_energy(self._cat("ei")) & ~active
            em = float(self._cat("em")[region].sum())
            ee = float(self._cat("ee")[region].sum())
            out["interferer_suppression_db"] = (
                math.nan if em <= 0.0 else 10.0 * math.log10(em / max(ee, em * 1e-12))
            )
        return out
