"""Suite B: the VoiceBank+DEMAND test set (824 utterances, speakers p232 and p257), Denoise mode.

Data: ``scripts/fetch_eval_data.sh vbd`` puts the official 48 kHz clean and noisy test sets
in ``.cache/data/vbd``; both are resampled to 16 kHz here with soxr ("HQ", the librosa
default) as they are read. Nothing is written back to disk.

Metrics: PESQ-WB, STOI, ESTOI, SI-SDR, SI-SDRi and CSIG/CBAK/COVL, each as a mean with an
utterance-level bootstrap interval. Only two speakers exist, so every table prints
'n=2 speakers' and no speaker-level claim is made from this suite. Personal mode here is
illustrative only.

Reproduction gates (run before any Suite B number is trusted; ``make gates``):

* unprocessed noisy input: PESQ-WB 1.97 and STOI 0.921, each within 0.02;
* the official GTCRN VoiceBank checkpoint: PESQ-WB 2.87 within 0.05.

CLI::

    python -m earmark.eval.suites.b_vbd --system unprocessed --check-gates
    python -m earmark.eval.suites.b_vbd --system gtcrn-vb --out /tmp/b_gtcrn_vb.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

from earmark import constants as C
from earmark.eval.assets import cache_root
from earmark.eval.runs_log import DEFAULT_RUNS_PATH, append_run, make_record, to_jsonable
from earmark.eval.stream_score import EvalItem, ItemAudio, score_items, summarize_rows
from earmark.eval.systems import Enhancer, Unprocessed

__all__ = [
    "GATES",
    "N_UTTERANCES",
    "PUBLISHED",
    "SUITE_METRICS",
    "SUMMARY_KEYS",
    "DataMissingError",
    "VbdUtterance",
    "check_gates",
    "default_root",
    "list_utterances",
    "load_wav_16k",
    "main",
    "make_items",
    "make_system",
    "run_suite",
]

N_UTTERANCES = 824
SPEAKERS = ("p232", "p257")
RESAMPLER_QUALITY = "HQ"
SUITE_METRICS: tuple[str, ...] = ("pesq_wb", "stoi", "estoi", "si_sdr", "si_sdri", "composite")
SUMMARY_KEYS: tuple[str, ...] = ("pesq_wb", "stoi", "estoi", "si_sdr", "si_sdri", "csig", "cbak", "covl", "segsnr")

#: Published VB-test numbers (SEGAN/MetricGAN-style tables; GTCRN README table 1).
PUBLISHED: dict[str, dict[str, float]] = {
    "unprocessed": {"pesq_wb": 1.97, "stoi": 0.921, "csig": 3.35, "cbak": 2.44, "covl": 2.63, "si_sdr": 8.45},
    "gtcrn-vb": {"pesq_wb": 2.87, "stoi": 0.940, "si_sdr": 18.83},
}
#: Gate targets: system -> metric -> (published value, tolerance).
GATES: dict[str, dict[str, tuple[float, float]]] = {
    "unprocessed": {"pesq_wb": (1.97, 0.02), "stoi": (0.921, 0.02)},
    "gtcrn-vb": {"pesq_wb": (2.87, 0.05)},
}

_FETCH_HINT = "run scripts/fetch_eval_data.sh vbd first"


class DataMissingError(FileNotFoundError):
    """The VoiceBank+DEMAND test set is not where Suite B expects it."""


@dataclass(frozen=True)
class VbdUtterance:
    """One clean/noisy test pair and its noise condition from ``log_testset.txt``."""

    item_id: str
    speaker: str
    clean_path: Path
    noisy_path: Path
    noise: str | None
    snr_db: float | None


def default_root() -> Path:
    return cache_root() / "data" / "vbd"


def read_noise_log(root: Path) -> dict[str, tuple[str, float]]:
    """``item_id -> (noise type, SNR dB)`` from ``logfiles/log_testset.txt`` (empty if absent)."""
    log = root / "logfiles" / "log_testset.txt"
    if not log.is_file():
        return {}
    table: dict[str, tuple[str, float]] = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3:
            table[parts[0]] = (parts[1], float(parts[2]))
    return table


def list_utterances(root: Path | None = None, *, expect_full: bool = True) -> list[VbdUtterance]:
    """All clean/noisy pairs, sorted by id. ``expect_full`` insists on the 824 official pairs."""
    base = default_root() if root is None else root
    clean_dir, noisy_dir = base / "clean_testset_wav", base / "noisy_testset_wav"
    if not clean_dir.is_dir() or not noisy_dir.is_dir():
        raise DataMissingError(f"VoiceBank+DEMAND test set not found under {base}; {_FETCH_HINT}")
    clean = {p.stem: p for p in clean_dir.glob("*.wav")}
    noisy = {p.stem: p for p in noisy_dir.glob("*.wav")}
    if set(clean) != set(noisy):
        missing = sorted(set(clean) ^ set(noisy))[:5]
        raise DataMissingError(f"clean and noisy sets do not pair up (e.g. {missing}); {_FETCH_HINT}")
    if expect_full and len(clean) != N_UTTERANCES:
        raise DataMissingError(f"expected {N_UTTERANCES} utterances under {base}, found {len(clean)}; {_FETCH_HINT}")
    log = read_noise_log(base)
    out = []
    for item_id in sorted(clean):
        noise, snr = log.get(item_id, (None, None))
        out.append(VbdUtterance(item_id, item_id.split("_")[0], clean[item_id], noisy[item_id], noise, snr))
    return out


def load_wav_16k(path: Path) -> np.ndarray:
    """Read a mono wav as float64 and resample it to 16 kHz with soxr (HQ) if needed."""
    audio, sr = sf.read(path, dtype="float64", always_2d=False)
    if audio.ndim != 1:
        raise ValueError(f"{path} is not mono (shape {audio.shape})")
    if sr != C.SAMPLE_RATE:
        audio = soxr.resample(audio, sr, C.SAMPLE_RATE, quality=RESAMPLER_QUALITY)
    return np.ascontiguousarray(audio, dtype=np.float64)


def _load_item(utt: VbdUtterance) -> ItemAudio:
    clean = load_wav_16k(utt.clean_path)
    noisy = load_wav_16k(utt.noisy_path)
    if clean.size != noisy.size:
        raise ValueError(f"{utt.item_id}: clean and noisy lengths differ ({clean.size} vs {noisy.size})")
    return ItemAudio(mixture=noisy.astype(np.float32), reference=clean)


def make_items(utterances: Sequence[VbdUtterance]) -> list[EvalItem]:
    """Lazy evaluation items (audio is read only when an item is scored)."""
    return [
        EvalItem(
            item_id=u.item_id,
            cluster=u.speaker,
            load=partial(_load_item, u),
            meta={"noise": u.noise, "snr_db": u.snr_db},
        )
        for u in utterances
    ]


def make_system(name: str) -> Enhancer:
    """Systems Suite B can build by name: ``unprocessed``, ``gtcrn-vb``, ``gtcrn-dns3``."""
    if name in ("unprocessed", "noisy"):
        return Unprocessed()
    if name in ("gtcrn-vb", "gtcrn-dns3"):
        from earmark.eval.baselines.gtcrn_torch import GtcrnTorch  # imports torch lazily

        return GtcrnTorch("vb" if name == "gtcrn-vb" else "dns3")
    raise ValueError(f"unknown Suite B system {name!r}; choose unprocessed, gtcrn-vb or gtcrn-dns3")


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _by_snr(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("snr_db") is not None:
            groups[f"{float(r['snr_db']):g}"].append(r)
    table = {}
    for snr in sorted(groups, key=float):
        entry: dict[str, float] = {"n": float(len(groups[snr]))}
        for k in keys:
            vals = [float(r[k]) for r in groups[snr] if isinstance(r.get(k), float) and math.isfinite(r[k])]
            entry[k] = float(np.mean(vals)) if vals else math.nan
        table[snr] = entry
    return table


def check_gates(system_name: str, metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare measured means with the reproduction gates for ``system_name`` (may be empty)."""
    results = []
    for metric, (target, tol) in GATES.get(system_name, {}).items():
        measured = float(metrics.get(metric, {}).get("mean", math.nan))
        results.append(
            {
                "metric": metric,
                "measured": measured,
                "target": target,
                "tolerance": tol,
                "passed": bool(math.isfinite(measured) and abs(measured - target) <= tol),
            }
        )
    return results


def run_suite(
    system: Enhancer,
    *,
    root: Path | None = None,
    workers: int | None = None,
    limit: int | None = None,
    metrics: Sequence[str] = SUITE_METRICS,
    n_resamples: int = 2000,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score ``system`` on the VB test set; return ``(summary, rows)`` (numbers only)."""
    utterances = list_utterances(root, expect_full=limit is None and root is None)
    if limit is not None:
        utterances = utterances[:limit]
    started = time.perf_counter()
    rows = score_items(
        make_items(utterances),
        system,
        metrics=metrics,
        workers=default_workers() if workers is None else workers,
        on_row=on_row,
    )
    keys = [k for k in SUMMARY_KEYS if any(k in r for r in rows)]
    summary_metrics = summarize_rows(rows, keys, n_resamples=n_resamples)
    name = system.info.name
    summary: dict[str, Any] = {
        "suite": "B",
        "dataset": "VoiceBank+DEMAND test set (Valentini-Botinhao 2017), CC BY 4.0",
        "system": system.info.as_dict(),
        "mode": "denoise",
        "n_items": len(rows),
        "n_speakers": len({r["cluster"] for r in rows}),
        "speakers_note": "n=2 speakers (p232, p257); intervals are utterance-level bootstraps",
        "sample_rate": C.SAMPLE_RATE,
        "resampler": f"soxr {RESAMPLER_QUALITY} 48 kHz -> 16 kHz",
        "contract_hash": C.CONTRACT_HASH,
        "metrics": summary_metrics,
        "by_snr_db": _by_snr(rows, ("pesq_wb", "stoi", "si_sdr")),
        "published": PUBLISHED.get(name, {}),
        "gates": check_gates(name, summary_metrics) if len(rows) == N_UTTERANCES else [],
        "n_rows_with_errors": sum(1 for r in rows if "errors" in r),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    return summary, rows


def _progress(total: int) -> Callable[[dict[str, Any]], None]:
    seen = 0

    def report(_row: dict[str, Any]) -> None:
        nonlocal seen
        seen += 1
        if seen % 100 == 0 or seen == total:
            print(f"suite B: scored {seen}/{total}", file=sys.stderr, flush=True)

    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--system", default="unprocessed", help="unprocessed | gtcrn-vb | gtcrn-dns3")
    parser.add_argument("--root", type=Path, default=None, help="VB data root (default .cache/data/vbd)")
    parser.add_argument("--workers", type=int, default=None, help="metric worker processes (0 = in-process)")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N utterances")
    parser.add_argument("--metrics", nargs="+", default=list(SUITE_METRICS), choices=SUITE_METRICS)
    parser.add_argument("--resamples", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    parser.add_argument("--rows-out", type=Path, default=None, help="write per-utterance rows (JSONL) here")
    parser.add_argument("--check-gates", action="store_true", help="exit 1 if a reproduction gate fails")
    parser.add_argument("--log-run", choices=("gate", "test", "dev", "smoke"), default=None,
                        help="append the summary to the run log with this split")
    parser.add_argument("--runs-log", type=Path, default=DEFAULT_RUNS_PATH)
    args = parser.parse_args(argv)

    system = make_system(args.system)
    total = args.limit or N_UTTERANCES
    summary, rows = run_suite(
        system,
        root=args.root,
        workers=args.workers,
        limit=args.limit,
        metrics=args.metrics,
        n_resamples=args.resamples,
        on_row=_progress(total),
    )
    text = json.dumps(to_jsonable(summary), indent=2, allow_nan=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.rows_out is not None:
        args.rows_out.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(to_jsonable(r), allow_nan=False) + "\n")
    if args.log_run is not None:
        record = make_record(
            suite="B",
            system=system.info.name,
            split=args.log_run,
            inference_path=system.info.inference_path,
            metrics={k: v["mean"] for k, v in summary["metrics"].items()},
            config={"n_items": summary["n_items"], "gates": summary["gates"], "resampler": summary["resampler"]},
        )
        append_run(record, args.runs_log)
    for g in summary["gates"]:
        status = "PASS" if g["passed"] else "FAIL"
        print(
            f"gate {status}: {system.info.name} {g['metric']} = {g['measured']:.4f} "
            f"(target {g['target']} +/- {g['tolerance']})",
            file=sys.stderr,
        )
    if args.check_gates and not all(g["passed"] for g in summary["gates"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
