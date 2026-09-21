"""Choose the barge-in gate from a dev run's per-frame scores, without running the model.

The model's VAD probability dips inside real speech, and the contract's onset rule needs
200 ms of *consecutive* active frames, so a plain threshold loses onsets. The gate
(:class:`earmark.eval.bargein.GateConfig`) adds hysteresis and bridges short dips. Its
settings are chosen here on dev, under a false-barge-in budget, and then frozen::

    python -m earmark.eval.dev_runner ... --frames-out results/dev/frames.npz
    python -m earmark.eval.tune_gate results/dev/frames.npz --max-false-per-min 4.5

The chosen setting is printed as the flags to pass back to the dev runner, and can be
written to JSON with ``--out``. Tune on dev only; applying it to test is a separate run.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from earmark.eval.bargein import GateChoice, tune_gate

__all__ = ["load_frames", "main", "report"]


def load_frames(path: str | Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The per-item scores and reference activity written by ``--frames-out``."""
    with np.load(path, allow_pickle=False) as data:
        scores, labels, offsets = data["scores"], data["labels"], data["offsets"]
    items = [(scores[a:b], labels[a:b]) for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    if not items:
        raise ValueError(f"{path} holds no items")
    return [s for s, _ in items], [lab for _, lab in items]


def report(best: GateChoice, results: Sequence[GateChoice], top: int = 8) -> str:
    """The chosen setting and the strongest alternatives, as a table."""
    lines = [
        f"{'attack':>7s} {'release':>8s} {'gap':>4s} | {'onsets':>7s} {'false/min':>10s} "
        f"{'frames':>7s} {'delay ms':>9s}",
    ]
    ranked = sorted(results, key=lambda r: (-r.onset_recall, r.false_per_minute))[:top]
    for choice in ranked:
        mark = " <-" if choice is best else ""
        lines.append(
            f"{choice.config.attack:7.3f} {choice.config.release_threshold:8.3f} "
            f"{choice.config.max_gap_frames:4d} | {choice.onset_recall:7.3f} "
            f"{choice.false_per_minute:10.2f} {choice.frame_recall:7.3f} {choice.median_delay_ms:9.0f}{mark}"
        )
    flags = (
        f"--vad-threshold {best.config.attack:.6f} --gate-release {best.config.release_threshold:.6f} "
        f"--gate-max-gap-frames {best.config.max_gap_frames}"
    )
    lines.append("")
    lines.append(f"chosen: {flags}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m earmark.eval.tune_gate", description=__doc__.split("\n")[0])
    parser.add_argument("frames", type=Path, help="the .npz written by dev_runner --frames-out")
    parser.add_argument(
        "--max-false-per-min",
        type=float,
        required=True,
        help="budget: pooled false barge-ins per minute of target silence",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        nargs="+",
        default=[0, 2, 4, 8],
        help="gap lengths to try, in 10 ms frames",
    )
    parser.add_argument("--candidates", type=int, default=16, help="attack thresholds to try")
    parser.add_argument("--out", type=Path, default=None, help="write the chosen setting as JSON")
    args = parser.parse_args(argv)

    try:
        scores, labels = load_frames(args.frames)
    except (OSError, ValueError, KeyError) as exc:
        print(f"cannot read {args.frames}: {exc}")
        return 1
    best, results = tune_gate(
        scores,
        labels,
        max_false_per_minute=args.max_false_per_min,
        max_gap_frames=args.max_gap_frames,
        n_candidates=args.candidates,
    )
    print(report(best, results))
    if best.false_per_minute > args.max_false_per_min:
        print(f"\nnothing met the budget of {args.max_false_per_min}/min; showing the quietest setting")
    if args.out is not None:
        payload: dict[str, Any] = {
            "frames": str(args.frames),
            "max_false_per_min": args.max_false_per_min,
            "chosen": best.as_dict(),
            "searched": [choice.as_dict() for choice in results],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
