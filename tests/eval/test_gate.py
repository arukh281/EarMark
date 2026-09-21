"""The barge-in gate: hysteresis, gap bridging, tuning, and the dev runner's frame dump."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from earmark import constants as C
from earmark.eval import dev_runner, tune_gate as T
from earmark.eval.bargein import (
    FrameGate,
    GateConfig,
    detect_onsets,
    gate_frames,
    score_bargein,
    score_gate,
    tune_gate,
)

MIN_ACTIVE = C.BARGEIN_MIN_ACTIVE_FRAMES
MIN_SILENCE = C.BARGEIN_MIN_SILENCE_FRAMES


def dipped_onset(dip_at: int = 10, dip_length: int = 1, high: float = 0.9, low: float = 0.1) -> np.ndarray:
    """Silence, then speech whose score dips for a few frames inside the 200 ms window."""
    scores = np.concatenate(
        [np.full(MIN_SILENCE, low), np.full(2 * MIN_ACTIVE, high)]
    )
    scores[MIN_SILENCE + dip_at : MIN_SILENCE + dip_at + dip_length] = low
    return scores


def test_a_plain_gate_is_exactly_a_threshold() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    assert np.array_equal(gate_frames(scores, GateConfig(attack=0.5)), scores >= 0.5)
    assert GateConfig(attack=0.5).is_plain
    assert not GateConfig(attack=0.5, max_gap_frames=1).is_plain


def test_streaming_and_vectorised_agree() -> None:
    rng = np.random.default_rng(1)
    scores = rng.random(2000)
    for config in (
        GateConfig(0.6),
        GateConfig(0.6, release=0.3),
        GateConfig(0.6, max_gap_frames=4),
        GateConfig(0.6, release=0.2, max_gap_frames=3),
    ):
        gate = FrameGate(config)
        streamed = np.array([gate.push(float(v)) for v in scores])
        assert np.array_equal(streamed, gate_frames(scores, config)), config


def test_hysteresis_holds_a_run_through_a_shallow_dip() -> None:
    scores = dipped_onset(dip_length=2, low=0.35)
    plain = gate_frames(scores, GateConfig(0.6))
    held = gate_frames(scores, GateConfig(0.6, release=0.3))
    assert not detect_onsets(plain)  # the dip restarts the 200 ms count
    assert len(detect_onsets(held)) == 1


def test_gap_bridging_recovers_an_onset_lost_to_a_deep_dip() -> None:
    scores = dipped_onset(dip_length=3, low=0.05)
    assert not detect_onsets(gate_frames(scores, GateConfig(0.6, release=0.3)))
    bridged = gate_frames(scores, GateConfig(0.6, release=0.3, max_gap_frames=3))
    onsets = detect_onsets(bridged)
    assert len(onsets) == 1
    assert onsets[0].start == MIN_SILENCE  # and it starts where the speech did


def test_a_gap_longer_than_allowed_still_closes_the_run() -> None:
    scores = dipped_onset(dip_length=5, low=0.05)
    assert not detect_onsets(gate_frames(scores, GateConfig(0.6, release=0.3, max_gap_frames=3)))


def test_bridging_does_not_invent_onsets_in_silence() -> None:
    quiet = np.full(10 * MIN_ACTIVE, 0.05)
    assert not detect_onsets(gate_frames(quiet, GateConfig(0.6, release=0.3, max_gap_frames=8)))


def test_release_above_attack_is_refused() -> None:
    with pytest.raises(ValueError, match="release"):
        GateConfig(attack=0.4, release=0.8)
    with pytest.raises(ValueError, match="max_gap_frames"):
        GateConfig(attack=0.4, max_gap_frames=-1)


def _dev_like_frames(n_items: int = 6) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Items whose target speaks in bursts, scored by a jittery model that dips mid-speech."""
    rng = np.random.default_rng(7)
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for _ in range(n_items):
        active = np.zeros(6 * (MIN_SILENCE + 2 * MIN_ACTIVE), dtype=bool)
        at = MIN_SILENCE
        while at + 2 * MIN_ACTIVE < active.size:
            active[at : at + 2 * MIN_ACTIVE] = True
            at += 2 * MIN_ACTIVE + MIN_SILENCE
        score = np.where(active, 0.85, 0.05) + 0.03 * rng.standard_normal(active.size)
        # one deep dip early in each burst, which is what costs onsets
        for start in np.flatnonzero(active & ~np.concatenate(([False], active[:-1]))):
            score[start + 8 : start + 10] = 0.02
        scores.append(np.clip(score, 0.0, 1.0))
        labels.append(active)
    return scores, labels


def test_tuning_beats_a_plain_threshold_within_the_budget() -> None:
    scores, labels = _dev_like_frames()
    plain = score_gate(scores, labels, GateConfig(attack=0.5))
    best, results = tune_gate(scores, labels, max_false_per_minute=1.0, max_gap_frames=(0, 2, 4))
    assert len(results) > 1
    assert best.false_per_minute <= 1.0
    assert best.onset_recall > plain.onset_recall
    assert best.config.max_gap_frames > 0  # the dips are what it has to bridge


def test_tuning_returns_the_quietest_setting_when_nothing_fits_the_budget() -> None:
    scores, labels = _dev_like_frames(2)
    best, results = tune_gate(scores, labels, max_false_per_minute=-1.0, max_gap_frames=(0, 2))
    assert best.false_per_minute == min(r.false_per_minute for r in results)


def test_score_gate_matches_scoring_the_frames_by_hand() -> None:
    scores, labels = _dev_like_frames(3)
    config = GateConfig(0.5, release=0.25, max_gap_frames=3)
    pooled = score_gate(scores, labels, config)
    by_hand = sum(score_bargein(gate_frames(s, config), lab).hits for s, lab in zip(scores, labels, strict=True))
    assert pooled.hits == by_hand


def test_tune_gate_cli_reads_a_frame_dump(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scores, labels = _dev_like_frames(3)
    lengths = [s.size for s in scores]
    path = tmp_path / "frames.npz"
    np.savez_compressed(
        path,
        scores=np.concatenate(scores).astype(np.float32),
        labels=np.concatenate(labels),
        offsets=np.cumsum([0, *lengths], dtype=np.int64),
        item_ids=np.array([f"item{i}" for i in range(len(scores))]),
        frame_rate_hz=np.int64(C.FRAME_RATE_HZ),
    )
    loaded_scores, loaded_labels = T.load_frames(path)
    assert [s.size for s in loaded_scores] == lengths
    assert np.array_equal(loaded_labels[0], labels[0])

    out = tmp_path / "gate.json"
    assert T.main([str(path), "--max-false-per-min", "1.0", "--max-gap-frames", "0", "4", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "--gate-max-gap-frames" in printed
    payload = json.loads(out.read_text())
    assert payload["chosen"]["onset_recall"] > 0
    assert len(payload["searched"]) >= 2


def test_tune_gate_cli_reports_a_bad_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "nope.npz"
    assert T.main([str(missing), "--max-false-per-min", "4.0"]) == 1
    assert "cannot read" in capsys.readouterr().out


def _items() -> list[dev_runner.ItemResult]:
    scores, labels = _dev_like_frames(3)
    return [
        dev_runner.ItemResult(spec=None, row={"item_id": f"item{i}"}, vad=s.astype(np.float32), labels=lab)
        for i, (s, lab) in enumerate(zip(scores, labels, strict=True))
    ]


def test_dev_runner_scores_barge_ins_through_the_gate() -> None:
    plain, gated = _items(), _items()
    dev_runner.apply_threshold(plain, 0.5)
    dev_runner.apply_threshold(gated, 0.5, GateConfig(attack=0.5, release=0.25, max_gap_frames=4))
    assert sum(it.bargein.hits for it in gated) > sum(it.bargein.hits for it in plain)
    assert gated[0].row["bargein_hits"] == gated[0].bargein.hits


def test_dev_runner_frame_dump_round_trips(tmp_path: Path) -> None:
    items = _items()
    path = dev_runner.save_frames(items, tmp_path / "frames.npz")
    scores, labels = T.load_frames(path)
    assert len(scores) == len(items)
    assert np.allclose(scores[1], items[1].vad)
    assert np.array_equal(labels[2], items[2].labels)
    with np.load(path) as data:
        assert list(data["item_ids"]) == ["item0", "item1", "item2"]
        assert int(data["frame_rate_hz"]) == C.FRAME_RATE_HZ
