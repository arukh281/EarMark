"""Unit tests for the barge-in event definition and its scores."""

from __future__ import annotations

import math

import numpy as np
import pytest

from earmark import constants as C
from earmark.eval import bargein as B

A = C.BARGEIN_MIN_ACTIVE_FRAMES  # 20
S = C.BARGEIN_MIN_SILENCE_FRAMES  # 30


def seq(*runs: tuple[int, int]) -> np.ndarray:
    """Build a frame mask from (value, length) runs."""
    return np.concatenate([np.full(n, bool(v)) for v, n in runs])


def test_contract_values_are_what_the_plan_says() -> None:
    assert (A, S, C.FRAME_RATE_HZ) == (20, 30, 100)


def test_single_onset_start_and_decision() -> None:
    onsets = B.detect_onsets(seq((0, 40), (1, 25), (0, 10)))
    assert onsets == [B.Onset(40, 40 + A - 1)]


def test_too_short_activity_or_silence_is_not_an_onset() -> None:
    assert B.detect_onsets(seq((0, 40), (1, A - 1), (0, 40))) == []
    two = B.detect_onsets(seq((0, 40), (1, 25), (0, S - 1), (1, 25)))
    assert [o.start for o in two] == [40]  # the second run follows only 290 ms of silence
    three = B.detect_onsets(seq((0, 40), (1, 25), (0, S), (1, 25)))
    assert [o.start for o in three] == [40, 95]


def test_blips_consume_the_silence() -> None:
    # A 1-frame blip resets the silence counter, so the following run is not an onset.
    assert B.detect_onsets(seq((0, 100), (1, 1), (0, 5), (1, 40))) == []


def test_initial_silence_behaviour() -> None:
    starts_active = seq((1, 30), (0, 5))
    assert B.detect_onsets(starts_active) == [B.Onset(0, A - 1)]
    assert B.detect_onsets(starts_active, initial_silence=False) == []
    short_lead = seq((0, 5), (1, 30))
    assert [o.start for o in B.detect_onsets(short_lead)] == [5]
    assert B.detect_onsets(short_lead, initial_silence=False) == []
    assert B.detect_onsets(np.zeros(0, dtype=bool)) == []


@pytest.mark.parametrize("initial_silence", [True, False])
def test_streaming_detector_matches_vectorised(initial_silence: bool) -> None:
    rng = np.random.default_rng(7)
    for trial in range(60):
        # Markov-ish random masks with runs on the scale of the thresholds.
        runs = []
        for _ in range(rng.integers(1, 25)):
            runs.append((int(rng.integers(0, 2)), int(rng.integers(1, 60))))
        mask = seq(*runs)
        det = B.OnsetDetector(initial_silence=initial_silence)
        streamed = [o for o in (det.push(bool(v)) for v in mask) if o is not None]
        assert streamed == B.detect_onsets(mask, initial_silence=initial_silence), trial


def test_score_perfect_detector_has_200ms_delay_and_no_false_alarms() -> None:
    ref = seq((0, 100), (1, 80), (0, 100), (1, 60), (0, 60))
    counts = B.score_bargein(ref, ref)
    assert counts.reference_onsets == counts.hits == 2
    assert counts.false_barge_ins == 0
    assert counts.median_delay_ms == pytest.approx(A * 1000 / C.FRAME_RATE_HZ)
    assert counts.onset_recall == 1.0 and counts.frame_recall == 1.0
    assert counts.silent_minutes == pytest.approx(260 / 6000)


def test_score_false_barge_ins_late_and_missed_onsets() -> None:
    n = 1000
    ref = np.zeros(n, dtype=bool)
    ref[200:300] = True  # target onset at 200
    ref[600:700] = True  # target onset at 600
    det = np.zeros(n, dtype=bool)
    det[50:80] = True  # interferer: false barge-in (decision at 69, target silent)
    det[215:300] = True  # hit, 150 ms late -> delay 35 frames
    det[850:900] = True  # false barge-in after the target stopped
    counts = B.score_bargein(det, ref)
    assert counts.hits == 1 and counts.reference_onsets == 2
    assert counts.delays_frames == [15 + A]
    assert counts.false_barge_ins == 2
    assert counts.false_per_minute == pytest.approx(2 / (800 / 6000))
    assert counts.onset_recall == 0.5
    assert counts.frame_recall == pytest.approx(85 / 200)


def test_detection_inside_target_speech_is_neither_hit_nor_false() -> None:
    ref = seq((0, 100), (1, 300), (0, 100))
    det = seq((0, 100), (1, 50), (0, 40), (1, 100), (0, 210))  # re-trigger mid-utterance
    counts = B.score_bargein(det, ref)
    assert counts.detected_onsets == 2 and counts.hits == 1 and counts.false_barge_ins == 0


def test_max_delay_window() -> None:
    ref = seq((0, 100), (1, 300), (0, 100))
    late = seq((0, 100), (0, 150), (1, 50), (0, 200))  # decision 150 + 20 frames after onset
    assert B.score_bargein(late, ref).hits == 0
    assert B.score_bargein(late, ref, max_delay_frames=200).hits == 1


def test_pool_counts_uses_ratio_of_sums() -> None:
    a = B.score_bargein(seq((0, 50), (1, 25), (0, 6025)), np.zeros(6100, dtype=bool))  # 1 FA in 61000 ms
    b = B.score_bargein(np.zeros(600, dtype=bool), np.zeros(600, dtype=bool))  # 0 FA in 6 s
    pooled = B.pool_counts([a, b])
    assert pooled.false_barge_ins == 1
    assert pooled.false_per_minute == pytest.approx(1 / (6700 / 6000))
    assert math.isnan(B.BargeInCounts().false_per_minute)
    assert set(pooled.summary()) >= {"false_barge_ins_per_min", "onset_recall", "median_onset_delay_ms"}


def test_score_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        B.score_bargein(np.zeros(10), np.zeros(11))


def test_frame_recall_threshold_is_the_largest_meeting_target(rng: np.random.Generator) -> None:
    labels = [rng.random(500) < 0.4 for _ in range(4)]
    scores = [np.where(lab, rng.uniform(0.3, 1.0, lab.size), rng.uniform(0.0, 0.6, lab.size)) for lab in labels]
    frozen = B.matched_threshold(scores, labels, 0.95)
    pos = np.concatenate([s[lab] for s, lab in zip(scores, labels, strict=True)])
    assert frozen.level == "frame" and frozen.split == "dev"
    assert np.mean(pos >= frozen.threshold) >= 0.95
    higher = np.min(pos[pos > frozen.threshold]) if np.any(pos > frozen.threshold) else np.inf
    assert np.mean(pos >= higher) < 0.95
    assert frozen.achieved_recall == pytest.approx(np.mean(pos >= frozen.threshold))
    assert frozen.as_dict()["threshold"] == frozen.threshold
    with pytest.raises(ValueError):
        B.matched_threshold(scores, labels, 0.0)


def test_onset_recall_threshold() -> None:
    ref = seq((0, 100), (1, 100), (0, 100), (1, 100), (0, 100))
    # First utterance scores high, second only moderately: recall 1.0 needs t <= 0.5.
    s = np.where(ref, 0.9, 0.1)
    s[300:400] = 0.5
    frozen = B.matched_threshold([s], [ref], 1.0, level="onset")
    assert frozen.level == "onset"
    assert frozen.achieved_recall == 1.0
    assert 0.1 < frozen.threshold <= 0.5
    half = B.onset_recall_threshold([s], [ref], 0.5)
    assert half.threshold > 0.5
    with pytest.raises(ValueError):
        B.matched_threshold([s], [ref], level="bogus")  # type: ignore[arg-type]


def test_binarize() -> None:
    np.testing.assert_array_equal(B.binarize([0.1, 0.5, 0.9], 0.5), [False, True, True])
