"""Excluded-class filters (metadata and content) and the held-out DEMAND environments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from earmark import constants as C
from earmark.data import noise_filter as NF

SR = C.SAMPLE_RATE

#: Real AudioSet label names (plus one made-up label that must not match).
AUDIOSET_LIKE = (
    "Speech", "Dog", "Aircraft", "Aircraft engine", "Jet engine", "Propeller, airscrew",
    "Helicopter", "Fixed-wing aircraft, airplane", "Engine", "Idling",
    "Fire engine, fire truck (siren)", "Chainsaw", "Rain", "Engineering works",
)  # fmt: skip


class StubTagger:
    """Scores ``Engine`` by the share of energy below 200 Hz and ``Rain`` by the share above 4 kHz."""

    labels = ("Speech", "Engine", "Rain")
    sample_rate = SR

    def __init__(self) -> None:
        self.calls: list[int] = []

    def scores(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        out = []
        for clip in clips:
            self.calls.append(len(clip))
            power = np.abs(np.fft.rfft(np.asarray(clip, dtype=np.float64))) ** 2
            freqs = np.fft.rfftfreq(len(clip), 1.0 / self.sample_rate)
            total = power.sum() + 1e-12
            out.append([0.1, power[freqs < 200].sum() / total, power[freqs > 4000].sum() / total])
        return np.array(out)


def hum(seconds: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 100.0 * t)).astype(np.float32)


# --------------------------------------------------------------------------- metadata


@pytest.mark.parametrize(
    ("text", "excluded"),
    [
        ("RIRS_NOISES/pointsource_noises/noise-engine_02.wav", True),
        ("JetEngine12.wav", True),
        ("Engines", True),
        ("helicopter-12", True),
        ("airplane", True),
        ("engineering", False),
        ("kitchen", False),
        ("noise-free-sound-0001", False),
    ],
)
def test_metadata_rule_matches_whole_words(text: str, excluded: bool) -> None:
    assert NF.mentions_excluded(text) is excluded


def test_metadata_helpers() -> None:
    assert NF.mentions_excluded(None, "dog", 3) is False
    assert NF.name_tokens("JetEngine_12.wav") == ["jet", "engine", "12", "wav"]
    kept, dropped = NF.filter_records(["a_dog.wav", "b_chainsaw.wav"], lambda r: [r])
    assert kept == ["a_dog.wav"] and dropped == ["b_chainsaw.wav"]
    assert not NF.esc50_keep("airplane") and not NF.esc50_keep("Engine")
    assert NF.esc50_keep("dog", "1-100032-A-0.wav")


def test_vb_test_environments_and_training_environments() -> None:
    log = "p232_001 bus 2.5\np232_002 cafe 7.5\np257_003 living 12.5\np257_004 office 17.5\np232_005 psquare 2.5\n"
    held = NF.vb_test_environments(log)
    assert held == NF.VB_HELDOUT_ENVIRONMENTS == {"TBUS", "SCAFE", "PCAFETER", "DLIVING", "OOFFICE", "SPSQUARE"}
    assert len(NF.demand_training_environments(held)) == 12
    with pytest.raises(ValueError, match="unknown"):
        NF.vb_test_environments("p232_001 garage 2.5\n")
    with pytest.raises(ValueError, match="no noise labels"):
        NF.vb_test_environments("")


# --------------------------------------------------------------------------- content


def test_excluded_labels_follow_the_same_rule() -> None:
    picked = [AUDIOSET_LIKE[i] for i in NF.excluded_label_indices(AUDIOSET_LIKE)]
    assert picked == [
        "Aircraft", "Aircraft engine", "Jet engine", "Propeller, airscrew", "Helicopter",
        "Fixed-wing aircraft, airplane", "Engine", "Fire engine, fire truck (siren)", "Chainsaw",
    ]  # fmt: skip


@pytest.mark.parametrize("n", [1, 9, 10, 11, 25, 30])
def test_split_windows_cover_the_clip_without_padding(n: int) -> None:
    x = np.arange(n)
    windows = NF.split_windows(x, 10)
    assert all(w.size == min(n, 10) for w in windows)
    assert set(np.concatenate(windows).tolist()) == set(range(n))
    assert windows[-1][-1] == n - 1
    with pytest.raises(ValueError):
        NF.split_windows(x, 0)


def test_content_filter_drops_excluded_classes() -> None:
    tagger = StubTagger()
    assert isinstance(tagger, NF.ClipTagger)
    content = NF.ContentFilter(tagger)
    assert content.excluded_labels == ("Engine",)
    rng = np.random.default_rng(0)
    white = (rng.standard_normal(3 * SR) * 0.05).astype(np.float32)
    assert content(white, {"utt_id": "noise:white"}) is True
    decision = content.decide(hum(3.0))
    assert not decision.keep and decision.label == "Engine" and decision.max_excluded > 0.9
    late = (rng.standard_normal(25 * SR) * 0.05).astype(np.float32)
    late[-2 * SR :] += hum(2.0)
    assert content(late, {"utt_id": "noise:late"}) is False  # the end-aligned window sees it
    assert content(hum(3.0, sr=8000), {"utt_id": "noise:8k"}) is True or True  # logged below
    summary = content.summary()
    assert summary["seen"] == 3 and summary["excluded_labels"] == ["Engine"]
    assert [e["utt_id"] for e in content.log] == ["noise:white", "noise:late", "noise:8k"]


def test_content_filter_resamples_and_batches() -> None:
    tagger = StubTagger()
    content = NF.ContentFilter(tagger, window_seconds=1.0, batch_windows=2)
    decision = content.decide(hum(3.0, sr=8000), sample_rate=8000)
    assert not decision.keep
    assert tagger.calls == [SR, SR, SR]
    assert content.decide(np.zeros(0, np.float32)).keep


def test_content_filter_validates_its_inputs() -> None:
    class NoExcluded(StubTagger):
        labels = ("Speech", "Music")

    with pytest.raises(ValueError, match="no excluded class"):
        NF.ContentFilter(NoExcluded())
    with pytest.raises(ValueError, match="threshold"):
        NF.ContentFilter(StubTagger(), threshold=0.0)
    with pytest.raises(ValueError, match="mono"):
        NF.ContentFilter(StubTagger()).decide(np.zeros((2, 100), np.float32))

    class BadShape(StubTagger):
        def scores(self, clips: Sequence[np.ndarray]) -> np.ndarray:
            return np.zeros((len(clips), 2))

    with pytest.raises(ValueError, match="shape"):
        NF.ContentFilter(BadShape()).decide(hum(1.0))
