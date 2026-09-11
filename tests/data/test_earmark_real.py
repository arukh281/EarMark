"""Earmark-Real: clap alignment, close-talk labels, transcripts and the take layout.

Takes are synthesised from one 48 kHz "room" so the laptop (16 kHz) and the close-talk
phone (44.1 kHz) hear the same clap, as real recordings do.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import soxr

from earmark import constants as C
from earmark.data import earmark_real as R
from earmark.data.labels import num_frames

SR = C.SAMPLE_RATE
MASTER = 48000
SECONDS = 7.0
LAG_16K = 250  # the phone hears everything 250 samples (15.6 ms) later than the laptop
_CLAP_48K = np.random.default_rng(99).standard_normal(int(0.006 * MASTER)) * np.exp(
    -np.arange(int(0.006 * MASTER)) / (0.001 * MASTER)
)


def clap_burst(rng: np.random.Generator) -> np.ndarray:
    n = int(0.006 * SR)
    return rng.standard_normal(n) * np.exp(-np.arange(n) / (0.001 * SR))


def with_clap(rng: np.random.Generator, seconds: float, start: int, burst: np.ndarray, *, noise: float = 1e-3) -> np.ndarray:
    x = rng.standard_normal(int(seconds * SR)) * noise
    x[start : start + burst.size] += burst
    return x


def frame_centres(n: int) -> np.ndarray:
    return (np.arange(n) * C.HOP_LENGTH + C.WINDOW_LENGTH / 2) / SR


def room(seed: int, *, delay48: int, target_amp: float, other_amp: float, noise: float) -> np.ndarray:
    """48 kHz scene: clap at 0.5 s, target at 2-3 s and 5-6 s, another voice at 3.5-4.5 s."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(SECONDS * MASTER)) * noise
    start = int(0.5 * MASTER) + delay48
    x[start : start + _CLAP_48K.size] += _CLAP_48K
    for (a, b), f0, amp in (((2.0, 3.0), 200.0, target_amp), ((5.0, 6.0), 210.0, target_amp), ((3.5, 4.5), 330.0, other_amp)):
        i0, i1 = int(a * MASTER) + delay48, int(b * MASTER) + delay48
        t = np.arange(i1 - i0) / MASTER
        x[i0:i1] += amp * np.sin(2 * np.pi * f0 * t)
    return x


def make_take(root: Path, take_id: str, speaker: str, scenario: str, *, labels: str | None = None) -> None:
    folder = root / "takes" / take_id
    folder.mkdir(parents=True)
    seed = zlib.crc32(take_id.encode())
    meta = {"speaker_id": speaker, "scenario": scenario, "interferers": ["second_person"], "room": "living room"}
    (folder / "meta.json").write_text(json.dumps(meta))
    laptop = soxr.resample(room(seed, delay48=0, target_amp=0.1, other_amp=0.1, noise=1e-3), MASTER, SR)
    phone = soxr.resample(room(seed + 1, delay48=3 * LAG_16K, target_amp=0.3, other_amp=0.003, noise=1e-4), MASTER, 44100)
    sf.write(folder / "laptop.wav", laptop.astype(np.float32), SR, subtype="FLOAT")
    sf.write(folder / "closetalk.wav", phone.astype(np.float32), 44100, subtype="FLOAT")
    (folder / "transcript.tsv").write_text(
        f"start_s\tend_s\tspeaker\ttext\n2.0\t3.0\t{speaker}\thello there\n"
        f"5.0\t6.0\t{speaker}\tsee you soon\n3.5\t4.5\tother\tsomething else\n"
    )
    if labels is not None:
        (folder / "labels.txt").write_text(labels)


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("earmark_real")
    speakers = {
        "c1": {"role": "author", "accent": "Indian English", "consent_release": True},
        "v1": {"role": "volunteer", "accent": "Scottish English", "consent_release": False},
        "v2": {"role": "volunteer", "accent": "Nigerian English", "consent_release": True},
    }
    (root / "speakers.json").write_text(json.dumps(speakers))
    make_take(root, "t1", "c1", "second_person")
    make_take(root, "t2", "v1", "agent_voice", labels="5.0\t6.0\ttarget\n\\\t100.0\t2000.0\n")
    make_take(root, "t3", "v2", "cafe")
    return root


def test_find_clap_locates_the_onset(rng: np.random.Generator) -> None:
    burst = clap_burst(rng)
    start = int(round(1.234 * SR))
    x = with_clap(rng, 5.0, start, 0.8 * burst / np.abs(burst).max())
    assert abs(R.find_clap(x) - start) <= 16
    with pytest.raises(ValueError):
        R.find_clap(np.zeros(5))


@pytest.mark.parametrize("lag", [537, -300, 0])
def test_align_by_clap_recovers_the_lag(rng: np.random.Generator, lag: int) -> None:
    burst = clap_burst(rng)
    reference = with_clap(rng, 6.0, 2 * SR, burst)
    other = with_clap(rng, 6.0, 2 * SR + lag, 0.5 * burst, noise=2e-3)
    assert abs(R.align_by_clap(reference, other) - lag) <= 1


def test_apply_lag_and_drift() -> None:
    x = np.arange(10.0)
    np.testing.assert_array_equal(R.apply_lag(x, 3, 5), [3, 4, 5, 6, 7])
    np.testing.assert_array_equal(R.apply_lag(x, -2, 5), [0, 0, 0, 1, 2])
    np.testing.assert_array_equal(R.apply_lag(x, 8, 5), [8, 9, 0, 0, 0])
    assert R.estimate_drift(100, 116, 16000) == pytest.approx(1.001)
    with pytest.raises(ValueError):
        R.estimate_drift(0, 1, 0)


def test_closetalk_labels_use_a_percentile_reference_and_skip_the_clap(rng: np.random.Generator) -> None:
    x = rng.standard_normal(8 * SR) * 1e-4
    t = np.arange(SR) / SR
    x[2 * SR : 3 * SR] += 0.3 * np.sin(2 * np.pi * 200 * t)
    x[5 * SR : 6 * SR] += 0.3 * np.sin(2 * np.pi * 220 * t)
    x[3 * SR + 8000 : 4 * SR + 8000] += 0.003 * np.sin(2 * np.pi * 330 * t)  # another voice, -40 dB
    burst = clap_burst(rng)
    x[SR // 2 : SR // 2 + burst.size] += 5.0 * burst / np.abs(burst).max()
    labels = R.closetalk_labels(x, exclude=[(0, SR)])
    centres = frame_centres(labels.size)
    inside = ((centres > 2.05) & (centres < 2.95)) | ((centres > 5.05) & (centres < 5.95))
    assert labels[inside].all()
    assert not labels[(centres > 3.6) & (centres < 4.4)].any()
    assert not labels[centres < 1.0].any()
    assert not labels[centres > 6.2].any()


def test_load_takes_roles_splits_and_consent(release: Path) -> None:
    takes = R.load_takes(release)
    assert [t.take_id for t in takes] == ["t1", "t2", "t3"]
    assert [t.split for t in takes] == ["dev", "test", "test"]
    assert takes[0].speaker == "earmark_real:c1" and takes[0].accent == "Indian English"
    assert takes[0].labels_path is None and takes[1].labels_path is not None
    assert [t.take_id for t in R.load_takes(release, released_only=True)] == ["t1", "t3"]


def test_invalid_roles_and_scenarios_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "speakers.json").write_text(json.dumps({"x": {"role": "boss"}}))
    with pytest.raises(ValueError, match="role"):
        R.load_speakers(tmp_path)
    (tmp_path / "speakers.json").write_text(json.dumps({"x": {"role": "volunteer"}}))
    folder = tmp_path / "takes" / "bad"
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"speaker_id": "x", "scenario": "karaoke"}))
    with pytest.raises(ValueError, match="scenario"):
        R.load_takes(tmp_path)
    (folder / "meta.json").write_text(json.dumps({"speaker_id": "nobody", "scenario": "cafe"}))
    with pytest.raises(ValueError, match="unknown speaker"):
        R.load_takes(tmp_path)


def test_prepare_take_aligns_and_labels_from_the_close_talk_track(release: Path) -> None:
    take = R.load_takes(release)[0]
    prepared = R.prepare_take(take)
    assert prepared.label_source == "closetalk"
    assert abs(prepared.lag - LAG_16K) <= 2
    assert prepared.laptop.size == int(SECONDS * SR)
    assert prepared.vad.size == num_frames(prepared.laptop.size)
    centres = frame_centres(prepared.vad.size)
    target = ((centres > 2.05) & (centres < 2.95)) | ((centres > 5.05) & (centres < 5.95))
    assert prepared.vad[target].mean() > 0.97
    assert not prepared.vad[(centres > 3.6) & (centres < 4.4)].any()  # loud on the laptop, faint at the phone
    assert not prepared.vad[centres < 1.5].any()  # the clap never counts as speech
    own = [s.text for s in prepared.transcript if s.speaker == "c1"]
    assert own == ["hello there", "see you soon"]


def test_hand_labels_win_over_close_talk_labels(release: Path) -> None:
    prepared = R.prepare_take(R.load_takes(release)[1])
    assert prepared.label_source == "hand"
    centres = frame_centres(prepared.vad.size)
    assert prepared.vad[(centres >= 5.0) & (centres < 6.0)].all()
    assert not prepared.vad[centres < 4.9].any()


def test_transcripts_and_label_files(tmp_path: Path) -> None:
    tsv = tmp_path / "t.tsv"
    tsv.write_text("start_s\tend_s\tspeaker\ttext\n1.0\t2.0\ta\tfirst\n0.2\t0.5\tb\tearlier\nbad line\n")
    segs = R.load_transcript(tsv)
    assert [(s.start_s, s.speaker, s.text) for s in segs] == [(0.2, "b", "earlier"), (1.0, "a", "first")]
    js = tmp_path / "t.json"
    js.write_text(json.dumps([{"start_s": 0.0, "end_s": 1.0, "speaker": "a", "text": "hi"}]))
    assert R.load_transcript(js)[0].text == "hi"
    aud = tmp_path / "labels.txt"
    aud.write_text("0.5\t1.0\tspeech\n\\\t100.0\t4000.0\n2.0\t2.5\n")
    labels = R.load_audacity_labels(aud)
    assert [(s.start_s, s.end_s) for s in labels] == [(0.5, 1.0), (2.0, 2.5)]
    frames = R.labels_from_segments(segs, 300, speaker="a", hangover_frames=0)
    centres = frame_centres(300)
    np.testing.assert_array_equal(frames, (centres >= 1.0) & (centres < 2.0))
