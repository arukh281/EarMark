"""The contract VAD rule: framing, windowed energy, -40 dB threshold, 50 ms hangover."""

from __future__ import annotations

import math

import numpy as np
import torch

from earmark import constants as C
from earmark.data import labels as L

SR = C.SAMPLE_RATE


def tone(amplitude: float, n: int, freq: float = 440.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * np.arange(n) / SR)


def test_frame_count_follows_uncentred_framing() -> None:
    assert L.num_frames(C.WINDOW_LENGTH - 1) == 0
    assert L.num_frames(C.WINDOW_LENGTH) == 1
    assert L.num_frames(C.WINDOW_LENGTH + C.HOP_LENGTH - 1) == 1
    assert L.num_frames(C.WINDOW_LENGTH + C.HOP_LENGTH) == 2
    assert L.num_frames(4 * SR) == 399


def test_frame_energy_is_windowed_energy(rng: np.random.Generator) -> None:
    x = rng.standard_normal(2000)
    w2 = np.sin(np.pi * np.arange(C.WINDOW_LENGTH) / C.WINDOW_LENGTH) ** 2
    manual = [
        np.sum(x[t * C.HOP_LENGTH : t * C.HOP_LENGTH + C.WINDOW_LENGTH] ** 2 * w2)
        for t in range(L.num_frames(x.size))
    ]
    np.testing.assert_allclose(L.frame_energy_np(x), manual, rtol=1e-12)
    np.testing.assert_allclose(L.frame_energy(torch.from_numpy(x)).numpy(), manual, rtol=1e-12)
    # Parseval: the windowed energy equals the STFT frame energy with the backward norm.
    frame = x[: C.WINDOW_LENGTH] * np.sqrt(w2)
    spec = np.fft.fft(frame)
    assert math.isclose(np.sum(np.abs(spec) ** 2) / C.N_FFT, manual[0], rel_tol=1e-10)


def test_threshold_is_minus_40_db_re_the_peak() -> None:
    n = 3200
    x = np.concatenate(
        [
            tone(1.0, n),
            np.zeros(n),
            tone(10 ** (-35 / 20), n),  # 35 dB down: active
            np.zeros(n),
            tone(10 ** (-45 / 20), n),  # 45 dB down: inactive
            np.zeros(n),
        ]
    )
    raw = L.vad_labels_np(x, hangover_frames=0)
    energy = L.frame_energy_np(x)
    np.testing.assert_array_equal(raw, energy > energy.max() * 10 ** (C.VAD_THRESHOLD_DB / 10))
    centre = lambda start: (start + n // 2 - C.WINDOW_LENGTH // 2) // C.HOP_LENGTH  # noqa: E731
    assert raw[centre(0)]
    assert raw[centre(2 * n)]
    assert not raw[centre(4 * n)]


def test_hangover_holds_exactly_the_contract_frames() -> None:
    n = 3200
    x = np.concatenate([tone(1.0, n), np.zeros(4 * n)])
    raw = L.vad_labels_np(x, hangover_frames=0)
    held = L.vad_labels_np(x)
    last = int(np.flatnonzero(raw)[-1])
    assert held[last + 1 : last + 1 + C.VAD_HANGOVER_FRAMES].all()
    assert not held[last + 1 + C.VAD_HANGOVER_FRAMES :].any()
    assert C.VAD_HANGOVER_FRAMES * C.HOP_LENGTH * 1000 // SR == C.VAD_HANGOVER_MS


def test_hangover_definition_and_torch_parity(rng: np.random.Generator) -> None:
    active = rng.random((4, 60)) < 0.15
    out = L.apply_hangover_np(active, 5)
    for row, labels in zip(active, out, strict=True):
        for t in range(active.shape[1]):
            assert labels[t] == row[max(0, t - 5) : t + 1].any()
    np.testing.assert_array_equal(out, L.apply_hangover(torch.from_numpy(active), 5).numpy())


def test_utterance_peak_reference_is_honoured() -> None:
    x = tone(0.01, 4000)
    own = L.vad_labels_np(x, hangover_frames=0)
    assert own.any()
    # Relative to an utterance peak 50 dB above this crop, nothing in it is active.
    peak = L.frame_energy_np(x).max() * 1e5
    assert not L.vad_labels_np(x, peak, hangover_frames=0).any()
    assert not L.vad_labels(torch.from_numpy(x), torch.tensor(peak), hangover_frames=0).any()


def test_torch_and_numpy_labels_agree(rng: np.random.Generator) -> None:
    x = rng.standard_normal((3, 5000)) * np.linspace(0.0, 1.0, 5000) ** 4
    np.testing.assert_array_equal(L.vad_labels_np(x), L.vad_labels(torch.from_numpy(x)).numpy())


def test_silence_is_never_active() -> None:
    assert not L.vad_labels_np(np.zeros(4000)).any()
    assert not L.vad_labels(torch.zeros(2, 4000)).any()


def test_intervals_round_trip() -> None:
    active = np.zeros(120, dtype=bool)
    active[10:20] = True
    active[50] = True
    active[100:120] = True
    np.testing.assert_array_equal(L.intervals_to_frames(L.frames_to_intervals(active), 120), active)


def test_intervals_use_frame_centres() -> None:
    labels = L.intervals_to_frames([(1.0, 1.5)], 200)
    centres = (np.arange(200) * C.HOP_LENGTH + C.WINDOW_LENGTH / 2) / SR
    np.testing.assert_array_equal(labels, (centres >= 1.0) & (centres < 1.5))
    shifted = L.intervals_to_frames([(1.2, 1.7)], 200, offset_s=0.2)
    np.testing.assert_array_equal(shifted, labels)


def test_active_power_ignores_silence(rng: np.random.Generator) -> None:
    x = np.zeros(16000)
    x[4000:8000] = rng.standard_normal(4000) * 0.1
    p = float(L.active_power(torch.from_numpy(x)))
    assert abs(10 * math.log10(p / 0.01)) < 0.3
    assert math.isclose(float(L.active_power_np(x)), p, rel_tol=1e-9)
    assert float(L.active_power(torch.zeros(1, 1000))) == 0.0
    mask = L.active_sample_mask(torch.tensor([[True, False, False, True]]), 5 * C.HOP_LENGTH + 7)
    assert mask.shape[-1] == 5 * C.HOP_LENGTH + 7
    assert mask[0, : 2 * C.HOP_LENGTH].all() and not mask[0, 2 * C.HOP_LENGTH : 3 * C.HOP_LENGTH].any()
    assert mask[0, 3 * C.HOP_LENGTH : 5 * C.HOP_LENGTH].all() and not mask[0, 5 * C.HOP_LENGTH :].any()
