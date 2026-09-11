"""Unit tests for earmark.eval.metrics on synthetic signals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from earmark import constants as C
from earmark.eval import metrics as M
from tests.eval.synth import add_noise, speech_like


# ---------------------------------------------------------------- SI-SDR


def test_si_sdr_matches_orthogonal_construction(rng: np.random.Generator) -> None:
    ref = rng.standard_normal(16000)
    ref -= ref.mean()
    noise = rng.standard_normal(16000)
    noise -= noise.mean()
    noise -= noise.dot(ref) / ref.dot(ref) * ref  # exactly orthogonal to ref
    noise *= np.sqrt(ref.dot(ref) / noise.dot(noise)) * 10 ** (-10 / 20)  # 10 dB below
    assert M.si_sdr(ref, 0.5 * ref + 0.5 * noise) == pytest.approx(10.0, abs=1e-9)


def test_si_sdr_is_scale_invariant(speech: np.ndarray, rng: np.random.Generator) -> None:
    est = add_noise(speech, 5.0, rng)
    base = M.si_sdr(speech, est)
    assert M.si_sdr(speech, 3.7 * est) == pytest.approx(base, abs=1e-9)
    assert M.si_sdr(0.01 * speech, est) == pytest.approx(base, abs=1e-9)
    assert M.si_sdr(speech, speech) > 100.0


def test_si_sdr_zero_mean_option(rng: np.random.Generator) -> None:
    ref = rng.standard_normal(8000)
    est = ref + 5.0  # DC offset only
    assert M.si_sdr(ref, est) > 100.0
    assert M.si_sdr(ref, est, zero_mean=False) < 0.0


def test_si_sdr_improvement(speech: np.ndarray, rng: np.random.Generator) -> None:
    mix = add_noise(speech, 0.0, rng)
    better = speech + 0.1 * (mix - speech)
    assert M.si_sdr_improvement(speech, mix, mix) == pytest.approx(0.0, abs=1e-12)
    assert M.si_sdr_improvement(speech, better, mix) == pytest.approx(20.0, abs=0.2)


def test_si_sdr_rejects_bad_input() -> None:
    with pytest.raises(M.MetricError):
        M.si_sdr(np.zeros(100), np.ones(100))
    with pytest.raises(ValueError, match="lengths differ"):
        M.si_sdr(np.ones(10), np.ones(11))
    with pytest.raises(ValueError, match="1-D"):
        M.si_sdr(np.ones((2, 5)), np.ones((2, 5)))
    with pytest.raises(ValueError, match="NaN"):
        M.si_sdr(np.array([1.0, np.nan]), np.ones(2))


# ---------------------------------------------------------------- PESQ / STOI / ESTOI


def test_pesq_stoi_estoi_order_by_noise_level(speech: np.ndarray, rng: np.random.Generator) -> None:
    mild = add_noise(speech, 20.0, rng)
    harsh = add_noise(speech, -5.0, rng)
    p_clean, p_mild, p_harsh = (M.pesq_wb(speech, x) for x in (speech, mild, harsh))
    assert p_clean > 4.4
    assert p_clean > p_mild > p_harsh >= 1.0
    s_clean, s_mild, s_harsh = (M.stoi(speech, x) for x in (speech, mild, harsh))
    assert s_clean == pytest.approx(1.0, abs=1e-6)
    assert s_clean > s_mild > s_harsh
    e_clean, e_harsh = M.estoi(speech, speech), M.estoi(speech, harsh)
    assert e_clean == pytest.approx(1.0, abs=1e-6)
    assert e_harsh < e_clean


def test_pesq_requires_16k_and_raises_metric_error_on_silence(speech: np.ndarray) -> None:
    with pytest.raises(ValueError, match="16 kHz"):
        M.pesq_wb(speech, speech, sample_rate=8000)
    with pytest.raises(M.MetricError):
        M.pesq_wb(np.zeros(16000), np.zeros(16000))


# ---------------------------------------------------------------- composite


def test_composite_identity_and_degradation(speech: np.ndarray, rng: np.random.Generator) -> None:
    ident = M.composite(speech, speech)
    assert ident.llr == pytest.approx(0.0, abs=1e-9)
    assert ident.wss == pytest.approx(0.0, abs=1e-9)
    assert ident.segsnr == pytest.approx(35.0)
    assert ident.csig == ident.cbak == ident.covl == 5.0
    noisy = M.composite(speech, add_noise(speech, 0.0, rng))
    assert noisy.csig < ident.csig and noisy.cbak < ident.cbak and noisy.covl < ident.covl
    assert noisy.llr > 0.0 and noisy.wss > 0.0
    for v in (noisy.csig, noisy.cbak, noisy.covl):
        assert 1.0 <= v <= 5.0


def test_composite_reuses_given_pesq(speech: np.ndarray, rng: np.random.Generator) -> None:
    deg = add_noise(speech, 5.0, rng)
    a = M.composite(speech, deg)
    b = M.composite(speech, deg, pesq_score=a.pesq_wb)
    assert a == b
    c = M.composite(speech, deg, pesq_score=a.pesq_wb + 0.5)
    assert c.pesq_wb == a.pesq_wb + 0.5 and (c.llr, c.wss, c.segsnr) == (a.llr, a.wss, a.segsnr)
    expected_covl = np.clip(1.594 + 0.805 * c.pesq_wb - 0.512 * c.llr - 0.007 * c.wss, 1.0, 5.0)
    assert c.covl == pytest.approx(expected_covl)


def test_levinson_matches_scipy_toeplitz_solve(rng: np.random.Generator) -> None:
    from scipy.linalg import solve_toeplitz

    frames = rng.standard_normal((4, 480))
    r = M._autocorr(frames, 16)
    poly = M._levinson(r)
    for f in range(4):
        a = solve_toeplitz(r[f, :16], r[f, 1:17])
        np.testing.assert_allclose(poly[f], np.concatenate(([1.0], -a)), rtol=1e-8, atol=1e-10)


def test_wss_peak_search_matches_loop_reference(rng: np.random.Generator) -> None:
    """The vectorised nearest-peak search equals the loop in composite.m."""
    energy = rng.standard_normal((6, 25)) * 10
    slope = np.diff(energy, axis=1)
    k = 24

    def loop(e: np.ndarray, s: np.ndarray) -> np.ndarray:
        out = np.empty(k)
        for i in range(k):
            if s[i] > 0:
                n = i
                while n < k and s[n] > 0:
                    n += 1
                out[i] = e[n - 1]
            else:
                n = i
                while n >= 0 and s[n] <= 0:
                    n -= 1
                out[i] = e[n + 1]
        return out

    idx = np.arange(k)
    right = np.minimum.accumulate(np.where(slope <= 0, idx, k)[:, ::-1], axis=1)[:, ::-1]
    left = np.maximum.accumulate(np.where(slope > 0, idx, -1), axis=1)
    got = np.take_along_axis(energy, np.where(slope > 0, right - 1, left + 1), axis=1)
    want = np.stack([loop(energy[f], slope[f]) for f in range(energy.shape[0])])
    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------- framing and activity


def test_framing_follows_contract() -> None:
    assert M.num_frames(C.WINDOW_LENGTH - 1) == 0
    assert M.num_frames(C.WINDOW_LENGTH) == 1
    assert M.num_frames(C.SAMPLE_RATE) == 1 + (C.SAMPLE_RATE - C.WINDOW_LENGTH) // C.HOP_LENGTH
    x = np.arange(1000, dtype=np.float64)
    frames = M.frame_signal(x)
    np.testing.assert_array_equal(frames[2], x[2 * C.HOP_LENGTH : 2 * C.HOP_LENGTH + C.WINDOW_LENGTH])


def test_activity_mask_threshold_and_hangover() -> None:
    sr = C.SAMPLE_RATE
    x = np.zeros(sr)
    x[int(0.3 * sr) : int(0.5 * sr)] = np.sin(2 * np.pi * 440 * np.arange(int(0.2 * sr)) / sr)
    act = M.activity_mask(x)
    raw = M.activity_mask(x, hangover_frames=0)
    last_raw = int(np.flatnonzero(raw)[-1])
    assert int(np.flatnonzero(act)[-1]) == last_raw + C.VAD_HANGOVER_FRAMES
    assert int(np.flatnonzero(act)[0]) == int(np.flatnonzero(raw)[0])  # hangover is causal only
    assert not M.activity_mask(np.zeros(sr)).any()
    quiet = x.copy()
    quiet[int(0.7 * sr) : int(0.8 * sr)] = 1e-3 * np.sin(2 * np.pi * 440 * np.arange(int(0.1 * sr)) / sr)  # -60 dB
    assert act.sum() == M.activity_mask(quiet).sum()


def test_longest_run() -> None:
    assert M.longest_run([]) == 0
    assert M.longest_run([0, 1, 1, 0, 1, 1, 1, 0]) == 3
    assert M.longest_run([1, 1]) == 2


# ---------------------------------------------------------------- TSOS


def test_tsos_identity_and_uniform_attenuation(speech: np.ndarray) -> None:
    assert M.tsos(speech, speech).percent == 0.0
    assert M.tsos(speech, 0.5 * speech).percent == 0.0  # -6 dB is not over-suppression
    full = M.tsos(speech, 0.1 * speech)  # -20 dB is
    assert full.percent == 100.0
    assert full.os_frames == full.active_frames > 0
    assert full.max_os_s == pytest.approx(M.longest_run(M.activity_mask(speech)) / C.FRAME_RATE_HZ)


def test_tsos_threshold_is_level_independent(speech: np.ndarray) -> None:
    """gamma=0.1, p=0.3 flags uniform attenuation beyond 20*log10(0.684**(1/0.3)) = -11.0 dB."""
    edge_db = 20 * math.log10((1 - math.sqrt(0.1)) ** (1 / 0.3))
    for level in (1e-3, 1.0, 1e3):
        ref = level * speech
        assert M.tsos(ref, ref * 10 ** ((edge_db + 0.3) / 20)).percent == 0.0
        assert M.tsos(ref, ref * 10 ** ((edge_db - 0.3) / 20)).percent == 100.0


def test_tsos_counts_muted_half_and_ignores_silence(rng: np.random.Generator) -> None:
    ref = speech_like(rng, seconds=4.0, pauses=False)
    est = ref.copy()
    est[ref.size // 2 :] = 0.0
    res = M.tsos(ref, est)
    assert 45.0 < res.percent < 55.0
    assert res.max_os_s == pytest.approx(res.total_os_s, abs=0.03)
    # A second of leading silence adds no active frames (bar the one frame straddling the edge).
    padded = np.concatenate([np.zeros(16000), ref])
    lead = 16000 // C.HOP_LENGTH - 1  # frames lying entirely in the silence
    assert not M.activity_mask(padded)[:lead].any()
    extra = M.tsos(padded, padded).active_frames - M.tsos(ref, ref).active_frames
    assert extra in (0, 1)


def test_tsos_custom_active_mask(speech: np.ndarray) -> None:
    n = M.num_frames(speech.size)
    none_active = M.tsos(speech, 0.0 * speech, active=np.zeros(n, dtype=bool))
    assert math.isnan(none_active.percent) and none_active.active_frames == 0
    with pytest.raises(ValueError):
        M.tsos(speech, speech, active=np.ones(n + 1, dtype=bool))


# ---------------------------------------------------------------- interferer suppression


def test_interferer_suppression(rng: np.random.Generator) -> None:
    sr = C.SAMPLE_RATE
    target = np.zeros(2 * sr)
    target[: sr] = speech_like(rng, 1.0, pauses=False)
    interferer = np.zeros(2 * sr)
    interferer[sr // 2 :] = speech_like(rng, 1.5, pauses=False)
    mix = target + interferer
    region = M.interferer_region(target, interferer)
    first = int(np.flatnonzero(region)[0])
    assert first * C.HOP_LENGTH >= sr  # only after the target (plus hangover) stops
    assert M.interferer_suppression_db(mix, mix, region) == pytest.approx(0.0, abs=1e-12)
    assert M.interferer_suppression_db(mix, 0.1 * mix, region) == pytest.approx(20.0, abs=1e-9)
    assert M.interferer_suppression_db(mix, 0.0 * mix, region) == pytest.approx(120.0)
    assert math.isnan(M.interferer_suppression_db(mix, mix, np.zeros_like(region)))
    only = M.interferer_region(None, interferer)
    assert only.sum() > region.sum()
