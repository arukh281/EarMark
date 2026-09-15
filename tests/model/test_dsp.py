"""WOLA, ERB and normalisation primitives in earmark.model.dsp."""

from __future__ import annotations

import math

import pytest
import torch

from earmark import constants as C
from earmark.model import dsp


def _noise(shape: tuple[int, ...], seed: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def test_window_is_power_complementary() -> None:
    w = dsp.sqrt_hann_window(dtype=torch.float64)
    assert w.shape == (C.WINDOW_LENGTH,)
    assert w[0] == 0.0
    total = w[: C.HOP_LENGTH] ** 2 + w[C.HOP_LENGTH :] ** 2
    torch.testing.assert_close(total, torch.ones_like(total), atol=1e-15, rtol=0)


@pytest.mark.parametrize(("dtype", "tol"), [(torch.float64, 1e-12), (torch.float32, 1e-6)])
def test_wola_perfect_reconstruction_with_identity_mask(dtype: torch.dtype, tol: float) -> None:
    x = _noise((2, 3200), seed=1, dtype=dtype)
    spec = dsp.stft(x)
    assert spec.shape == (2, dsp.num_frames(3200), C.N_BINS)
    y = dsp.istft(spec * torch.ones(C.N_BINS, dtype=dtype))
    assert y.shape == x.shape
    inner = slice(C.HOP_LENGTH, x.shape[-1] - C.HOP_LENGTH)
    assert (y[:, inner] - x[:, inner]).abs().max().item() < tol


def test_streaming_wola_matches_offline() -> None:
    x = _noise((1, 1600), seed=2)
    window = dsp.sqrt_hann_window(dtype=torch.float64)
    frames = dsp.frame_signal(x)
    offline, _ = dsp.overlap_add(dsp.synthesis_frame(dsp.analysis_frame(frames, window), window))
    tail = torch.zeros(1, C.HOP_LENGTH, dtype=torch.float64)
    blocks = []
    for t in range(frames.shape[1]):
        frame = dsp.synthesis_frame(dsp.analysis_frame(frames[:, t], window), window)
        block, tail = dsp.overlap_add(frame[:, None], tail)
        blocks.append(block)
    torch.testing.assert_close(torch.cat(blocks, dim=-1), offline, atol=1e-13, rtol=0)


def test_fft_norm_is_backward() -> None:
    impulse = torch.zeros(1, C.WINDOW_LENGTH, dtype=torch.float64)
    impulse[0, C.WINDOW_LENGTH // 2] = 1.0  # window value 1 at the centre
    spec = dsp.analysis_frame(impulse)
    torch.testing.assert_close(spec.abs(), torch.ones(1, C.N_BINS, dtype=torch.float64))


def test_erb_bands_tile_all_bins() -> None:
    edges = dsp.erb_band_edges()
    assert edges[0] == 0 and edges[-1] == C.N_BINS and len(edges) == C.ERB_BANDS + 1
    index = dsp.erb_band_index()
    assert index.shape == (C.N_BINS,)
    assert torch.equal(torch.bincount(index), torch.tensor(C.ERB_WIDTHS))
    matrix = dsp.erb_matrix(dtype=torch.float64)
    torch.testing.assert_close(matrix.sum(0), torch.ones(C.ERB_BANDS, dtype=torch.float64))


def test_erb_power_and_expand() -> None:
    spec = torch.full((3, C.N_BINS), 2.0 + 0.0j, dtype=torch.complex128)
    torch.testing.assert_close(dsp.erb_power(spec), torch.full((3, C.ERB_BANDS), 4.0, dtype=torch.float64))
    gains = torch.arange(C.ERB_BANDS, dtype=torch.float64)
    torch.testing.assert_close(dsp.erb_expand(gains), dsp.erb_band_index().to(torch.float64))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_exp_mean_scan_matches_recurrence(dtype: torch.dtype) -> None:
    x = _noise((2, 200, 5), seed=3, dtype=dtype) * 20.0 - 50.0  # dB-like, crosses 3 chunks
    init = _noise((2, 5), seed=4, dtype=dtype)
    means, last = dsp.exp_mean_scan(x, init)
    m = init
    expected = []
    for t in range(x.shape[1]):
        m = dsp.exp_mean_step(x[:, t], m)
        expected.append(m)
    ref = torch.stack(expected, dim=1)
    tol = 1e-10 if dtype == torch.float64 else 2e-4  # float32 on values near -50
    torch.testing.assert_close(means, ref, atol=tol, rtol=0)
    torch.testing.assert_close(last, ref[:, -1], atol=tol, rtol=0)


def test_exp_mean_scan_resumes_from_state() -> None:
    x = _noise((1, 150, 3), seed=5)
    init = torch.zeros(3, dtype=torch.float64)
    whole, _ = dsp.exp_mean_scan(x, init)
    first, carry = dsp.exp_mean_scan(x[:, :70], init)
    second, _ = dsp.exp_mean_scan(x[:, 70:], carry)
    torch.testing.assert_close(torch.cat([first, second], dim=1), whole, atol=1e-12, rtol=0)


def test_exp_mean_time_constant() -> None:
    x = torch.ones(1, C.FRAME_RATE_HZ, 1, dtype=torch.float64)  # one second of a unit step
    means, _ = dsp.exp_mean_scan(x, torch.zeros(1, dtype=torch.float64))
    assert means[0, -1, 0].item() == pytest.approx(1.0 - math.exp(-1.0), rel=1e-12)


def test_features_step_matches_sequence() -> None:
    x = _noise((2, 4000), seed=6)
    spec = dsp.stft(x)
    erb_state = dsp.erb_norm_init(dtype=torch.float64).expand(2, -1)
    unit_state = dsp.unit_norm_init(dtype=torch.float64).expand(2, -1)
    erb_seq, erb_last = dsp.erb_features(spec, erb_state)
    unit_seq, unit_last = dsp.unit_norm_features(spec[..., : C.DF_BINS], unit_state)
    for t in range(spec.shape[1]):
        erb_t, erb_state = dsp.erb_features_step(spec[:, t], erb_state)
        unit_t, unit_state = dsp.unit_norm_features_step(spec[:, t, : C.DF_BINS], unit_state)
        torch.testing.assert_close(erb_t, erb_seq[:, t], atol=1e-10, rtol=0)
        torch.testing.assert_close(unit_t, unit_seq[:, t], atol=1e-10, rtol=0)
    torch.testing.assert_close(erb_state, erb_last, atol=1e-10, rtol=0)
    torch.testing.assert_close(unit_state, unit_last, atol=1e-10, rtol=0)


def test_norm_initial_values() -> None:
    erb = dsp.erb_norm_init(dtype=torch.float64)
    unit = dsp.unit_norm_init(dtype=torch.float64)
    assert erb.shape == (C.ERB_BANDS,) and unit.shape == (C.DF_BINS,)
    assert erb[0].item() == pytest.approx(-60.0 + 20 * math.log10(C.N_FFT))
    assert erb[-1].item() == pytest.approx(-90.0 + 20 * math.log10(C.N_FFT))
    assert unit[0].item() == pytest.approx(1e-3 * C.N_FFT) and (unit > 0).all()


def _complex(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g, dtype=torch.float64),
                         torch.randn(*shape, generator=g, dtype=torch.float64))


def test_deep_filter_matches_naive_sum() -> None:
    batch, frames, bins = 2, 9, C.DF_BINS
    x = _complex((batch, frames, bins), seed=7)
    coefs = _complex((batch, frames, C.DF_ORDER, bins), seed=8)
    history = _complex((batch, C.DF_ORDER - 1, bins), seed=9)
    out, new_history = dsp.deep_filter(x, coefs, history)
    padded = torch.cat([history, x], dim=1)
    ref = torch.zeros_like(x)
    for t in range(frames):
        for tap in range(C.DF_ORDER):
            ref[:, t] += coefs[:, t, tap] * padded[:, t + C.DF_ORDER - 1 - tap]
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=0)
    torch.testing.assert_close(new_history, x[:, -(C.DF_ORDER - 1) :])


def test_deep_filter_identity_and_resume() -> None:
    x = _complex((1, 12, C.DF_BINS), seed=10)
    identity = torch.zeros(1, 12, C.DF_ORDER, C.DF_BINS, dtype=torch.complex128)
    identity[:, :, 0] = 1.0
    out, _ = dsp.deep_filter(x, identity)
    torch.testing.assert_close(out, x)
    coefs = _complex((1, 12, C.DF_ORDER, C.DF_BINS), seed=11)
    whole, _ = dsp.deep_filter(x, coefs)
    first, hist = dsp.deep_filter(x[:, :5], coefs[:, :5])
    second, _ = dsp.deep_filter(x[:, 5:], coefs[:, 5:], hist)
    torch.testing.assert_close(torch.cat([first, second], dim=1), whole, atol=1e-12, rtol=0)


def test_pad_to_hop() -> None:
    x = torch.ones(1, 330)
    padded, n = dsp.pad_to_hop(x)
    assert n == 330 and padded.shape[-1] == 480 and padded[0, 330:].abs().sum() == 0
