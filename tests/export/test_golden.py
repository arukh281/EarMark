"""Tests for the engine goldens and the float64 references behind them."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from earmark import constants as C
from earmark.export import blob as B
from earmark.export import golden as G
from earmark.export import reference as R
from earmark.model import dsp
from earmark.model.earmark_net import GroupedLinear


@pytest.fixture(scope="module")
def fresh() -> dict[str, B.Blob]:
    """Every golden, rebuilt once for this module."""
    return {name: B.unpack_blob(G.build_golden(name)) for name in G.GOLDENS}


# ------------------------------------------------------------------ files and determinism


def test_committed_goldens_are_in_sync() -> None:
    assert G.compare_goldens() == []


def test_goldens_rebuild_bit_identically_and_carry_the_contract(fresh: dict[str, B.Blob]) -> None:
    for name in ("ringbuf", "stft", "erb", "gru", "matvec", "weights_small"):
        assert G.build_golden(name) == G.build_golden(name), name
    for name, blob in fresh.items():
        assert blob.contract_hash == C.CONTRACT_HASH, name


def test_write_goldens_round_trips(tmp_path: Path) -> None:
    written = G.write_goldens(tmp_path, ["weights_small", "matvec"])
    assert {p.name for p in written} == {"weights_small.emwb", "weights_small.json", "matvec.emwb"}
    assert G.compare_goldens(tmp_path, ["weights_small", "matvec"]) == []
    (tmp_path / "matvec.emwb").write_bytes(G.build_golden("ringbuf"))
    assert G.compare_goldens(tmp_path, ["matvec"]) == ["matvec: tensor names differ"]
    assert G.compare_goldens(tmp_path, ["gru"])[0].startswith("gru: ")


def test_cli_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert G.main(["--check", "matvec", "weights_small"]) == 0
    assert "in sync" in capsys.readouterr().out


def test_unknown_golden_is_rejected() -> None:
    with pytest.raises(KeyError):
        G.build_golden("nope")


# ------------------------------------------------------------ goldens vs the model code


def test_stft_golden_equals_the_model_stft(fresh: dict[str, B.Blob]) -> None:
    golden = fresh["stft"]
    x = torch.from_numpy(golden["input"].astype(np.float64))[None]
    padded = torch.cat([torch.zeros(1, C.HOP_LENGTH, dtype=torch.float64), x], dim=-1)
    spec = torch.view_as_real(dsp.stft(padded)[0]).numpy()
    np.testing.assert_allclose(golden["spec"], spec, rtol=0, atol=1e-6 * np.abs(spec).max())
    np.testing.assert_array_equal(golden["window"], dsp.sqrt_hann_window().numpy())


def test_erb_golden_matches_the_streaming_step_functions(fresh: dict[str, B.Blob]) -> None:
    golden = fresh["erb"]
    spec = torch.view_as_complex(torch.from_numpy(golden["spec"].astype(np.float64)).contiguous())
    erb_state = torch.from_numpy(golden["erb_norm_init"].astype(np.float64))[None]
    unit_state = torch.from_numpy(golden["spec_norm_init"].astype(np.float64))[None]
    matrix = dsp.erb_matrix(dtype=torch.float64)
    for t in range(spec.shape[0]):
        feat, erb_state = dsp.erb_features_step(spec[t][None], erb_state, matrix)
        np.testing.assert_allclose(golden["erb_feat"][t], feat[0].numpy(), rtol=1e-6, atol=1e-6)
        unit, unit_state = dsp.unit_norm_features_step(spec[t][None, : C.DF_BINS], unit_state)
        np.testing.assert_allclose(golden["spec_feat"][t], torch.view_as_real(unit[0]).numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(golden["erb_norm_final"], erb_state[0].numpy(), rtol=1e-6)
    # The ERB golden covers exact silence, where power is 0 and features hit the eps floor.
    assert (golden["erb_power"] == 0).any()


def test_gru_reference_matches_torch(fresh: dict[str, B.Blob]) -> None:
    golden = fresh["gru"]
    for case, (inp, hidden, layers, frames) in G.GRU_CASES.items():
        gru = torch.nn.GRU(inp, hidden, layers, batch_first=True, dtype=torch.float64)
        with torch.no_grad():
            for layer in range(layers):
                for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(gru, f"{key}_l{layer}").copy_(torch.from_numpy(golden[f"{case}.{key}_l{layer}"].astype(np.float64)))
            x = torch.from_numpy(golden[f"{case}.x"].astype(np.float64))[None]
            h0 = torch.from_numpy(golden[f"{case}.h0"].astype(np.float64))[:, None]
            y, h = gru(x, h0)
        assert golden[f"{case}.y"].shape == (frames, hidden)
        np.testing.assert_allclose(golden[f"{case}.y"], y[0].numpy(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(golden[f"{case}.h_final"], h[:, 0].numpy(), rtol=0, atol=1e-6)


def test_matvec_goldens_match_numpy_and_the_model_grouped_linear(fresh: dict[str, B.Blob]) -> None:
    golden = fresh["matvec"]
    for index, (rows, cols) in enumerate(G.MATVEC_SHAPES):
        w, x, b = (golden[f"case{index}.{k}"].astype(np.float64) for k in "wxb")
        assert w.shape == (rows, cols)
        np.testing.assert_allclose(golden[f"case{index}.y"], w @ x + b, rtol=1e-6, atol=1e-6)
    inp, out, groups = G.GROUPED_DIMS
    layer = GroupedLinear(inp, out, groups).double()
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(golden["grouped.w"].astype(np.float64)))
        layer.bias.copy_(torch.from_numpy(golden["grouped.b"].astype(np.float64)))
        y = layer(torch.from_numpy(golden["grouped.x"].astype(np.float64)))
    np.testing.assert_allclose(golden["grouped.y"], y.numpy(), rtol=1e-6, atol=1e-6)


def test_ringbuf_golden_replays_against_the_model(fresh: dict[str, B.Blob]) -> None:
    golden = fresh["ringbuf"]
    model = R.RingBufferModel(int(golden["capacity"][0]))
    inputs = golden["input"]
    in_pos = out_pos = 0
    outputs = []
    for (kind, n), count, size in zip(golden["ops"], golden["expected_counts"], golden["expected_sizes"], strict=True):
        if kind == G.RING_WRITE:
            got = model.write(inputs[in_pos : in_pos + n])
            in_pos += n
        elif kind == G.RING_WRITE_ZEROS:
            got = model.write_zeros(n)
        elif kind == G.RING_DISCARD:
            got = model.discard(n)
        else:
            values = model.read(n) if kind == G.RING_READ else model.peek(n)
            outputs.append(values)
            got = len(values)
        assert (got, model.size) == (count, size)
    np.testing.assert_array_equal(np.concatenate(outputs).astype(np.float32), golden["expected_output"])
    # The schedule exercises every op and both the empty and the full ring.
    assert set(golden["ops"][:, 0]) == {0, 1, 2, 3, 4}
    assert golden["expected_sizes"].min() == 0 and golden["expected_sizes"].max() == golden["capacity"][0]


def test_ring_buffer_model_partial_transfers() -> None:
    ring = R.RingBufferModel(3)
    assert ring.write(np.array([1.0, 2.0, 3.0, 4.0])) == 3
    assert ring.peek(2).tolist() == [1.0, 2.0]
    assert ring.read(5).tolist() == [1.0, 2.0, 3.0]
    assert ring.write_zeros(4) == 3 and ring.discard(10) == 3 and ring.size == 0
    with pytest.raises(ValueError):
        R.RingBufferModel(0)


# ----------------------------------------------------------------------------- resampler


@pytest.mark.parametrize(("in_rate", "out_rate"), G.RESAMPLER_CASES)
def test_resampler_design_is_linear_phase_with_unit_gain(in_rate: int, out_rate: int) -> None:
    design = R.design_resampler(in_rate, out_rate)
    assert math.gcd(design.up, design.down) == 1
    assert design.up * in_rate == design.down * out_rate
    assert design.prototype_taps == design.taps_per_phase * design.up <= R.RESAMPLER_MAX_TAPS
    taps = R.resampler_taps(design).astype(np.float64)
    assert taps.sum() == pytest.approx(design.up, rel=1e-6)
    np.testing.assert_allclose(taps, taps[::-1], rtol=0, atol=1e-7 * np.abs(taps).max())

    # Frequency response at the upsampled rate, normalised to unit passband gain.
    n_fft = 1 << max(16, int(np.ceil(np.log2(design.prototype_taps))) + 3)
    response = np.abs(np.fft.rfft(taps, n_fft)) / design.up
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / (in_rate * design.up))
    low = min(in_rate, out_rate)
    passband = response[freqs <= R.RESAMPLER_PASSBAND_FRACTION * low]
    stopband = response[(freqs >= (1 - R.RESAMPLER_PASSBAND_FRACTION) * low) & (freqs <= in_rate * design.up / 2)]
    assert np.abs(20 * np.log10(passband)).max() < 1e-4  # dB of ripple
    assert 20 * np.log10(stopband.max()) < -110.0


@pytest.mark.parametrize("device_rate", [48000, 44100])
def test_reference_round_trip_snr_is_above_90_db(device_rate: int) -> None:
    up = R.design_resampler(16000, device_rate)
    down = R.design_resampler(device_rate, 16000)
    t = np.arange(8000) / 16000.0
    tones = [(173.0, 0.3, 0.1), (997.0, 0.25, 1.3), (2512.0, 0.2, 2.2), (6100.0, 0.05, 2.9)]

    def signal(time: np.ndarray) -> np.ndarray:
        return sum(a * np.sin(2 * np.pi * f * time + p) for f, a, p in tones)

    mid = R.resample_reference(signal(t).astype(np.float32), up, R.resampler_taps(up))
    assert len(mid) == math.ceil(len(t) * up.up / up.down)
    y = R.resample_reference(mid, down, R.resampler_taps(down))
    delay = up.delay_seconds + down.delay_seconds
    want = signal(np.arange(len(y)) / 16000.0 - delay)
    sl = slice(400, None)
    snr = 10 * np.log10(np.sum(want[sl] ** 2) / np.sum((y[sl] - want[sl]) ** 2))
    assert snr > 90.0


def test_resampler_rejects_unsupported_rates() -> None:
    with pytest.raises(ValueError):
        R.design_resampler(44099, 16000)
    with pytest.raises(ValueError):
        R.design_resampler(0, 16000)
    identity = R.design_resampler(16000, 16000)
    assert identity.identity and identity.delay_seconds == 0.0
    np.testing.assert_array_equal(R.resample_reference(np.ones(5, np.float32), identity, R.resampler_taps(identity)), np.ones(5))


def test_resampler_golden_blocks_cover_the_input(fresh: dict[str, B.Blob]) -> None:
    golden = fresh["resampler"]
    for index, (in_rate, out_rate) in enumerate(G.RESAMPLER_CASES):
        prefix = f"case{index}."
        assert golden[prefix + "blocks"].sum() == len(golden[prefix + "input"])
        design = R.design_resampler(in_rate, out_rate)
        assert len(golden[prefix + "expected"]) == math.ceil(len(golden[prefix + "input"]) * design.up / design.down)
