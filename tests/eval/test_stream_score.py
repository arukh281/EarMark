"""Unit tests for score-as-you-go evaluation."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from earmark import constants as C
from earmark.eval import metrics as M
from earmark.eval import stream_score as SS
from earmark.eval.systems import EnhancerOutput, SystemInfo, Unprocessed
from tests.eval.synth import add_noise, speech_like


class HalfGain:
    """Toy enhancer: halves the input and reports a VAD from input energy."""

    sample_rate = C.SAMPLE_RATE

    def __init__(self) -> None:
        self.info = SystemInfo(name="half-gain", inference_path="pytorch-offline", params=0, mmac_per_s=0.0)

    def enhance(self, mixture: np.ndarray, embedding: np.ndarray | None = None) -> EnhancerOutput:
        vad = M.activity_mask(mixture.astype(np.float64)).astype(np.float32)
        return EnhancerOutput(0.5 * mixture, vad)


def _load(seed: int) -> SS.ItemAudio:
    rng = np.random.default_rng(seed)
    clean = speech_like(rng, 2.0)
    noisy = add_noise(clean, 5.0, rng)
    return SS.ItemAudio(mixture=noisy.astype(np.float32), reference=clean, reference_vad=M.activity_mask(clean))


def _items(n: int) -> list[SS.EvalItem]:
    return [SS.EvalItem(f"utt{i}", f"spk{i % 2}", partial(_load, i), {"snr_db": 5.0}) for i in range(n)]


def test_streaming_scorer_matches_offline_metrics(rng: np.random.Generator) -> None:
    sr = C.SAMPLE_RATE
    ref = np.concatenate([speech_like(rng, 2.0, pauses=False), np.zeros(sr)])
    interferer = np.concatenate([np.zeros(sr), speech_like(rng, 2.0, pauses=False)])
    mix = ref + 0.5 * interferer + 0.01 * rng.standard_normal(ref.size)
    est = 0.8 * ref + 0.05 * interferer
    est[5000:9000] *= 0.05  # a stretch of over-suppression

    scorer = SS.StreamingScorer()
    pos = 0
    while pos < ref.size:  # ragged chunk sizes, including ones shorter than a hop
        step = int(rng.integers(1, 2500))
        sl = slice(pos, pos + step)
        scorer.update(ref[sl], est[sl], mix[sl], interferer[sl])
        pos += step
    res = scorer.result()

    assert res["frames"] == M.num_frames(ref.size)
    assert res["si_sdr"] == pytest.approx(M.si_sdr(ref, est), abs=1e-7)
    assert res["si_sdri"] == pytest.approx(M.si_sdr_improvement(ref, est, mix), abs=1e-7)
    offline = M.tsos(ref, est)
    assert (res["tsos_percent"], res["tsos_os_frames"], res["tsos_active_frames"]) == (
        offline.percent,
        offline.os_frames,
        offline.active_frames,
    )
    assert res["tsos_os_frames"] > 0
    region = M.interferer_region(ref, interferer)
    assert res["interferer_suppression_db"] == pytest.approx(M.interferer_suppression_db(mix, est, region), abs=1e-9)


def test_streaming_scorer_validates_inputs(rng: np.random.Generator) -> None:
    scorer = SS.StreamingScorer()
    with pytest.raises(ValueError):
        scorer.result()
    x = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="same length"):
        scorer.update(x, x[:-1])
    scorer.update(x, x, x)
    with pytest.raises(ValueError, match="same set"):
        scorer.update(x, x)
    silent = SS.StreamingScorer()
    silent.update(np.zeros(1000), x)
    assert math.isnan(silent.result()["si_sdr"])


def test_utterance_metrics_groups_and_errors(speech: np.ndarray, rng: np.random.Generator) -> None:
    mix = add_noise(speech, 0.0, rng)
    out = SS.utterance_metrics(speech, mix, mix, metrics=SS.UTTERANCE_METRICS[:-1])
    assert out["si_sdri"] == pytest.approx(0.0, abs=1e-9)
    assert {"pesq_wb", "stoi", "estoi", "si_sdr", "csig", "cbak", "covl", "segsnr", "tsos_percent"} <= set(out)
    assert "errors" not in out
    silent = SS.utterance_metrics(np.zeros(16000), np.zeros(16000), metrics=("pesq_wb", "si_sdri"))
    assert math.isnan(silent["pesq_wb"]) and set(silent["errors"]) == {"pesq_wb", "si_sdri"}
    target_absent = SS.utterance_metrics(None, mix, mix, metrics=("pesq_wb", "si_sdr"))
    assert target_absent == {}
    with pytest.raises(ValueError, match="unknown metrics"):
        SS.utterance_metrics(speech, mix, metrics=("mos",))


def test_score_items_in_process_rows_and_bargein() -> None:
    seen: list[str] = []
    rows = SS.score_items(
        _items(3), HalfGain(), metrics=("si_sdr", "si_sdri", "stoi"), vad_threshold=0.5, on_row=lambda r: seen.append(r["item_id"])
    )
    assert [r["item_id"] for r in rows] == ["utt0", "utt1", "utt2"] == seen
    for r in rows:
        assert r["duration_s"] == pytest.approx(2.0)
        assert r["snr_db"] == 5.0 and r["cluster"] in {"spk0", "spk1"}
        assert r["bargein_reference_onsets"] >= 1
        assert isinstance(r["bargein_delays_frames"], list)
    pooled = SS.pooled_bargein(rows)
    assert pooled is not None and pooled["reference_onsets"] == sum(r["bargein_reference_onsets"] for r in rows)
    summary = SS.summarize_rows(rows, ["si_sdr", "stoi", "missing"], n_resamples=200)
    assert summary["si_sdr"]["n"] == 3 and summary["missing"]["n"] == 0
    assert summary["si_sdr"]["ci95"][0] <= summary["si_sdr"]["mean"] <= summary["si_sdr"]["ci95"][1]
    clustered = SS.summarize_rows(rows, ["si_sdr"], cluster_level=True, n_resamples=200)
    assert clustered["si_sdr"]["unit"] == "cluster" and clustered["si_sdr"]["n_units"] == 2


def test_score_items_pool_matches_in_process_and_writes_no_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    items = _items(4)
    inline = SS.score_items(items, Unprocessed(), metrics=("pesq_wb", "si_sdr"), workers=0)
    pooled = SS.score_items(items, Unprocessed(), metrics=("pesq_wb", "si_sdr"), workers=2, max_in_flight=2)
    assert [r["item_id"] for r in pooled] == [r["item_id"] for r in inline]
    for a, b in zip(inline, pooled, strict=True):
        assert a["pesq_wb"] == pytest.approx(b["pesq_wb"]) and a["si_sdr"] == pytest.approx(b["si_sdr"])
    assert list(tmp_path.rglob("*")) == []  # scoring never touches the disk


def test_score_items_rejects_bad_system_output() -> None:
    class Broken:
        sample_rate = C.SAMPLE_RATE
        info = SystemInfo(name="broken", inference_path="pytorch-offline")

        def enhance(self, mixture: np.ndarray, embedding: np.ndarray | None = None) -> np.ndarray:
            return mixture[:-1]

    with pytest.raises(ValueError, match="samples"):
        SS.score_items(_items(1), Broken())
    with pytest.raises(ValueError):
        SS.score_items(_items(1), Unprocessed(), workers=-1)


def test_gallery_cap_is_ten() -> None:
    assert SS.GALLERY_MAX_CLIPS == 10
