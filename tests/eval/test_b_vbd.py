"""Unit tests for Suite B on a tiny synthetic VoiceBank+DEMAND-shaped folder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import soxr

from earmark.eval.runs_log import read_runs, verify_log
from earmark.eval.suites import b_vbd as B
from earmark.eval.systems import Unprocessed
from tests.eval.synth import add_noise, speech_like

IDS = ("p232_001", "p232_002", "p257_001")


@pytest.fixture
def vbd_root(tmp_path: Path) -> Path:
    """Three 48 kHz clean/noisy pairs plus a noise log, laid out like the official zips."""
    rng = np.random.default_rng(5)
    root = tmp_path / "vbd"
    (root / "clean_testset_wav").mkdir(parents=True)
    (root / "noisy_testset_wav").mkdir()
    (root / "logfiles").mkdir()
    log = []
    for i, item in enumerate(IDS):
        clean16 = speech_like(rng, 1.5)
        clean48 = soxr.resample(clean16, 16000, 48000)
        snr = (2.5, 7.5, 12.5)[i]
        noisy48 = add_noise(clean48, snr, rng)
        sf.write(root / "clean_testset_wav" / f"{item}.wav", clean48, 48000, subtype="PCM_16")
        sf.write(root / "noisy_testset_wav" / f"{item}.wav", noisy48, 48000, subtype="PCM_16")
        log.append(f"{item} cafe {snr:e}")
    (root / "logfiles" / "log_testset.txt").write_text("\n".join(log) + "\n")
    return root


def test_list_utterances_pairs_and_parses_noise_log(vbd_root: Path) -> None:
    utts = B.list_utterances(vbd_root, expect_full=False)
    assert [u.item_id for u in utts] == list(IDS)
    assert [u.speaker for u in utts] == ["p232", "p232", "p257"]
    assert utts[1].noise == "cafe" and utts[1].snr_db == 7.5
    with pytest.raises(B.DataMissingError, match="expected 824"):
        B.list_utterances(vbd_root)


def test_missing_or_unpaired_data_is_reported(tmp_path: Path, vbd_root: Path) -> None:
    with pytest.raises(B.DataMissingError, match="fetch_eval_data"):
        B.list_utterances(tmp_path / "nowhere")
    (vbd_root / "noisy_testset_wav" / "p257_001.wav").unlink()
    with pytest.raises(B.DataMissingError, match="pair"):
        B.list_utterances(vbd_root, expect_full=False)


def test_load_resamples_to_16k(vbd_root: Path) -> None:
    x = B.load_wav_16k(vbd_root / "clean_testset_wav" / "p232_001.wav")
    assert x.dtype == np.float64 and x.size == 24000


def test_run_suite_on_synthetic_folder(vbd_root: Path) -> None:
    summary, rows = B.run_suite(Unprocessed(), root=vbd_root, workers=0, n_resamples=200)
    assert summary["suite"] == "B" and summary["n_items"] == 3 and summary["n_speakers"] == 2
    assert "n=2 speakers" in summary["speakers_note"]
    assert summary["system"]["inference_path"] == "unprocessed"
    assert summary["gates"] == []  # gates only apply to the full 824-utterance set
    assert set(summary["metrics"]) == set(B.SUMMARY_KEYS)
    assert summary["metrics"]["si_sdri"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert list(summary["by_snr_db"]) == ["2.5", "7.5", "12.5"]
    pesq = [r["pesq_wb"] for r in rows]
    assert pesq[0] < pesq[2]  # 2.5 dB SNR scores below 12.5 dB SNR
    assert all("errors" not in r for r in rows)


def test_check_gates_logic() -> None:
    ok = B.check_gates("unprocessed", {"pesq_wb": {"mean": 1.985}, "stoi": {"mean": 0.90}})
    assert [g["passed"] for g in ok] == [True, False]  # |0.90 - 0.921| > 0.02
    assert B.check_gates("gtcrn-vb", {"pesq_wb": {"mean": 2.83}})[0]["passed"]
    assert not B.check_gates("gtcrn-vb", {"pesq_wb": {"mean": 2.81}})[0]["passed"]
    assert not B.check_gates("gtcrn-vb", {})[0]["passed"]
    assert B.check_gates("earmark-m", {"pesq_wb": {"mean": 3.0}}) == []


def test_make_system_names() -> None:
    assert B.make_system("noisy").info.name == "unprocessed"
    with pytest.raises(ValueError):
        B.make_system("rnnoise")


def test_cli_writes_numbers_only_and_logs_run(vbd_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out" / "summary.json"
    rows_out = tmp_path / "out" / "rows.jsonl"
    runs = tmp_path / "out" / "runs.jsonl"
    code = B.main(
        [
            "--system", "unprocessed", "--root", str(vbd_root), "--workers", "0",
            "--metrics", "pesq_wb", "si_sdr", "--resamples", "100",
            "--out", str(out), "--rows-out", str(rows_out), "--log-run", "smoke", "--runs-log", str(runs),
        ]
    )  # fmt: skip
    assert code == 0
    summary = json.loads(out.read_text())
    assert summary["metrics"]["pesq_wb"]["n"] == 3
    assert len(rows_out.read_text().splitlines()) == 3
    assert {p.suffix for p in (tmp_path / "out").iterdir()} == {".json", ".jsonl"}  # no audio written
    (record,) = read_runs(runs)
    assert record["split"] == "smoke" and record["suite"] == "B" and record["inference_path"] == "unprocessed"
    assert verify_log(runs) == 1
