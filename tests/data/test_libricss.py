"""LibriCSS parsing, enrolment lists, frame labels and the Colab channel-0 export."""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from earmark import constants as C
from earmark.data import libricss as L
from earmark.data.labels import apply_hangover_np, num_frames

SR = C.SAMPLE_RATE
MEETING = (
    "start_time\tend_time\tspeaker\tutterance_id\ttranscription\n"
    "0.50\t2.00\t1089\t1089-134686-0000\tHE HOPED THERE WOULD BE STEW\n"
    "1.50\t3.00\t121\t121-121726-0000\tALSO A POPULAR CONTRIVANCE\n"
    "4.00\t5.00\t1089\t1089-134686-0001\tSTUFF IT INTO YOU\n"
)
SESSIONS = (
    ("0L", "overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0"),
    ("OV10", "overlap_ratio_10.0_sil0.1_1.0_session1_actual10.2"),
)


def session(name: str = SESSIONS[1][1], condition: str = "OV10") -> L.MiniSession:
    index, actual = L.parse_session_name(name)
    utts = tuple(L.parse_meeting_info(MEETING))
    return L.MiniSession(name, condition, index, L.split_for_session(index), utts, actual)


def flac_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(x, dtype=np.float32), SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def seven_channel(seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return np.stack([(c + 1) * 0.05 * np.sin(2 * np.pi * (200.0 + 50.0 * c) * t) for c in range(7)], axis=1)


def test_parse_meeting_info_and_session_names() -> None:
    utts = L.parse_meeting_info(MEETING)
    assert [u.utt_id for u in utts] == ["1089-134686-0000", "121-121726-0000", "1089-134686-0001"]
    assert utts[0].speaker == "libri:1089" and utts[0].text == "HE HOPED THERE WOULD BE STEW"
    assert (utts[1].start_s, utts[1].end_s) == (1.5, 3.0)
    assert L.parse_session_name(SESSIONS[1][1]) == (1, 10.2)
    assert L.parse_session_name("overlap_ratio_0.0_sil0.1_1.0_session9") == (9, None)
    with pytest.raises(ValueError):
        L.parse_session_name("no_index_here")
    assert L.split_for_session(0) == "dev" and L.split_for_session(9) == "test"
    with pytest.raises(ValueError):
        L.split_for_session(10)


def test_frame_labels_regions_and_hangover() -> None:
    s = session()
    assert s.speakers == ("libri:1089", "libri:121") and L.target_cases(s) == s.speakers
    assert s.duration_s == 5.0
    n = num_frames(6 * SR)
    labels = L.frame_labels(s, "libri:1089", n)
    centres = (np.arange(n) * C.HOP_LENGTH + C.WINDOW_LENGTH / 2) / SR
    target = ((centres >= 0.5) & (centres < 2.0)) | ((centres >= 4.0) & (centres < 5.0))
    other = (centres >= 1.5) & (centres < 3.0)
    np.testing.assert_array_equal(labels["region"] == L.REGION_TARGET, target & ~other)
    np.testing.assert_array_equal(labels["region"] == L.REGION_INTERFERER, other & ~target)
    np.testing.assert_array_equal(labels["region"] == L.REGION_OVERLAP, target & other)
    np.testing.assert_array_equal(labels["region"] == L.REGION_SILENCE, ~target & ~other)
    np.testing.assert_array_equal(labels["target"], apply_hangover_np(target))
    np.testing.assert_array_equal(labels["interferer"], apply_hangover_np(other))
    with pytest.raises(KeyError):
        L.frame_labels(s, "libri:9999", n)


def test_speaker_info_jsonl_in_the_released_format() -> None:
    text = (
        json.dumps({"dataid": SESSIONS[0][1], "unused_uttid": {"6930": ["6930-75918-0001", "6930-75918-0000"], "8224": ["8224-274381-0007"]}})
        + "\n"
        + json.dumps({"dataid": SESSIONS[1][1], "unused_uttid": [["1089-134686-0005"], ["121-121726-0003"]]})
        + "\n"
    )
    info = L.parse_speaker_info_jsonl(text)
    assert info[SESSIONS[0][1]] == {
        "libri:6930": ("6930-75918-0000", "6930-75918-0001"),
        "libri:8224": ("8224-274381-0007",),
    }
    assert info[SESSIONS[1][1]] == {"libri:1089": ("1089-134686-0005",), "libri:121": ("121-121726-0003",)}
    with pytest.raises(ValueError):
        L.parse_speaker_info_jsonl('{"foo": 1}\n')
    needed = L.needed_enrolment_utts(info)
    assert needed["libri:6930"] == ("6930-75918-0000", "6930-75918-0001")
    assert L.needed_enrolment_utts(info, per_speaker=1)["libri:6930"] == ("6930-75918-0000",)
    assert set(L.needed_enrolment_utts(info, sessions=[session()])) == {"libri:1089", "libri:121"}


def test_derive_unused_enrolment_prefers_unreplayed_utterances() -> None:
    s = session()
    librispeech = ["1089-134686-0000", "1089-134686-0001", "1089-134686-0002", "121-121726-0000"]
    out = L.derive_unused_enrolment([s], librispeech)
    assert out[s.name]["libri:1089"] == ("1089-134686-0002",)
    assert out[s.name]["libri:121"] == ()  # its only utterance is replayed in the session


def test_suite_r_cases_pair_each_target_with_its_enrolment() -> None:
    s = session()
    cases = L.suite_r_cases([s], {s.name: {"libri:1089": ["1089-134686-0005"]}})
    assert len(cases) == 1  # libri:121 has no enrolment list, so it cannot be a target
    case = cases[0]
    assert case["target"] == "libri:1089" and case["split"] == "test" and case["condition"] == "OV10"
    assert case["n_target_utts"] == 2 and case["n_interferer_utts"] == 1
    assert case["enrol_utts"] == ["1089-134686-0005"]


def test_find_mini_sessions_on_a_release_tree(tmp_path: Path) -> None:
    for cond, name in reversed(SESSIONS):
        folder = tmp_path / "for_release" / cond / name / "transcription"
        folder.mkdir(parents=True)
        (folder / "meeting_info.txt").write_text(MEETING)
    found = L.find_mini_sessions(tmp_path / "for_release")
    assert [s.name for s in found] == [SESSIONS[0][1], SESSIONS[1][1]]
    assert [s.split for s in found] == ["dev", "test"]
    expected = tmp_path / "for_release" / "OV10" / SESSIONS[1][1] / "record" / "raw_recording.wav"
    assert found[1].recording == expected


def test_export_channel0_reads_the_zip_in_place(tmp_path: Path) -> None:
    zip_path = tmp_path / "for_release.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for cond, name in SESSIONS:
            prefix = f"for_release/{cond}/{name}/"
            zf.writestr(prefix + "transcription/meeting_info.txt", MEETING)
            buf = io.BytesIO()
            sf.write(buf, seven_channel(), SR, format="WAV", subtype="PCM_16")
            zf.writestr(prefix + "record/raw_recording.wav", buf.getvalue())
            zf.writestr(prefix + "clean/mix.wav", b"not audio, never read")
        zf.writestr("for_release/segment_libricss.py", "# helper\n")
    assert [z.name for z in L.zip_mini_sessions(zipfile.ZipFile(zip_path).namelist())] == [n for _, n in SESSIONS]
    out = tmp_path / "libricss_16k"
    progress: list[tuple[int, int]] = []
    sessions = L.export_channel0(zip_path, out, progress=lambda k, n, name: progress.append((k, n)))
    assert progress == [(1, 2), (2, 2)]
    assert [s.split for s in sessions] == ["dev", "test"]
    audio, sr = sf.read(str(sessions[1].recording))
    assert sr == SR and audio.ndim == 1 and audio.shape[0] == 2 * SR
    np.testing.assert_allclose(audio, seven_channel()[:, 0], atol=2e-4)
    back = L.load_exported_sessions(out)
    assert [(s.name, s.condition, s.split, s.utterances, s.recording) for s in back] == [
        (s.name, s.condition, s.split, s.utterances, s.recording) for s in sessions
    ]
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    with pytest.raises(FileNotFoundError):
        L.export_channel0(empty, tmp_path / "nothing")


def test_extract_librispeech_utts_streams_only_wanted_files(tmp_path: Path) -> None:
    tar_path = tmp_path / "test-clean.tar.gz"
    wanted = ["1089-134686-0000", "121-121726-0003"]
    with tarfile.open(tar_path, "w:gz") as tf:
        for utt in [*wanted, "1089-134686-0001"]:
            spk, chapter, _ = utt.split("-")
            data = flac_bytes(0.1 * np.sin(2 * np.pi * 220.0 * np.arange(SR // 2) / SR))
            info = tarfile.TarInfo(f"LibriSpeech/test-clean/{spk}/{chapter}/{utt}.flac")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        text = b"1089-134686-0000 HE HOPED\n"
        info = tarfile.TarInfo("LibriSpeech/test-clean/1089/134686/1089-134686.trans.txt")
        info.size = len(text)
        tf.addfile(info, io.BytesIO(text))
    found = L.extract_librispeech_utts(tar_path, wanted, tmp_path / "ls")
    assert set(found) == set(wanted)
    assert found["1089-134686-0000"] == L.librispeech_flac_path(tmp_path / "ls", "1089-134686-0000")
    assert not (tmp_path / "ls" / "1089" / "134686" / "1089-134686-0001.flac").exists()
    audio, sr = sf.read(str(found["121-121726-0003"]))
    assert sr == SR and audio.size == SR // 2
