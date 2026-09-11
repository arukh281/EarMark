"""Exporters for VCTK, MUSAN music, RIRS_NOISES and DEMAND on small synthetic archives."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from earmark import constants as C
from earmark.data import corpora as K
from earmark.data.noise_filter import mentions_excluded
from earmark.data.shards import ShardedCorpus, add_pools, check_pools_disjoint, finalize_dataset

SR = C.SAMPLE_RATE
SPEAKER_INFO = (
    "ID  AGE  GENDER  ACCENTS  REGION COMMENTS \n"
    "p225  23  F    English    Southern  England\n"
    "p226  22  M    English    Surrey\n"
    "p232  23  M    English    Southern  England\n"
)
REAL = "RIRS_NOISES/real_rirs_isotropic_noises"
POINT = "RIRS_NOISES/pointsource_noises"


def audio_bytes(x: np.ndarray, sr: int, fmt: str = "WAV") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(x, dtype=np.float32), sr, format=fmt, subtype="PCM_16")
    return buf.getvalue()


def tone(seconds: float, sr: int, *, f0: float = 300.0, amp: float = 0.3, lead: float = 0.0, tail: float = 0.0) -> np.ndarray:
    t = np.arange(int(round(seconds * sr))) / sr
    body = amp * np.sin(2 * np.pi * f0 * t)
    return np.concatenate([np.zeros(int(round(lead * sr))), body, np.zeros(int(round(tail * sr)))])


def rir(rng: np.random.Generator, delay: int = 20, taps: int = 3000) -> np.ndarray:
    h = rng.standard_normal(taps) * np.exp(-np.arange(taps) / 800.0) * 0.1
    h[:delay] = 0.0
    h[delay] = 0.5
    return h


# --------------------------------------------------------------------------- VCTK


@pytest.fixture(scope="module")
def vctk_zip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("vctk") / "VCTK-Corpus-0.92.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("speaker-info.txt", SPEAKER_INFO)
        for spk in ("p225", "p226", "p232"):
            for num in (1, 2, 101):
                utt = f"{spk}_{num:03d}"
                for mic, amp in (("mic1", 0.3), ("mic2", 0.1)):
                    wave = tone(1.0, 48000, amp=amp, lead=0.4, tail=0.4)
                    zf.writestr(f"wav48_silence_trimmed/{spk}/{utt}_{mic}.flac", audio_bytes(wave, 48000, "FLAC"))
                zf.writestr(f"txt/{spk}/{utt}.txt", f"Sentence {num} read by {spk}.\n")
    return path


def test_vctk_member_parsing() -> None:
    names = [
        "wav48_silence_trimmed/p225/p225_001_mic1.flac",
        "wav48_silence_trimmed/p225/p225_001_mic2.flac",
        "VCTK/wav48_silence_trimmed/s5/s5_010_mic1.flac",
        "wav48_silence_trimmed/p232/p232_001_mic1.flac",
        "txt/p225/p225_001.txt",
    ]
    members = K.vctk_members(names)
    assert [(m.speaker, m.number, m.mic) for m in members] == [("p225", 1, "mic1"), ("s5", 10, "mic1")]
    assert K.vctk_group(99) == "block00" and K.vctk_group(100) == "block01"
    info = K.parse_vctk_speaker_info(SPEAKER_INFO)
    assert info["p225"] == {"age": "23", "gender": "F", "accent": "English", "region": "Southern England"}
    assert set(info) == {"p225", "p226", "p232"}


def test_export_vctk_reads_in_place_excludes_vb_speakers_and_trims(vctk_zip: Path, tmp_path: Path) -> None:
    out = tmp_path / "vctk_16k"
    stats = K.export_vctk(vctk_zip, out, cap_seconds=100.0)
    finalize_dataset(out, name="vctk_test")
    add_pools(out, enrol_seconds=1.0)
    corpus = ShardedCorpus(out)
    assert stats["speakers"] == 2 and stats["rows"] == 6 == len(corpus)
    assert set(corpus.speaker.astype(str)) == {"vctk:p225", "vctk:p226"}
    assert set(corpus.column("mic")) == {"mic1"}
    assert set(corpus.group.astype(str)) == {"block00", "block01"}
    ids = corpus.column("utt_id").astype(str)
    texts = dict(zip(ids, corpus.column("text"), strict=True))
    assert texts["vctk:p225_001_mic1"] == "Sentence 1 read by p225."
    assert set(corpus.column("accent")) == {"English"}
    for n in corpus.num_samples:
        assert 1.15 * SR <= int(n) <= 1.3 * SR  # one second of speech plus 0.1 s margins
    assert 0.25 < float(np.abs(corpus.audio(0)).max()) < 0.35  # mic1, not mic2
    check_pools_disjoint(corpus.speaker, corpus.group, corpus.column("pool"))
    assert {"enrol", "target"} <= set(corpus.column("pool").astype(str))


def test_export_vctk_caps_each_speaker(vctk_zip: Path, tmp_path: Path) -> None:
    stats = K.export_vctk(vctk_zip, tmp_path / "cap", cap_seconds=1.5)
    assert stats["rows"] == 4  # two utterances of about 1.2 s reach 1.5 s per speaker


# --------------------------------------------------------------------------- RIRS_NOISES


@pytest.fixture(scope="module")
def rirs_zip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(5)
    path = tmp_path_factory.mktemp("rirs") / "rirs_noises.zip"
    real = ["RVB2014_type1_rir_largeroom1_far_angla.wav", "air_type1_air_binaural_lecture_0_1.wav"]
    points = ["noise-free-sound-0000.wav", "noise-free-sound-0001.wav", "noise-engine-hum-0002.wav"]
    with zipfile.ZipFile(path, "w") as zf:
        for size, rooms in (("smallroom", ("Room001", "Room002")), ("mediumroom", ("Room001",))):
            for room in rooms:
                for k in range(1, 4):
                    zf.writestr(f"RIRS_NOISES/simulated_rirs/{size}/{room}/{room}-{k:05d}.wav", audio_bytes(rir(rng), SR))
            zf.writestr(f"RIRS_NOISES/simulated_rirs/{size}/rir_list", "--rir-id x --room-id y z.wav\n")
        for name in [*real, "mystery_take.wav"]:
            zf.writestr(f"{REAL}/{name}", audio_bytes(rir(rng), SR))
        zf.writestr(f"{REAL}/RVB2014_type1_noise_largeroom1_1.wav", audio_bytes(rng.standard_normal(3 * SR) * 0.05, SR))
        zf.writestr(f"{REAL}/rir_list", "".join(f"--rir-id {i:05d} --room-id r {REAL}/{n}\n" for i, n in enumerate(real, 1)))
        zf.writestr(
            f"{REAL}/noise_list",
            f"--noise-id 00001 --noise-type isotropic --room-linkage RVB2014_largeroom1 {REAL}/RVB2014_type1_noise_largeroom1_1.wav\n",
        )
        for name in points:
            zf.writestr(f"{POINT}/{name}", audio_bytes(rng.standard_normal(2 * SR) * 0.05, SR))
        zf.writestr(
            f"{POINT}/noise_list",
            "".join(f"--noise-id {n[:-4]} --noise-type point-source --bg-fg-type foreground {POINT}/{n}\n" for n in points),
        )
    return path


def test_rirs_members_are_classified_by_the_zip_lists() -> None:
    real_rirs = {"a_rir.wav"}
    iso = {"b_noise.wav"}
    classify = lambda name: K.classify_rirs_member(f"{REAL}/{name}", real_rirs=real_rirs, iso_noises=iso)  # noqa: E731
    assert classify("a_rir.wav").kind == "real_rir"
    assert classify("b_noise.wav").kind == "iso_noise"
    assert classify("c_noise.wav").kind == "real_rir"  # unlisted: held out
    assert K.classify_rirs_member(f"{REAL}/c_noise.wav").kind == "iso_noise"  # no lists: by name
    assert K.classify_rirs_member(f"{POINT}/x.wav").kind == "point_noise"
    assert K.classify_rirs_member("RIRS_NOISES/README") is None


def test_export_rirs_noises(rirs_zip: Path, tmp_path: Path) -> None:
    dropped_by_content: list[str] = []

    def keep(audio: np.ndarray, meta: dict[str, Any]) -> bool:
        assert audio.dtype == np.float32 and audio.ndim == 1
        if "noise-free-sound-0001" in meta["utt_id"]:
            dropped_by_content.append(meta["utt_id"])
            return False
        return True

    out = {k: tmp_path / k for k in ("sim", "noise", "real")}
    stats = K.export_rirs_noises(
        rirs_zip, out_rir_sim=out["sim"], out_noise_train=out["noise"], out_rir_real=out["real"],
        rirs_per_room=2, noise_segment_seconds=1.0, keep=keep,
    )
    for name, path in out.items():
        finalize_dataset(path, name=name)
    sim, noise, real = (ShardedCorpus(out[k]) for k in ("sim", "noise", "real"))
    assert stats["sim_rir"] == 6 == len(sim)  # three rooms, two each
    assert set(sim.column("kind")) == {"simulated"}
    assert set(sim.column("room").astype(str)) == {"smallroom/Room001", "smallroom/Room002", "mediumroom/Room001"}
    for i in range(len(sim)):
        assert abs(float(np.abs(sim.audio(i)).max()) - 0.9) < 1e-3
    real_ids = real.column("utt_id").astype(str)
    assert set(real.column("kind")) == {"real"} and len(real) == 3
    assert any("air_type1" in u for u in real_ids) and any("mystery_take" in u for u in real_ids)
    noise_ids = noise.column("utt_id").astype(str)
    assert set(noise.column("kind")) == {"isotropic", "point_source"}
    assert not any(mentions_excluded(u) for u in noise_ids)
    assert not any("noise-free-sound-0001" in u for u in noise_ids)
    assert stats["dropped"] == 1 and stats["dropped_content"] == len(dropped_by_content) == 2
    assert len(noise) == 3 + 2  # isotropic noise (3 s) and one kept point-source noise (2 s)


# --------------------------------------------------------------------------- DEMAND


def test_export_demand_splits_held_out_environments(tmp_path: Path) -> None:
    zips = {}
    for env, amp in (("DKITCHEN", 0.2), ("TBUS", 0.1)):
        path = tmp_path / f"{env}_16k.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(f"{env}/ch01.wav", audio_bytes(tone(3.0, SR, amp=amp), SR))
            zf.writestr(f"{env}/ch02.wav", audio_bytes(tone(3.0, SR, amp=0.5), SR))
        zips[env] = path
    stats = K.export_demand(
        zips, out_train=tmp_path / "train", out_heldout=tmp_path / "held", held_out={"tbus"},
        segment_seconds=1.0, keep=lambda audio, meta: meta["utt_id"] != "demand:DKITCHEN:001",
    )
    finalize_dataset(tmp_path / "train", name="train")
    finalize_dataset(tmp_path / "held", name="held")
    train, held = ShardedCorpus(tmp_path / "train"), ShardedCorpus(tmp_path / "held")
    assert train.column("utt_id").astype(str).tolist() == ["demand:DKITCHEN:000", "demand:DKITCHEN:002"]
    assert set(held.column("environment")) == {"TBUS"} and len(held) == 3
    assert set(train.column("split")) == {"train"} and set(held.column("split")) == {"heldout"}
    assert stats["dropped_content"] == 1 and stats["train_envs"] == 1 and stats["heldout_envs"] == 1
    assert abs(float(np.abs(train.audio(0)).max()) - 0.2) < 0.01  # channel 1, not channel 2
    broken = tmp_path / "BROKEN_16k.zip"
    with zipfile.ZipFile(broken, "w") as zf:
        zf.writestr("BROKEN/ch09.wav", audio_bytes(tone(1.0, SR), SR))
    with pytest.raises(FileNotFoundError):
        K.export_demand({"BROKEN": broken}, out_train=tmp_path / "t2", out_heldout=tmp_path / "h2", held_out=())


# --------------------------------------------------------------------------- MUSAN music


def test_export_musan_music_is_artist_disjoint(tmp_path: Path) -> None:
    root = tmp_path / "musan" / "music"
    tracks = {
        "fma": [("music-fma-0000", "Y", "pop", "Artist A"), ("music-fma-0001", "N", "rock", "Artist A"),
                ("music-fma-0002", "N", "jazz", "Artist B")],
        "rfm": [("music-rfm-0000", "N", "classical", "Artist C"), ("music-rfm-0001", "N", "engine-noise", "Artist D")],
    }  # fmt: skip
    for sub, rows in tracks.items():
        folder = root / sub
        folder.mkdir(parents=True)
        lines = []
        for k, (name, vocals, genre, artist) in enumerate(rows):
            sf.write(folder / f"{name}.wav", tone(3.0, SR, f0=200.0 + 20 * k).astype(np.float32), SR)
            # The two annotation column orders found in MUSAN.
            lines.append(f"{name} {genre} {vocals} {artist}" if sub == "rfm" else f"{name} {vocals} {genre} {artist}")
        (folder / "ANNOTATIONS").write_text("\n".join(lines) + "\n")
    stats = K.export_musan_music(
        root, tmp_path / "train", tmp_path / "held", excerpt_seconds=1.0, heldout_fraction=0.5, seed=0
    )
    split = K.artist_split(["Artist A", "Artist B", "Artist C", "Artist D"], heldout_fraction=0.5, seed=0)
    assert stats["dropped"] == 1
    kept = {"Artist A": 2, "Artist B": 1, "Artist C": 1}
    for name, out in (("train", tmp_path / "train"), ("heldout", tmp_path / "held")):
        expected = sum(n for artist, n in kept.items() if split[artist] == name)
        assert stats[name] == expected
        if expected:
            finalize_dataset(out, name=name)
            corpus = ShardedCorpus(out)
            assert set(corpus.column("split")) == {name}
            assert {split[a] for a in corpus.column("artist")} == {name}
            assert all(int(n) == SR for n in corpus.num_samples)
    parsed = K.parse_musan_music_annotations("music-x-1 classical N Someone Else\nmusic-x-2 Y pop Band\n")
    assert parsed["music-x-1"].artist == "Someone Else" and parsed["music-x-1"].vocals == "N"
    assert parsed["music-x-2"].genre == "pop"


# --------------------------------------------------------------------------- small helpers


def test_export_esc50_keeps_permitted_categories(tmp_path: Path) -> None:
    root = tmp_path / "ESC-50-master"
    (root / "meta").mkdir(parents=True)
    (root / "audio").mkdir()
    rows = [
        ("1-100032-A-0.wav", 1, 0, "dog", "True"),
        ("1-11687-A-47.wav", 1, 47, "airplane", "False"),
        ("2-109505-A-10.wav", 2, 10, "rain", "False"),
    ]
    lines = ["filename,fold,target,category,esc10,src_file,take"]
    for name, fold, target, category, esc10 in rows:
        sf.write(root / "audio" / name, tone(5.0, 44100, amp=0.1).astype(np.float32), 44100)
        lines.append(f"{name},{fold},{target},{category},{esc10},{name.split('-')[1]},A")
    (root / "meta" / "esc50.csv").write_text("\n".join(lines) + "\n")
    stats = K.export_esc50(tmp_path, tmp_path / "esc50_16k")
    finalize_dataset(tmp_path / "esc50_16k", name="esc50")
    corpus = ShardedCorpus(tmp_path / "esc50_16k")
    assert stats == {"rows": 2.0, "dropped": 1.0}
    assert set(corpus.column("category")) == {"dog", "rain"}
    assert set(corpus.column("source")) == {"esc50"} and set(corpus.column("split")) == {"test"}
    assert dict(zip(corpus.column("category"), corpus.column("license"), strict=True)) == {
        "dog": "CC BY", "rain": "CC BY-NC 3.0",
    }
    assert all(int(n) == 5 * SR for n in corpus.num_samples)
    with pytest.raises(FileNotFoundError):
        K.export_esc50(tmp_path / "missing", tmp_path / "none")


def test_esc50_meta_and_stable_hash() -> None:
    rows = K.parse_esc50_meta(
        "filename,fold,target,category,esc10,src_file,take\n"
        "1-100032-A-0.wav,1,0,dog,True,100032,A\n"
        "1-100038-A-14.wav,1,14,chirping_birds,False,100038,A\n"
    )
    assert rows[0]["esc10"] is True and rows[1]["esc10"] is False
    assert rows[1]["fold"] == 1 and rows[1]["target"] == 14 and rows[1]["category"] == "chirping_birds"
    assert K.stable_hash("a", 1) == K.stable_hash("a", 1) != K.stable_hash("a", 2)
    assert 0 <= K.stable_hash("x") < 2**32
