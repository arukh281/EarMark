"""Kokoro agent voices: the voice split, prompts, the synthesis plan and shard writing."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from earmark import constants as C
from earmark.data import agent_voice as AV
from earmark.data.shards import ShardedCorpus, ShardWriter, finalize_dataset

GROUPS = ("af", "am", "bf", "bm")


def test_voice_list_and_language_codes() -> None:
    assert len(AV.ENGLISH_VOICES) == len(set(AV.ENGLISH_VOICES)) == 28
    assert {AV.voice_group(v) for v in AV.ENGLISH_VOICES} == set(GROUPS)
    assert AV.voice_lang_code("af_heart") == "a" and AV.voice_lang_code("bm_george") == "b"
    for bad in ("ef_dora", "jf_alpha", "zm_yunxi", "a_heart"):
        with pytest.raises(ValueError):
            AV.voice_lang_code(bad)


def test_voice_split_is_stratified_deterministic_and_disjoint() -> None:
    split = AV.split_voices()
    assert split == AV.split_voices()
    test = {v for v, s in split.items() if s == "test"}
    train = {v for v, s in split.items() if s == "train"}
    assert not test & train and test | train == set(AV.ENGLISH_VOICES)
    assert len(test) == 7  # 3 + 2 + 1 + 1 over the four groups
    for group in GROUPS:
        assert any(AV.voice_group(v) == group for v in test)
        assert any(AV.voice_group(v) == group for v in train)
    assert any(AV.split_voices(seed=s) != split for s in range(1, 5))


def test_prompts_are_cleaned_and_chosen_deterministically() -> None:
    raw = "  “Hello,”   she said.  It’s a long enough line of text here. "
    assert AV.clean_prompt(raw) == '"Hello," she said. It\'s a long enough line of text here.'
    assert AV.clean_prompt("too short") is None
    assert AV.clean_prompt("Ünïcödé " * 10) is None
    assert AV.clean_prompt("1234567890 " * 5) is None
    assert AV.clean_prompt(None) is None
    texts = [f"This is sentence number {i} with enough characters to be used." for i in range(50)]
    texts += [None, "short"]
    picked = AV.select_prompts(texts, 10, seed=3)
    assert picked == AV.select_prompts(texts, 10, seed=3) and len(set(picked)) == 10
    with pytest.raises(ValueError):
        AV.select_prompts(texts, 60)


def test_plan_speaks_each_prompt_once_with_balanced_voices() -> None:
    split = AV.split_voices()
    prompts = [f"Prompt number {i} is long enough to be spoken aloud clearly." for i in range(60)]
    plan = AV.plan_agent_utterances(split, prompts, seed=1)
    assert plan == AV.plan_agent_utterances(split, prompts, seed=1)
    assert len(plan) == 60 and sorted(r["text"] for r in plan) == sorted(prompts)
    assert len({r["utt_id"] for r in plan}) == 60
    for r in plan:
        assert r["split"] == split[r["voice_id"]]
        assert r["speaker"] == f"kokoro:{r['voice_id']}"
        assert r["lang_code"] == r["voice_id"][0]
    counts = Counter(r["voice_id"] for r in plan)
    assert set(counts) == set(split) and max(counts.values()) - min(counts.values()) <= 1
    with pytest.raises(ValueError):
        AV.plan_agent_utterances({}, prompts)


def stub_tts(text: str, voice: str) -> tuple[np.ndarray, int]:
    """One second of a voice-specific tone between 0.4 s pads (silence for 'silent' prompts)."""
    sr = AV.KOKORO_SAMPLE_RATE
    if "silent" in text:
        return np.zeros(sr, dtype=np.float32), sr
    f0 = 120.0 + 10.0 * AV.ENGLISH_VOICES.index(voice)
    t = np.arange(sr) / sr
    pad = np.zeros(int(0.4 * sr))
    return np.concatenate([pad, 0.3 * np.sin(2 * np.pi * f0 * t), pad]).astype(np.float32), sr


def test_synthesize_plan_writes_trimmed_16k_shards(tmp_path: Path) -> None:
    split = AV.split_voices()
    prompts = [f"Prompt {i} has plenty of characters for the stub voice." for i in range(5)]
    prompts.append("This one is silent but long enough to pass the filter.")
    plan = AV.plan_agent_utterances(split, prompts)
    seen: list[int] = []
    with ShardWriter(tmp_path / "k", prefix="k") as writer:
        stats = AV.synthesize_plan(plan, stub_tts, writer, progress=seen.append)
    finalize_dataset(tmp_path / "k", name="kokoro_test")
    corpus = ShardedCorpus(tmp_path / "k")
    assert stats["rows"] == 5 and stats["skipped"] == 1 and len(corpus) == 5 and len(seen) == 5
    for n in corpus.num_samples:
        assert C.SAMPLE_RATE <= int(n) <= int(1.25 * C.SAMPLE_RATE)
    assert set(corpus.column("source")) == {"kokoro"}
    for voice, group, spl, speaker in zip(
        corpus.column("voice_id"), corpus.group, corpus.column("split"), corpus.speaker, strict=True
    ):
        assert group == voice and spl == split[voice] and speaker == f"kokoro:{voice}"
