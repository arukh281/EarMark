"""The "agent's own voice" interferer set: Kokoro-82M voices, prompts, the voice split and synthesis.

About 2,000 utterances are synthesised on Colab (``notebooks/colab_kokoro_agent_voice.py``)
from LibriTTS-R training text. The split is by voice ID: training examples only ever use
``train`` voices, and the held-out ``test`` voices feed Earmark-Synth's agent-voice
condition. The split is stratified over the four American/British female/male groups, so
both sides cover every group.

:func:`synthesize_plan` renders a plan through any ``tts(text, voice) -> (audio, rate)``
callable into int16 shards; :func:`kokoro_tts` builds that callable from the ``kokoro``
package (Colab only; the tests use a stub).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from earmark import constants as C
from earmark.data.corpora import stable_hash
from earmark.data.shards import ShardWriter, resample_to_contract, to_mono, trim_silence
from earmark.data.splits import KOKORO, speaker_key

__all__ = [
    "ENGLISH_VOICES",
    "KOKORO_REPO",
    "KOKORO_SAMPLE_RATE",
    "TtsFn",
    "clean_prompt",
    "kokoro_tts",
    "plan_agent_utterances",
    "select_prompts",
    "split_voices",
    "synthesize_plan",
    "voice_group",
    "voice_lang_code",
]

KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000

#: American (``a``) and British (``b``) English voice packs in Kokoro-82M v1.0
#: (the Hub's ``voices/`` folder, checked 2026-09-11).
ENGLISH_VOICES: tuple[str, ...] = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
    "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)  # fmt: skip

#: ``tts(text, voice) -> (mono audio, sample rate)``.
TtsFn = Callable[[str, str], tuple[np.ndarray, int]]


def voice_group(voice: str) -> str:
    """Accent and gender group of a voice (``"af"``, ``"am"``, ``"bf"``, ``"bm"``)."""
    return voice[:2]


def voice_lang_code(voice: str) -> str:
    """Kokoro ``KPipeline`` language code for an English voice (``"a"`` or ``"b"``)."""
    code = voice[:1]
    if code not in {"a", "b"} or voice[1:3] not in {"f_", "m_"}:
        raise ValueError(f"{voice!r} is not an American or British English Kokoro voice")
    return code


def split_voices(
    voices: Sequence[str] = ENGLISH_VOICES, *, test_fraction: float = 0.25, seed: int = 0
) -> dict[str, str]:
    """Deterministic, group-stratified ``{voice: "train" | "test"}`` split.

    In each group the voices are ordered by a seeded hash and the first
    ``round(len * test_fraction)`` (at least one, when the group has two or more voices)
    go to ``test``.
    """
    groups: dict[str, list[str]] = {}
    for v in sorted(set(voices)):
        voice_lang_code(v)
        groups.setdefault(voice_group(v), []).append(v)
    out: dict[str, str] = {}
    for members in groups.values():
        ordered = sorted(members, key=lambda v: (stable_hash(seed, "kokoro-voice", v), v))
        n_test = max(1, round(len(ordered) * test_fraction)) if len(ordered) >= 2 else 0
        for i, v in enumerate(ordered):
            out[v] = "test" if i < n_test else "train"
    return out


_WS = re.compile(r"\s+")


def clean_prompt(text: str | None, *, min_chars: int = 40, max_chars: int = 240) -> str | None:
    """Normalise a transcript into a TTS prompt, or ``None`` if it is unsuitable.

    Keeps ASCII-renderable text with at least one letter, collapses whitespace and drops
    prompts outside ``[min_chars, max_chars]``.
    """
    if not text:
        return None
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = _WS.sub(" ", t).strip()
    if not (min_chars <= len(t) <= max_chars):
        return None
    if not t.isascii() or not any(c.isalpha() for c in t):
        return None
    return t


def select_prompts(
    texts: Iterable[str | None],
    n: int,
    *,
    seed: int = 0,
    min_chars: int = 40,
    max_chars: int = 240,
) -> list[str]:
    """``n`` distinct cleaned prompts chosen deterministically from ``texts``."""
    pool = sorted({p for p in (clean_prompt(t, min_chars=min_chars, max_chars=max_chars) for t in texts) if p})
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} usable prompts for {n} requested")
    idx = np.random.default_rng(seed).choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(idx.tolist())]


def plan_agent_utterances(
    voice_split: Mapping[str, str], prompts: Sequence[str], *, seed: int = 0
) -> list[dict[str, Any]]:
    """One utterance per prompt, voices assigned round-robin in a seeded order.

    Each prompt is spoken once, so train and test voices never share a text. Returns
    records with ``utt_id``, ``speaker`` (namespaced voice), ``voice_id``, ``lang_code``,
    ``split`` and ``text``.
    """
    voices = sorted(voice_split)
    if not voices:
        raise ValueError("no voices")
    order = np.random.default_rng(seed).permutation(len(prompts))
    plan = []
    for k, i in enumerate(order.tolist()):
        voice = voices[k % len(voices)]
        plan.append(
            {
                "utt_id": f"{KOKORO}:{voice}:{i:05d}",
                "speaker": speaker_key(KOKORO, voice),
                "voice_id": voice,
                "lang_code": voice_lang_code(voice),
                "split": voice_split[voice],
                "text": prompts[i],
            }
        )
    return sorted(plan, key=lambda r: r["utt_id"])


def synthesize_plan(
    plan: Iterable[Mapping[str, Any]],
    tts: TtsFn,
    writer: ShardWriter,
    *,
    trim: bool = True,
    margin_s: float = 0.1,
    min_seconds: float = 0.5,
    progress: Callable[[int], None] | None = None,
) -> dict[str, float]:
    """Synthesise every planned utterance and append it to ``writer`` at 16 kHz.

    Leading and trailing silence is trimmed by the contract VAD rule (``margin_s`` kept);
    utterances shorter than ``min_seconds`` afterwards are skipped. Manifest columns are
    ``voice_id``, ``lang_code``, ``split``, ``text`` and ``source="kokoro"``, and the
    ``group`` is the voice.
    """
    stats = {"rows": 0.0, "seconds": 0.0, "skipped": 0.0}
    min_len = max(C.WINDOW_LENGTH, int(round(min_seconds * C.SAMPLE_RATE)))
    for k, rec in enumerate(plan):
        audio, sr = tts(str(rec["text"]), str(rec["voice_id"]))
        x = resample_to_contract(to_mono(np.asarray(audio, dtype=np.float32)), int(sr))
        if trim:
            start, end = trim_silence(x, margin_s=margin_s)
            x = x[start:end]
        if x.size < min_len:
            stats["skipped"] += 1
            continue
        writer.add(
            x,
            C.SAMPLE_RATE,
            utt_id=str(rec["utt_id"]),
            speaker=str(rec["speaker"]),
            group=str(rec["voice_id"]),
            voice_id=str(rec["voice_id"]),
            lang_code=str(rec["lang_code"]),
            split=str(rec["split"]),
            text=str(rec["text"]),
            source="kokoro",
        )
        stats["rows"] += 1
        stats["seconds"] += x.size / C.SAMPLE_RATE
        if progress is not None:
            progress(k + 1)
    return stats


def kokoro_tts(
    *, device: str | None = None, speed: float = 1.0, repo_id: str = KOKORO_REPO
) -> TtsFn:
    """A :data:`TtsFn` backed by ``kokoro.KPipeline`` (Colab only).

    Needs ``pip install "kokoro>=0.9.2"`` and the ``espeak-ng`` system package. One
    pipeline per language code is built on first use; a prompt's audio chunks are
    concatenated.
    """
    from kokoro import KPipeline

    pipelines: dict[str, Any] = {}

    def tts(text: str, voice: str) -> tuple[np.ndarray, int]:
        lang = voice_lang_code(voice)
        if lang not in pipelines:
            pipelines[lang] = KPipeline(lang_code=lang, repo_id=repo_id, device=device)
        chunks: list[np.ndarray] = []
        for _, _, audio in pipelines[lang](text, voice=voice, speed=speed):
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        wave = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return wave, KOKORO_SAMPLE_RATE

    return tts
