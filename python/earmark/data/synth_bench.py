"""Earmark-Synth: the seeded, sparse-design dev/test benchmark generator and its manifest.

Suite A (plan): LibriTTS-R test.clean targets (dev.clean for dev) with the clean restored
references; interferers from test.other (dev.other) speakers, held-out Kokoro voices and
TV speech over held-out MUSAN music; noise from the held-out VB-test DEMAND environments
plus filtered ESC-50; real RIRs; enrolment from a different chapter of the target
speaker; 8 s mixtures.

Sparse design. :data:`DEFAULT_FRACTIONS` puts 40% of the mixtures at the hypothesis
conditions (0 dB SIR, interferer only, long target-absent stretches) and spreads the rest
over SIR {-5, 0, 5, 10} x SNR {0, 5, 10, 20} plus the named conditions: the agent's own
TTS voice through laptop speakers, an Opus 16 kb/s round trip, ESC-50-only noise with a
real RIR, and the do-no-harm clean control. The final size comes from the dev pilot and
is recorded in PREREG; :class:`SuiteDesign` holds it.

A mixture is fully described by its :class:`MixtureSpec` (source utterance ids and
offsets, levels, RIR ids, loudspeaker parameters, codec). :func:`render_mixture` rebuilds
its audio deterministically from the spec and the source corpora, so the manifest, not the
audio, is what gets stored, shared and regenerated (for example in CI for rows that only
run on Linux). Mixing follows the training mixer's definitions (:mod:`.mixer`): active-level
SNR and SIR, a direct-path-plus-50 ms reference, and contract VAD labels.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from earmark import constants as C
from earmark.data.embeddings import opus_roundtrip
from earmark.data.labels import (
    active_power,
    apply_hangover,
    db_to_power_ratio,
    frame_energy,
    num_frames,
)
from earmark.data.mixer import (
    INTERFERER_AGENT,
    INTERFERER_HUMAN,
    INTERFERER_NONE,
    INTERFERER_TV,
    MixerConfig,
    level_gain,
    loudspeaker_response,
    next_fft_size,
    rir_windows,
    segment_peak_frames,
    soft_clip,
    synthetic_rir,
)
from earmark.data.shards import POOL_ENROL, POOL_TARGET, ShardedCorpus
from earmark.data.splits import SplitLeakError, check_speaker_disjoint

__all__ = [
    "COND_AGENT",
    "COND_CLEAN",
    "COND_ESC50",
    "COND_GRID",
    "COND_INTERFERER_ONLY",
    "COND_LONG_ABSENCE",
    "COND_OPUS",
    "COND_SIR0",
    "COND_SVARAH",
    "CONDITIONS",
    "DEFAULT_FRACTIONS",
    "HYPOTHESIS_CONDITIONS",
    "BenchPools",
    "MixtureSpec",
    "SourceRef",
    "SuiteDesign",
    "allocate_conditions",
    "check_manifest_disjoint",
    "design_suite",
    "load_bench_pools",
    "read_manifest",
    "render_mixture",
    "summarize",
    "write_manifest",
]

COND_SIR0 = "hyp_sir0"
COND_INTERFERER_ONLY = "hyp_interferer_only"
COND_LONG_ABSENCE = "hyp_long_absence"
COND_GRID = "grid"
COND_AGENT = "agent_tts"
COND_OPUS = "opus16k"
COND_ESC50 = "esc50_real_rir"
COND_CLEAN = "clean_control"
#: Indian-accent slice: like ``grid`` but the targets have no studio-clean reference, so
#: it is scored with WER and barge-in metrics only.
COND_SVARAH = "svarah"
CONDITIONS: tuple[str, ...] = (
    COND_SIR0, COND_INTERFERER_ONLY, COND_LONG_ABSENCE, COND_GRID, COND_AGENT,
    COND_OPUS, COND_ESC50, COND_CLEAN, COND_SVARAH,
)  # fmt: skip
HYPOTHESIS_CONDITIONS: tuple[str, ...] = (COND_SIR0, COND_INTERFERER_ONLY, COND_LONG_ABSENCE)
DEFAULT_FRACTIONS: Mapping[str, float] = {
    COND_SIR0: 0.15,
    COND_INTERFERER_ONLY: 0.125,
    COND_LONG_ABSENCE: 0.125,
    COND_GRID: 0.35,
    COND_AGENT: 0.08,
    COND_OPUS: 0.07,
    COND_ESC50: 0.05,
    COND_CLEAN: 0.05,
}
_CORPORA = ("targets", "interferers", "noise", "rirs", "agent", "music")


@dataclass(frozen=True)
class SuiteDesign:
    """Size and condition mix of one Earmark-Synth split."""

    n_mixtures: int = 1500
    seconds: float = 8.0
    fractions: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_FRACTIONS))
    sir_grid: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0)
    snr_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    target_level_db: tuple[float, float] = (-30.0, -20.0)
    p_reverb: float = 0.5
    p_tv: float = 1 / 3
    p_loudspeaker_other: float = 0.5
    music_bed_db: tuple[float, float] = (-20.0, -5.0)
    enrol_seconds: float = 10.0
    max_segments: int = 3
    long_absence_speech_s: tuple[float, float] = (1.0, 2.5)
    min_segment_seconds: float = 0.5

    def __post_init__(self) -> None:
        unknown = set(self.fractions) - set(CONDITIONS)
        if unknown:
            raise ValueError(f"unknown conditions {sorted(unknown)}")
        if any(v < 0 for v in self.fractions.values()) or sum(self.fractions.values()) <= 0:
            raise ValueError("fractions must be non-negative and not all zero")
        if self.n_mixtures < 1:
            raise ValueError("n_mixtures must be positive")


@dataclass(frozen=True)
class SourceRef:
    """A piece of one source placed in the mixture.

    Either ``[src, src + length)`` of the utterance goes to ``[dst, dst + length)``, or,
    when ``tile_start`` is set, the whole mixture is filled by looping the source from
    that sample (noise, music).
    """

    corpus: str
    utt_id: str
    src: int = 0
    dst: int = 0
    length: int = 0
    tile_start: int | None = None


@dataclass(frozen=True)
class MixtureSpec:
    """Everything needed to rebuild one benchmark mixture."""

    mixture_id: str
    split: str
    condition: str
    seconds: float
    seed: int
    target_speaker: str
    target_present: bool
    target_segments: tuple[SourceRef, ...]
    target_group: str
    enrol_group: str
    enrol_utts: tuple[str, ...]
    target_level_db: float
    target_rir: str | None = None
    interferer_kind: int = INTERFERER_NONE
    interferer: tuple[SourceRef, ...] = ()
    interferer_speaker: str | None = None
    sir_db: float | None = None
    music: SourceRef | None = None
    music_db: float | None = None
    loudspeaker: Mapping[str, Any] | None = None
    interferer_rir: str | None = None
    noise: SourceRef | None = None
    snr_db: float | None = None
    codec: str | None = None
    reference_clean: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> MixtureSpec:
        d = json.loads(text)
        d["target_segments"] = tuple(SourceRef(**s) for s in d["target_segments"])
        d["interferer"] = tuple(SourceRef(**s) for s in d["interferer"])
        d["enrol_utts"] = tuple(d["enrol_utts"])
        for key in ("music", "noise"):
            if d.get(key) is not None:
                d[key] = SourceRef(**d[key])
        return cls(**d)


@dataclass
class BenchPools:
    """Source corpora of one split. ``targets`` needs a ``pool`` column; ``noise`` may
    carry a ``source`` column (``"esc50"`` rows feed only the ESC-50 condition)."""

    targets: ShardedCorpus
    interferers: ShardedCorpus
    noise: ShardedCorpus
    rirs: ShardedCorpus | None = None
    agent: ShardedCorpus | None = None
    music: ShardedCorpus | None = None
    _rows: dict[str, dict[str, int]] = field(default_factory=dict, repr=False)

    def corpus(self, name: str) -> ShardedCorpus:
        if name not in _CORPORA:
            raise KeyError(name)
        corpus = getattr(self, name)
        if corpus is None:
            raise KeyError(f"no {name} corpus in these pools")
        return corpus

    def row(self, name: str, utt_id: str) -> int:
        if name not in self._rows:
            ids = self.corpus(name).column("utt_id").astype(str)
            self._rows[name] = {u: i for i, u in enumerate(ids)}
        return self._rows[name][utt_id]


def _open(paths: str | Path | Sequence[str | Path] | None) -> ShardedCorpus | None:
    if paths is None:
        return None
    items = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not items:
        return None
    corpora = [ShardedCorpus(p) for p in items]
    return corpora[0] if len(corpora) == 1 else ShardedCorpus.concat(corpora)


def _only_splits(corpus: ShardedCorpus | None, splits: Iterable[str] | None) -> ShardedCorpus | None:
    if corpus is None or splits is None or not corpus.has_column("split"):
        return corpus
    return corpus.where("split", set(splits))


def load_bench_pools(
    *,
    targets: str | Path | Sequence[str | Path],
    interferers: str | Path | Sequence[str | Path],
    noise: str | Path | Sequence[str | Path],
    rirs: str | Path | Sequence[str | Path] | None = None,
    agent: str | Path | Sequence[str | Path] | None = None,
    music: str | Path | Sequence[str | Path] | None = None,
    agent_splits: Iterable[str] | None = ("test",),
    music_splits: Iterable[str] | None = ("heldout",),
) -> BenchPools:
    """Open prepared dataset folders as the pools of one Earmark-Synth split.

    For example, test: ``devtest_16k/test_clean`` targets, ``devtest_16k/test_other``
    interferers, ``demand_eval`` plus ESC-50 noise, ``rir_real_eval`` RIRs, the Kokoro
    dataset and ``musan_music_16k/heldout``. Agent and music rows are kept only for the
    held-out ``agent_splits`` (test voices) and ``music_splits`` (held-out artists);
    ``None`` keeps every row.
    """
    t, i, n = _open(targets), _open(interferers), _open(noise)
    if t is None or i is None or n is None:
        raise ValueError("targets, interferers and noise are required")
    return BenchPools(
        targets=t,
        interferers=i,
        noise=n,
        rirs=_open(rirs),
        agent=_only_splits(_open(agent), agent_splits),
        music=_only_splits(_open(music), music_splits),
    )


def allocate_conditions(n: int, fractions: Mapping[str, float]) -> list[str]:
    """``n`` condition labels in proportion to ``fractions`` (largest remainder)."""
    names = sorted(c for c, v in fractions.items() if v > 0)
    total = sum(fractions[c] for c in names)
    raw = {c: n * fractions[c] / total for c in names}
    counts = {c: int(math.floor(raw[c])) for c in names}
    for c in sorted(names, key=lambda c: (-(raw[c] - counts[c]), c))[: n - sum(counts.values())]:
        counts[c] += 1
    return [c for c in names for _ in range(counts[c])]


class _Designer:
    """Draws one :class:`MixtureSpec` at a time from a split's pools."""

    def __init__(
        self, pools: BenchPools, design: SuiteDesign, chain: MixerConfig, split: str, seed: int
    ) -> None:
        self.pools, self.design, self.chain, self.split, self.seed = pools, design, chain, split, seed
        self.sr = C.SAMPLE_RATE
        self.t = int(round(design.seconds * self.sr))
        self.min_len = int(round(design.min_segment_seconds * self.sr))
        tg = pools.targets
        if not tg.has_column("pool"):
            raise ValueError("target corpus needs a pool column (shards.add_pools)")
        spk = tg.speaker.astype(str)
        pool = tg.column("pool").astype(str)
        self.t_ids = tg.column("utt_id").astype(str)
        self.t_group = tg.group.astype(str)
        self.t_len = tg.num_samples
        self.enrol_rows: dict[str, np.ndarray] = {}
        self.target_rows: dict[str, np.ndarray] = {}
        for s in sorted(set(spk.tolist())):
            e = np.flatnonzero((spk == s) & (pool == POOL_ENROL))
            r = np.flatnonzero((spk == s) & (pool == POOL_TARGET) & (self.t_len >= self.min_len))
            if e.size and r.size:
                self.enrol_rows[s], self.target_rows[s] = e, r
        self.speakers = sorted(self.target_rows)
        if not self.speakers:
            raise ValueError("no target speaker has both enrolment and target utterances")
        it = pools.interferers
        self.i_spk = it.speaker.astype(str)
        self.i_rows = np.flatnonzero(it.num_samples >= self.min_len)
        nz = pools.noise
        src = nz.column("source").astype(str) if nz.has_column("source") else np.full(len(nz), "demand")
        self.esc_rows = np.flatnonzero(src == "esc50")
        self.demand_rows = np.flatnonzero(src != "esc50")
        self.rir_rows = np.arange(len(pools.rirs)) if pools.rirs is not None else np.zeros(0, np.int64)
        self.agent_rows = (
            np.flatnonzero(pools.agent.num_samples >= self.min_len) if pools.agent is not None else np.zeros(0, np.int64)
        )
        self.music_rows = np.arange(len(pools.music)) if pools.music is not None else np.zeros(0, np.int64)
        self.cells = [(sir, snr) for sir in design.sir_grid for snr in design.snr_grid]

    # -- helpers ----------------------------------------------------------------------

    def _ref(self, name: str, row: int, src: int, dst: int, length: int) -> SourceRef:
        uid = str(self.pools.corpus(name).column("utt_id")[row])
        return SourceRef(name, uid, int(src), int(dst), int(length))

    def _tiled(self, name: str, rows: np.ndarray, rng: np.random.Generator) -> SourceRef:
        corpus = self.pools.corpus(name)
        row = int(rows[rng.integers(len(rows))])
        n = int(corpus.num_samples[row])
        start = int(rng.integers(max(1, n - self.t + 1))) if n >= self.t else int(rng.integers(n))
        return SourceRef(name, str(corpus.column("utt_id")[row]), tile_start=start)

    def _single(self, name: str, row: int, rng: np.random.Generator, *, dst_min: int = 0) -> SourceRef:
        n = int(self.pools.corpus(name).num_samples[row])
        room = self.t - dst_min
        if n >= room:
            return self._ref(name, row, int(rng.integers(n - room + 1)), dst_min, room)
        return self._ref(name, row, 0, dst_min + int(rng.integers(room - n + 1)), n)

    def _layout(self, name: str, rows: np.ndarray, rng: np.random.Generator) -> tuple[SourceRef, ...]:
        """1-``max_segments`` utterances with a lead-in and pauses, filling the mixture."""
        corpus = self.pools.corpus(name)
        cursor = int(rng.uniform(0.0, 1.0) * self.sr)
        refs: list[SourceRef] = []
        while len(refs) < self.design.max_segments and self.t - cursor >= self.min_len:
            row = int(rows[rng.integers(len(rows))])
            n = int(corpus.num_samples[row])
            length = min(n, self.t - cursor)
            refs.append(self._ref(name, row, int(rng.integers(n - length + 1)), cursor, length))
            cursor += length + int(rng.uniform(0.3, 1.5) * self.sr)
        return tuple(refs)

    def _loudspeaker(self, rng: np.random.Generator) -> dict[str, Any]:
        c = self.chain
        return {
            "highpass_hz": float(rng.uniform(*c.highpass_hz)),
            "lowpass_hz": float(rng.uniform(*c.lowpass_hz)),
            "eq_centre_hz": np.exp2(rng.uniform(np.log2(150.0), np.log2(6000.0), c.eq_bumps)).tolist(),
            "eq_gain_db": rng.uniform(-c.eq_db, c.eq_db, c.eq_bumps).tolist(),
            "eq_width_oct": rng.uniform(*c.eq_octaves, c.eq_bumps).tolist(),
            "drive": float(rng.uniform(*c.drive)),
            "small_room_rt60_s": float(rng.uniform(*c.small_room_rt60_s)),
            "small_room_seed": int(rng.integers(2**31)),
        }

    def _rir(self, rng: np.random.Generator, p: float) -> str | None:
        if not self.rir_rows.size or rng.random() >= p:
            return None
        row = int(self.rir_rows[rng.integers(len(self.rir_rows))])
        return str(self.pools.corpus("rirs").column("utt_id")[row])

    # -- one mixture ------------------------------------------------------------------

    def one(self, i: int, cond: str, k: int, rng: np.random.Generator) -> MixtureSpec:
        d = self.design
        spk = self.speakers[int(rng.integers(len(self.speakers)))]
        enrol = rng.permutation(self.enrol_rows[spk])
        acc, enrol_utts = 0.0, []
        for r in enrol:
            enrol_utts.append(str(self.t_ids[r]))
            acc += self.t_len[r] / self.sr
            if acc >= d.enrol_seconds:
                break
        enrol_group = str(self.t_group[enrol[0]])
        present = cond != COND_INTERFERER_ONLY
        segments: tuple[SourceRef, ...] = ()
        if cond == COND_LONG_ABSENCE:
            row = int(self.target_rows[spk][rng.integers(len(self.target_rows[spk]))])
            n = int(self.t_len[row])
            length = min(n, int(rng.uniform(*d.long_absence_speech_s) * self.sr))
            dst = int(rng.uniform(0.0, 0.5) * self.sr)
            segments = (self._ref("targets", row, int(rng.integers(n - length + 1)), dst, length),)
        elif present:
            segments = self._layout("targets", self.target_rows[spk], rng)
        target_group = str(self.t_group[self.pools.row("targets", segments[0].utt_id)]) if segments else ""
        level = float(rng.uniform(*d.target_level_db))
        snr: float | None = float(rng.choice(d.snr_grid))
        sir: float | None = None
        kind = INTERFERER_NONE
        codec = None
        target_rir = self._rir(rng, d.p_reverb)
        noise_rows = self.demand_rows

        if cond == COND_CLEAN:
            snr, target_rir = None, None
        elif cond == COND_SIR0:
            kind, sir = INTERFERER_HUMAN, 0.0
        elif cond in (COND_INTERFERER_ONLY, COND_LONG_ABSENCE):
            kind = INTERFERER_TV if (cond == COND_INTERFERER_ONLY and rng.random() < d.p_tv) else INTERFERER_HUMAN
            sir = float(rng.choice(d.sir_grid))
        elif cond in (COND_GRID, COND_SVARAH, COND_OPUS):
            sir, snr = self.cells[k % len(self.cells)] if cond == COND_GRID else self.cells[int(rng.integers(len(self.cells)))]
            kind = INTERFERER_TV if rng.random() < d.p_tv else INTERFERER_HUMAN
            codec = "opus16k" if cond == COND_OPUS else None
        elif cond == COND_AGENT:
            if not self.agent_rows.size:
                raise ValueError("agent_tts condition needs held-out agent voices")
            kind, sir = INTERFERER_AGENT, float(rng.choice(d.sir_grid))
        elif cond == COND_ESC50:
            if not self.esc_rows.size:
                raise ValueError("esc50_real_rir condition needs ESC-50 noise rows")
            noise_rows = self.esc_rows
            target_rir = self._rir(rng, 1.0)
            if rng.random() < 0.5:
                kind, sir = INTERFERER_HUMAN, float(rng.choice(d.sir_grid))

        interferer: tuple[SourceRef, ...] = ()
        i_speaker = music = music_db = loud = i_rir = None
        if kind != INTERFERER_NONE:
            if kind == INTERFERER_AGENT:
                row = int(self.agent_rows[rng.integers(len(self.agent_rows))])
                interferer = (self._single("agent", row, rng),)
                i_speaker = str(self.pools.corpus("agent").speaker[row])
            else:
                rows = self.i_rows[self.i_spk[self.i_rows] != spk]
                if cond == COND_INTERFERER_ONLY:
                    interferer = self._layout("interferers", rows, rng)
                else:
                    start = 0
                    if cond == COND_LONG_ABSENCE and segments:
                        start = min(self.t - self.min_len, segments[0].dst + segments[0].length)
                    interferer = (self._single("interferers", int(rows[rng.integers(len(rows))]), rng, dst_min=start),)
                i_speaker = str(self.pools.interferers.speaker[self.pools.row("interferers", interferer[0].utt_id)])
            if kind == INTERFERER_TV and self.music_rows.size:
                music = self._tiled("music", self.music_rows, rng)
                music_db = float(rng.uniform(*d.music_bed_db))
            if kind == INTERFERER_AGENT or rng.random() < d.p_loudspeaker_other:
                loud = self._loudspeaker(rng)
            else:
                i_rir = self._rir(rng, d.p_reverb)
        noise = None if snr is None or not noise_rows.size else self._tiled("noise", noise_rows, rng)
        if noise is None:
            snr = None
        return MixtureSpec(
            mixture_id=f"{self.split}-{i:05d}",
            split=self.split,
            condition=cond,
            seconds=float(d.seconds),
            seed=int(self.seed),
            target_speaker=spk,
            target_present=present,
            target_segments=segments,
            target_group=target_group,
            enrol_group=enrol_group,
            enrol_utts=tuple(enrol_utts),
            target_level_db=level,
            target_rir=target_rir,
            interferer_kind=int(kind),
            interferer=interferer,
            interferer_speaker=i_speaker,
            sir_db=None if sir is None else float(sir),
            music=music,
            music_db=music_db,
            loudspeaker=loud,
            interferer_rir=i_rir,
            noise=noise,
            snr_db=None if snr is None else float(snr),
            codec=codec,
            reference_clean=cond != COND_SVARAH,
        )


def design_suite(
    pools: BenchPools,
    *,
    split: str,
    design: SuiteDesign = SuiteDesign(),
    seed: int = 0,
    chain: MixerConfig = MixerConfig(),
    train_speakers: Iterable[str] | None = None,
) -> list[MixtureSpec]:
    """Draw the specs of one split. Mixture ``i`` depends only on ``(seed, i)`` and its
    condition, so the same seed gives the same manifest.

    ``train_speakers`` (namespaced) makes the call fail if any target, interferer or agent
    speaker also appears in training.
    """
    designer = _Designer(pools, design, chain, split, seed)
    if train_speakers is not None:
        held = {
            "benchmark targets": set(designer.speakers),
            "benchmark interferers": set(designer.i_spk.tolist()),
        }
        if pools.agent is not None:
            held["benchmark agent voices"] = set(pools.agent.speaker.astype(str).tolist())
        check_speaker_disjoint(train_speakers, held)
    conditions = allocate_conditions(design.n_mixtures, design.fractions)
    order = np.random.default_rng([seed, 1]).permutation(len(conditions))
    counters: Counter[str] = Counter()
    specs = []
    for i, j in enumerate(order.tolist()):
        cond = conditions[j]
        specs.append(designer.one(i, cond, counters[cond], np.random.default_rng([seed, 2, i])))
        counters[cond] += 1
    return specs


def check_manifest_disjoint(specs: Sequence[MixtureSpec], train_speakers: Iterable[str]) -> None:
    """Raise :class:`SplitLeakError` if a manifest uses any training speaker."""
    used = {s.target_speaker for s in specs} | {s.interferer_speaker for s in specs if s.interferer_speaker}
    check_speaker_disjoint(train_speakers, {"benchmark manifest": used})
    for s in specs:
        if s.target_present and s.enrol_group == s.target_group:
            raise SplitLeakError(f"{s.mixture_id}: enrolment and target share group {s.target_group}")


def summarize(specs: Sequence[MixtureSpec]) -> dict[str, Any]:
    """Counts per condition and the share at the hypothesis conditions."""
    counts = Counter(s.condition for s in specs)
    hyp = sum(counts[c] for c in HYPOTHESIS_CONDITIONS)
    return {
        "n": len(specs),
        "per_condition": dict(sorted(counts.items())),
        "hypothesis_fraction": hyp / len(specs) if specs else 0.0,
        "speakers": len({s.target_speaker for s in specs}),
    }


def write_manifest(specs: Sequence[MixtureSpec], path: str | Path) -> Path:
    """Parquet manifest: flat columns for filtering plus the full ``spec_json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {
        "mixture_id": [s.mixture_id for s in specs],
        "split": [s.split for s in specs],
        "condition": [s.condition for s in specs],
        "seconds": [s.seconds for s in specs],
        "target_speaker": [s.target_speaker for s in specs],
        "target_present": [s.target_present for s in specs],
        "target_group": [s.target_group for s in specs],
        "enrol_group": [s.enrol_group for s in specs],
        "interferer_kind": [s.interferer_kind for s in specs],
        "interferer_speaker": [s.interferer_speaker for s in specs],
        "sir_db": [s.sir_db for s in specs],
        "snr_db": [s.snr_db for s in specs],
        "reverberant": [s.target_rir is not None for s in specs],
        "loudspeaker": [s.loudspeaker is not None for s in specs],
        "codec": [s.codec for s in specs],
        "reference_clean": [s.reference_clean for s in specs],
        "contract_hash": [C.CONTRACT_HASH] * len(specs),
        "spec_json": [s.to_json() for s in specs],
    }
    table = pa.table(rows)
    pq.write_table(table, path)
    return path


def read_manifest(path: str | Path) -> list[MixtureSpec]:
    """Specs back from :func:`write_manifest`."""
    table = pq.read_table(Path(path), columns=["spec_json"])
    return [MixtureSpec.from_json(t) for t in table.column("spec_json").to_pylist()]


# --------------------------------------------------------------------------- rendering


def _gather(refs: Iterable[SourceRef | None], pools: BenchPools, t: int) -> tuple[np.ndarray, list[tuple[float, int, int]]]:
    out = np.zeros(t, dtype=np.float64)
    placed: list[tuple[float, int, int]] = []
    for ref in refs:
        if ref is None:
            continue
        corpus = pools.corpus(ref.corpus)
        row = pools.row(ref.corpus, ref.utt_id)
        if ref.tile_start is not None:
            src = corpus.audio(row).astype(np.float64)
            out += src[(ref.tile_start + np.arange(t)) % src.size]
        else:
            out[ref.dst : ref.dst + ref.length] += corpus.audio(row, ref.src, ref.length)
            placed.append((float(corpus.peak_energy[row]), ref.dst, ref.length))
    return out, placed


def _rir(pools: BenchPools, utt_id: str) -> torch.Tensor:
    h = pools.corpus("rirs").audio(pools.row("rirs", utt_id)).astype(np.float64)
    peak = np.max(np.abs(h)) if h.size else 0.0
    return torch.from_numpy(h / peak if peak > 0 else h)[None]


def _through_room(x: torch.Tensor, h: torch.Tensor | None, taps: int) -> tuple[torch.Tensor, ...]:
    """(full, early, direct) versions of ``x [1, T]`` through RIR ``h [1, L]`` (or dry)."""
    if h is None:
        return x, x, x
    peak = torch.argmax(h.abs(), dim=-1)
    direct, early = rir_windows(h, peak, direct_samples=int(round(2.5 * C.SAMPLE_RATE / 1000)), early_samples=taps)
    n = next_fft_size(x.shape[-1] + h.shape[-1] - 1)
    stack = torch.stack([h, early, direct], 1)
    conv = torch.fft.irfft(torch.fft.rfft(x, n=n).unsqueeze(1) * torch.fft.rfft(stack, n=n), n=n)[..., : x.shape[-1]]
    return conv[:, 0], conv[:, 1], conv[:, 2]


def render_mixture(
    spec: MixtureSpec,
    pools: BenchPools,
    *,
    apply_codec: bool = True,
    max_peak: float = 0.99,
    early_ms: float = 50.0,
) -> dict[str, Any]:
    """Rebuild one mixture from its spec (float64 on the CPU, returned as float32).

    Returns ``mixture``, ``reference`` (target direct path plus the first 50 ms; zeros when
    absent), ``interferer`` and ``noise`` as mixed, ``vad`` (bool ``[F]``, contract rule),
    ``target_present`` and ``output_gain``. With ``apply_codec`` an ``opus16k`` mixture is
    passed through Opus at 16 kb/s (needs ffmpeg with libopus).
    """
    t = int(round(spec.seconds * C.SAMPLE_RATE))
    f = num_frames(t)
    early = int(round(early_ms * C.SAMPLE_RATE / 1000))
    f64 = torch.float64
    dry_np, placed = _gather(spec.target_segments, pools, t)
    dry = torch.from_numpy(dry_np)[None]
    h_t = _rir(pools, spec.target_rir) if spec.target_rir else None
    t_rev, t_early, t_direct = _through_room(dry, h_t, early)
    delay = int(torch.argmax(h_t.abs()).item()) if h_t is not None else 0

    s_np, _ = _gather(spec.interferer, pools, t)
    s = torch.from_numpy(s_np)[None]
    u = s
    if spec.music is not None and spec.music_db is not None:
        m = torch.from_numpy(_gather([spec.music], pools, t)[0])[None]
        g_m = level_gain(m.square().mean(-1), 10 * torch.log10(active_power(s).clamp_min(1e-20)) + spec.music_db)
        u = s + g_m[:, None] * m
    h_i: torch.Tensor | None = None
    if spec.loudspeaker is not None:
        ls = spec.loudspeaker
        n = next_fft_size(2 * t)
        freqs = torch.fft.rfftfreq(n, d=1.0 / C.SAMPLE_RATE).to(f64)
        resp = loudspeaker_response(
            freqs,
            torch.tensor([ls["highpass_hz"]], dtype=f64),
            torch.tensor([ls["lowpass_hz"]], dtype=f64),
            torch.tensor([ls["eq_centre_hz"]], dtype=f64),
            torch.tensor([ls["eq_gain_db"]], dtype=f64),
            torch.tensor([ls["eq_width_oct"]], dtype=f64),
        )
        u = soft_clip(torch.fft.irfft(torch.fft.rfft(u, n=n) * resp, n=n)[..., :t], torch.tensor([ls["drive"]], dtype=f64))
        s = torch.fft.irfft(torch.fft.rfft(s, n=n) * resp, n=n)[..., :t]
        room_rng = np.random.default_rng(int(ls["small_room_seed"]))
        taps = int(round(MixerConfig().rir_max_seconds * C.SAMPLE_RATE))
        h_i = torch.from_numpy(synthetic_rir(room_rng, float(ls["small_room_rt60_s"]), taps).astype(np.float64))[None]
    elif spec.interferer_rir:
        h_i = _rir(pools, spec.interferer_rir)
    i_out = _through_room(u, h_i, early)[0]

    noise = torch.from_numpy(_gather([spec.noise], pools, t)[0])[None]
    level = torch.tensor([spec.target_level_db], dtype=f64)
    zero = torch.zeros(1, dtype=f64)
    g_t = level_gain(active_power(t_rev), level) if spec.target_present else zero
    g_n = level_gain(noise.square().mean(-1), level - spec.snr_db) if spec.snr_db is not None else zero
    g_i = level_gain(active_power(i_out), level - spec.sir_db) if spec.sir_db is not None and spec.interferer else zero
    target_mix, interferer_mix, noise_mix = g_t[:, None] * t_rev, g_i[:, None] * i_out, g_n[:, None] * noise
    mixture = target_mix + interferer_mix + noise_mix
    peak = float(mixture.abs().max())
    gain = max_peak / peak if peak > max_peak else 1.0
    reference = gain * g_t[:, None] * t_early
    direct = gain * g_t[:, None] * t_direct
    vad = torch.zeros(1, f, dtype=torch.bool)
    if spec.target_present and placed:
        rho = t_direct.square().sum() / dry.square().sum().clamp_min(1e-20)
        peaks = torch.from_numpy(segment_peak_frames(placed, f, delay=delay))[None]
        thr = peaks * rho * (gain * g_t) ** 2 * db_to_power_ratio(C.VAD_THRESHOLD_DB)
        vad = apply_hangover(frame_energy(direct) > thr)
    mix = (gain * mixture)[0].numpy().astype(np.float32)
    if apply_codec and spec.codec == "opus16k":
        mix = opus_roundtrip(mix, bitrate=16000)
    return {
        "mixture": mix,
        "reference": reference[0].numpy().astype(np.float32),
        "interferer": (gain * interferer_mix)[0].numpy().astype(np.float32),
        "noise": (gain * noise_mix)[0].numpy().astype(np.float32),
        "vad": vad[0].numpy(),
        "target_present": bool(spec.target_present),
        "output_gain": float(gain),
    }
