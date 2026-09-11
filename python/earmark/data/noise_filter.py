"""Keeping the excluded sound classes out of noise sets, plus the held-out DEMAND environments.

The plan keeps a fixed list of sound classes out of every noise set; :data:`EXCLUDED_TERMS`
is exactly that list. Two filters apply it:

* **Metadata.** :func:`mentions_excluded` drops a clip whose class label, file name,
  list-file line or annotation names an excluded class as a whole word (names are split
  on punctuation, digits and camelCase; a plural ``s`` is ignored). ESC-50's
  ``airplane``, ``helicopter``, ``engine`` and ``chainsaw`` categories go this way
  (:func:`esc50_keep`).
* **Content.** Many noise files carry no class metadata at all (RIRS_NOISES point-source
  noises are named ``noise-free-sound-NNNN``; DEMAND environments are named by place),
  so :class:`ContentFilter` asks a sound-event tagger instead: a clip is dropped when any
  label that names an excluded class reaches the threshold in any window. The excluded
  labels are found by applying the same whole-word rule to the tagger's own label names
  (:func:`excluded_label_indices`), so the term list stays the single source.
  :class:`AstAudioSetTagger` wraps an AudioSet tagger for the notebooks; tests use a stub.

VoiceBank-DEMAND's test noise comes from DEMAND, so the environments named in VB's
``log_testset.txt`` are held out of training (:data:`VB_HELDOUT_ENVIRONMENTS`,
:func:`vb_test_environments`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import soxr

from earmark import constants as C

__all__ = [
    "AST_AUDIOSET_MODEL",
    "DEFAULT_TAG_THRESHOLD",
    "DEFAULT_TAG_WINDOW_SECONDS",
    "DEMAND_16K_ENVIRONMENTS",
    "ESC50_EXCLUDED_CATEGORIES",
    "EXCLUDED_TERMS",
    "VB_HELDOUT_ENVIRONMENTS",
    "VB_NOISE_LABEL_TO_DEMAND",
    "AstAudioSetTagger",
    "ClipTagger",
    "ContentFilter",
    "TagDecision",
    "demand_training_environments",
    "esc50_keep",
    "excluded_label_indices",
    "filter_records",
    "mentions_excluded",
    "name_tokens",
    "split_windows",
    "vb_test_environments",
]

#: Sound classes kept out of every noise set (whole-word match, plural ``s`` ignored).
EXCLUDED_TERMS: tuple[str, ...] = (
    "aircraft",
    "airplane",
    "helicopter",
    "drone",
    "propeller",
    "engine",
    "chainsaw",
)

#: ESC-50 categories excluded from the held-out test noise.
ESC50_EXCLUDED_CATEGORIES: frozenset[str] = frozenset({"airplane", "helicopter", "engine", "chainsaw"})

#: DEMAND environments with a 16 kHz release on Zenodo (SCAFE exists only at 48 kHz).
DEMAND_16K_ENVIRONMENTS: tuple[str, ...] = (
    "DKITCHEN",
    "DLIVING",
    "DWASHING",
    "NFIELD",
    "NPARK",
    "NRIVER",
    "OHALLWAY",
    "OMEETING",
    "OOFFICE",
    "PCAFETER",
    "PRESTO",
    "PSTATION",
    "SPSQUARE",
    "STRAFFIC",
    "TBUS",
    "TCAR",
    "TMETRO",
)

#: VB ``log_testset.txt`` noise labels -> DEMAND environments to hold out. VB's ``cafe``
#: is SCAFE in the 48 kHz DEMAND release; PCAFETER is held out too, because the label is
#: ambiguous and holding it out is the safe side. That leaves the 12 training
#: environments the plan counts.
VB_NOISE_LABEL_TO_DEMAND: Mapping[str, tuple[str, ...]] = {
    "bus": ("TBUS",),
    "cafe": ("SCAFE", "PCAFETER"),
    "living": ("DLIVING",),
    "office": ("OOFFICE",),
    "psquare": ("SPSQUARE",),
}

#: Every DEMAND environment behind VoiceBank-DEMAND's test noise (held out of training).
VB_HELDOUT_ENVIRONMENTS: frozenset[str] = frozenset(
    env for envs in VB_NOISE_LABEL_TO_DEMAND.values() for env in envs
)

#: A clip is dropped when an excluded label's probability reaches this in any window.
DEFAULT_TAG_THRESHOLD = 0.2
#: Tagging window in seconds (AudioSet taggers are trained on 10 s clips).
DEFAULT_TAG_WINDOW_SECONDS = 10.0
#: AudioSet tagger the notebooks use (Audio Spectrogram Transformer, 527 labels).
AST_AUDIOSET_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_EXCLUDED = frozenset(EXCLUDED_TERMS)

T = TypeVar("T")


# --------------------------------------------------------------------------- metadata


def name_tokens(text: str) -> list[str]:
    """Lower-case word tokens of a label, path or free text (camelCase and digits split)."""
    tokens: list[str] = []
    for part in _SPLIT.split(str(text)):
        if part:
            tokens.extend(t.lower() for t in _CAMEL.split(part) if t)
    return tokens


def _is_excluded(token: str) -> bool:
    if token in _EXCLUDED:
        return True
    return token.endswith("s") and token[:-1] in _EXCLUDED


def mentions_excluded(*texts: Any) -> bool:
    """True when any text (or ``str()`` of a value) contains an excluded term as a word."""
    for text in texts:
        if text is None:
            continue
        if any(_is_excluded(t) for t in name_tokens(str(text))):
            return True
    return False


def filter_records(
    records: Iterable[T], fields: Callable[[T], Iterable[Any]]
) -> tuple[list[T], list[T]]:
    """Split records into (kept, dropped) by :func:`mentions_excluded` on ``fields(record)``."""
    kept: list[T] = []
    dropped: list[T] = []
    for record in records:
        (dropped if mentions_excluded(*fields(record)) else kept).append(record)
    return kept, dropped


def esc50_keep(category: str, filename: str = "") -> bool:
    """Whether an ESC-50 clip may be used as held-out test noise."""
    if category.strip().lower() in ESC50_EXCLUDED_CATEGORIES:
        return False
    return not mentions_excluded(category, filename)


def vb_test_environments(log_text: str) -> frozenset[str]:
    """DEMAND environments behind VB-DEMAND's test noise, parsed from ``log_testset.txt``.

    Lines are ``<utterance> <noise label> <snr>``. Unknown labels raise ``ValueError``
    naming them, so a changed log format fails loudly instead of leaking an environment.
    """
    labels: set[str] = set()
    for line in log_text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            labels.add(parts[1].lower())
    if not labels:
        raise ValueError("no noise labels found in the VB test log")
    unknown = sorted(labels - set(VB_NOISE_LABEL_TO_DEMAND))
    if unknown:
        raise ValueError(f"unknown VB noise labels {unknown}; extend VB_NOISE_LABEL_TO_DEMAND")
    return frozenset(env for label in labels for env in VB_NOISE_LABEL_TO_DEMAND[label])


def demand_training_environments(held_out: Iterable[str]) -> tuple[str, ...]:
    """The 16 kHz DEMAND environments left for training once ``held_out`` is removed."""
    blocked = {h.upper() for h in held_out}
    return tuple(env for env in DEMAND_16K_ENVIRONMENTS if env not in blocked)


# --------------------------------------------------------------------------- content


def excluded_label_indices(labels: Sequence[str]) -> tuple[int, ...]:
    """Indices of a tagger's labels that name an excluded class (the whole-word rule)."""
    return tuple(i for i, name in enumerate(labels) if mentions_excluded(name))


def split_windows(x: np.ndarray, window: int) -> list[np.ndarray]:
    """Non-overlapping windows covering ``x``; the last one is aligned to the end.

    A clip no longer than ``window`` is a single window, so nothing is zero-padded.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    n = int(x.shape[-1])
    if n <= window:
        return [x]
    starts = list(range(0, n - window + 1, window))
    if starts[-1] + window < n:
        starts.append(n - window)
    return [x[s : s + window] for s in starts]


@runtime_checkable
class ClipTagger(Protocol):
    """A sound-event tagger: mono clips at ``sample_rate`` in, per-label probabilities out."""

    labels: Sequence[str]
    sample_rate: int

    def scores(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        """Probabilities ``[len(clips), len(labels)]`` in ``[0, 1]``."""
        ...


@dataclass(frozen=True)
class TagDecision:
    """Outcome of :meth:`ContentFilter.decide`: the verdict and the strongest excluded label."""

    keep: bool
    max_excluded: float
    label: str | None


class ContentFilter:
    """Drop clips that a tagger scores as an excluded class.

    An instance is the ``keep(audio, meta)`` predicate the corpus exporters accept
    (:mod:`.corpora`). Every call is recorded in :attr:`log`; :meth:`summary` condenses it
    for the dataset card.
    """

    def __init__(
        self,
        tagger: ClipTagger,
        *,
        threshold: float = DEFAULT_TAG_THRESHOLD,
        window_seconds: float = DEFAULT_TAG_WINDOW_SECONDS,
        batch_windows: int = 16,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.tagger = tagger
        self.threshold = float(threshold)
        self.window = int(round(window_seconds * tagger.sample_rate))
        self.batch_windows = max(1, int(batch_windows))
        self.excluded = excluded_label_indices(tagger.labels)
        if not self.excluded:
            raise ValueError("the tagger's labels name no excluded class")
        self.log: list[dict[str, Any]] = []

    @property
    def excluded_labels(self) -> tuple[str, ...]:
        """The tagger labels that count as excluded classes."""
        return tuple(self.tagger.labels[i] for i in self.excluded)

    def decide(self, audio: np.ndarray, sample_rate: int = C.SAMPLE_RATE) -> TagDecision:
        """Score every window of a mono clip and keep it only below the threshold."""
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("ContentFilter expects mono audio")
        if x.size == 0:
            return TagDecision(True, 0.0, None)
        if int(sample_rate) != int(self.tagger.sample_rate):
            x = np.asarray(
                soxr.resample(x, int(sample_rate), int(self.tagger.sample_rate)), dtype=np.float32
            )
        windows = split_windows(x, self.window)
        parts: list[np.ndarray] = []
        for i in range(0, len(windows), self.batch_windows):
            s = np.asarray(self.tagger.scores(windows[i : i + self.batch_windows]), dtype=np.float64)
            if s.ndim != 2 or s.shape[1] != len(self.tagger.labels):
                raise ValueError(f"tagger returned scores of shape {s.shape}")
            parts.append(s)
        per_label = np.concatenate(parts, axis=0)[:, list(self.excluded)].max(axis=0)
        j = int(np.argmax(per_label))
        top = float(per_label[j])
        return TagDecision(top < self.threshold, top, str(self.tagger.labels[self.excluded[j]]))

    def __call__(self, audio: np.ndarray, meta: Mapping[str, Any] | None = None) -> bool:
        decision = self.decide(audio)
        entry: dict[str, Any] = dict(meta or {})
        entry.update(keep=decision.keep, max_excluded=decision.max_excluded, label=decision.label)
        self.log.append(entry)
        return decision.keep

    def summary(self) -> dict[str, Any]:
        """Clips seen and dropped, and which labels dropped them."""
        by_label: dict[str, int] = {}
        dropped = 0
        for entry in self.log:
            if not entry["keep"]:
                dropped += 1
                label = str(entry["label"])
                by_label[label] = by_label.get(label, 0) + 1
        return {
            "seen": len(self.log),
            "dropped": dropped,
            "threshold": self.threshold,
            "window_samples": self.window,
            "dropped_by_label": dict(sorted(by_label.items())),
            "excluded_labels": list(self.excluded_labels),
        }


class AstAudioSetTagger:
    """AudioSet tagger (Audio Spectrogram Transformer) for the notebooks.

    Needs ``transformers`` and downloads :data:`AST_AUDIOSET_MODEL` (about 350 MB) from the
    Hub. The unit tests never construct it.
    """

    sample_rate = 16000

    def __init__(
        self,
        model_id: str = AST_AUDIOSET_MODEL,
        *,
        revision: str | None = None,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self._torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.extractor = ASTFeatureExtractor.from_pretrained(model_id, revision=revision)
        model = ASTForAudioClassification.from_pretrained(model_id, revision=revision)
        self.model = model.to(self.device).eval()
        config = self.model.config
        self.labels: tuple[str, ...] = tuple(str(config.id2label[i]) for i in range(config.num_labels))
        self.model_id = model_id

    def scores(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        torch = self._torch
        feats = self.extractor(
            [np.asarray(c, dtype=np.float32) for c in clips],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in feats.items()}).logits
        return torch.sigmoid(logits.float()).cpu().numpy()
