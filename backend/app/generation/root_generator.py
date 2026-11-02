import random
from collections.abc import Mapping, Sequence

from pydantic import Field

from app.generation.phoneme_generator import GenerationError, derive_rng, generate_syllable, render_phonemes
from app.models.common import DomainModel, new_id
from app.models.phonology import Phonology
from app.validation.phonology_validator import validate_phoneme_sequence

DEFAULT_SYLLABLE_COUNT_WEIGHTS: Mapping[int, float] = {1: 0.25, 2: 0.55, 3: 0.20}
# Single phonemes like "a" are legal syllables but make poor roots.
MIN_ROOT_PHONEMES = 2


def generate_root(
    phonology: Phonology, syllable_count: int, rng: random.Random, max_attempts: int = 100
) -> list[str]:
    """Build a root by joining syllables, retrying until it passes validation.

    Individual syllables are always legal, but joining them can create
    word-level problems (e.g. hiatus in "sa" + "a"), so the full validator
    has the final say. This keeps generator and validator in agreement.
    """
    for _ in range(max_attempts):
        phonemes = [p for _ in range(syllable_count) for p in generate_syllable(phonology, rng)]
        if len(phonemes) >= MIN_ROOT_PHONEMES and validate_phoneme_sequence(phonemes, phonology).ok:
            return phonemes
    raise GenerationError(f"could not generate a legal {syllable_count}-syllable root")


class RootProvenance(DomainModel):
    seed: int
    stream: str
    index: int


class RootCandidate(DomainModel):
    id: str = Field(default_factory=new_id)
    phonemes: list[str]
    orthography: str
    provenance: RootProvenance


class ReservedForm(DomainModel):
    phonemes: list[str]
    owner_id: str
    owner_kind: str  # "user_word", "root", "affix", "derived"


class DuplicateFormError(ValueError):
    def __init__(self, phonemes: Sequence[str], existing: ReservedForm):
        self.existing = existing
        super().__init__(
            f"form {''.join(phonemes)!r} is already reserved by {existing.owner_kind} {existing.owner_id}"
        )


class RootRegistry:
    """Pool of candidate roots plus a registry of every form already in use.

    Candidates are not part of the language until `accept` is called with
    the lexeme/morpheme that owns them. Every form (user words, affixes,
    derived words) goes through `reserve` so accidental homophones are caught.
    """

    def __init__(self, phonology: Phonology, seed: int, stream: str = "roots"):
        self.phonology = phonology
        self.seed = seed
        self.stream = stream
        self._rng = derive_rng(seed, stream)
        self._next_index = 0
        self._candidates: dict[str, RootCandidate] = {}
        self._reserved: dict[tuple[str, ...], ReservedForm] = {}

    @property
    def unused_candidates(self) -> list[RootCandidate]:
        return list(self._candidates.values())

    @property
    def reserved_forms(self) -> list[ReservedForm]:
        return list(self._reserved.values())

    def is_reserved(self, phonemes: Sequence[str]) -> bool:
        return tuple(phonemes) in self._reserved

    def reserve(self, phonemes: Sequence[str], owner_id: str, owner_kind: str) -> ReservedForm:
        key = tuple(phonemes)
        existing = self._reserved.get(key)
        if existing is not None and existing.owner_id != owner_id:
            raise DuplicateFormError(phonemes, existing)
        form = ReservedForm(phonemes=list(phonemes), owner_id=owner_id, owner_kind=owner_kind)
        self._reserved[key] = form
        # A reserved form can no longer be handed out as a fresh candidate.
        self._candidates = {cid: c for cid, c in self._candidates.items() if tuple(c.phonemes) != key}
        return form

    def generate_candidates(
        self,
        count: int,
        syllable_count_weights: Mapping[int, float] = DEFAULT_SYLLABLE_COUNT_WEIGHTS,
        max_attempts: int | None = None,
    ) -> list[RootCandidate]:
        max_attempts = max_attempts or count * 50
        taken = set(self._reserved) | {tuple(c.phonemes) for c in self._candidates.values()}
        counts, weights = list(syllable_count_weights), list(syllable_count_weights.values())
        new: list[RootCandidate] = []
        for _ in range(max_attempts):
            if len(new) == count:
                break
            syllables = self._rng.choices(counts, weights=weights)[0]
            phonemes = generate_root(self.phonology, syllables, self._rng)
            if tuple(phonemes) in taken:
                continue
            taken.add(tuple(phonemes))
            candidate = RootCandidate(
                phonemes=phonemes,
                orthography=render_phonemes(phonemes, self.phonology),
                provenance=RootProvenance(seed=self.seed, stream=self.stream, index=self._next_index),
            )
            self._next_index += 1
            self._candidates[candidate.id] = candidate
            new.append(candidate)
        if len(new) < count:
            raise GenerationError(
                f"only {len(new)} of {count} unique roots could be generated; "
                "the phoneme inventory or syllable patterns may be too small"
            )
        return new

    def accept(self, candidate_id: str, owner_id: str) -> RootCandidate:
        candidate = self._candidates[candidate_id]
        self.reserve(candidate.phonemes, owner_id, "root")
        return candidate
