from enum import StrEnum
from typing import NamedTuple

from pydantic import Field, model_validator

from app.models.common import DomainModel, new_id


class PhonemeClass(StrEnum):
    CONSONANT = "C"
    VOWEL = "V"


class Position(StrEnum):
    ONSET = "onset"
    CODA = "coda"
    WORD_INITIAL = "word_initial"
    WORD_FINAL = "word_final"


class CatalogEntry(NamedTuple):
    phoneme_class: PhonemeClass
    ipa: str


# MVP orthography: every supported grapheme is one Latin character mapped to
# exactly one phoneme. A language picks a subset of this catalog.
GRAPHEME_CATALOG: dict[str, CatalogEntry] = {
    **{v: CatalogEntry(PhonemeClass.VOWEL, v) for v in "aeiou"},
    **{
        g: CatalogEntry(PhonemeClass.CONSONANT, ipa)
        for g, ipa in {
            "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "ɡ", "q": "q",
            "m": "m", "n": "n", "f": "f", "v": "v", "s": "s", "z": "z", "h": "h",
            "x": "x", "l": "l", "r": "r", "w": "w", "y": "j", "c": "tʃ", "j": "dʒ",
        }.items()
    },
}

# Syllable shapes the MVP engine understands. No clusters, tone, or length yet.
SUPPORTED_PATTERNS: frozenset[str] = frozenset({"V", "CV", "VC", "CVC"})


class Phoneme(DomainModel):
    # In the MVP a phoneme's id is its grapheme. The id is what lexeme forms
    # store, so it must stay stable when sound changes are added later.
    id: str
    grapheme: str
    ipa: str
    phoneme_class: PhonemeClass
    # Relative sampling weight for root generation ("common" vs "rare" sounds).
    frequency: float = Field(default=1.0, gt=0)

    @classmethod
    def from_grapheme(cls, grapheme: str, frequency: float = 1.0) -> "Phoneme":
        entry = GRAPHEME_CATALOG[grapheme]
        return cls(
            id=grapheme,
            grapheme=grapheme,
            ipa=entry.ipa,
            phoneme_class=entry.phoneme_class,
            frequency=frequency,
        )


class PositionRestriction(DomainModel):
    """Forbids a phoneme from appearing in certain positions, e.g. no /h/ in codas."""

    phoneme_id: str
    forbidden_positions: list[Position] = Field(min_length=1)
    reason: str | None = None


class Phonology(DomainModel):
    id: str = Field(default_factory=new_id)
    phonemes: list[Phoneme]
    syllable_patterns: list[str] = Field(min_length=1)
    restrictions: list[PositionRestriction] = Field(default_factory=list)
    # Whether two vowels may be adjacent across a syllable boundary (e.g. "ri.a").
    allow_hiatus: bool = False

    @model_validator(mode="after")
    def _check_consistency(self) -> "Phonology":
        problems: list[str] = []

        ids = [p.id for p in self.phonemes]
        graphemes = [p.grapheme for p in self.phonemes]
        if len(set(ids)) != len(ids):
            problems.append("duplicate phoneme ids")
        if len(set(graphemes)) != len(graphemes):
            problems.append("duplicate graphemes")
        for p in self.phonemes:
            entry = GRAPHEME_CATALOG.get(p.grapheme)
            if entry is None:
                problems.append(f"unsupported grapheme {p.grapheme!r}")
            elif entry.phoneme_class != p.phoneme_class:
                problems.append(f"grapheme {p.grapheme!r} must be class {entry.phoneme_class}")

        unsupported = set(self.syllable_patterns) - SUPPORTED_PATTERNS
        if unsupported:
            problems.append(f"unsupported syllable patterns {sorted(unsupported)}")
        if len(set(self.syllable_patterns)) != len(self.syllable_patterns):
            problems.append("duplicate syllable patterns")

        if not self.vowels:
            problems.append("at least one vowel is required")
        if any("C" in pat for pat in self.syllable_patterns) and not self.consonants:
            problems.append("patterns use consonants but none are defined")

        known = set(ids)
        for r in self.restrictions:
            if r.phoneme_id not in known:
                problems.append(f"restriction references unknown phoneme {r.phoneme_id!r}")
            elif self.phoneme(r.phoneme_id).phoneme_class == PhonemeClass.VOWEL and (
                {Position.ONSET, Position.CODA} & set(r.forbidden_positions)
            ):
                problems.append(f"vowel {r.phoneme_id!r} cannot be restricted in onset/coda")

        # Every consonant slot used by a pattern needs at least one legal filler,
        # otherwise the generator could never realize that pattern.
        if any(p.startswith("C") for p in self.syllable_patterns) and not self.consonants_allowed_in(
            Position.ONSET
        ):
            problems.append("onset patterns exist but every consonant is forbidden in onsets")
        if any(p.endswith("VC") for p in self.syllable_patterns) and not self.consonants_allowed_in(
            Position.CODA
        ):
            problems.append("coda patterns exist but every consonant is forbidden in codas")

        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def consonants(self) -> list[Phoneme]:
        return [p for p in self.phonemes if p.phoneme_class == PhonemeClass.CONSONANT]

    @property
    def vowels(self) -> list[Phoneme]:
        return [p for p in self.phonemes if p.phoneme_class == PhonemeClass.VOWEL]

    def phoneme(self, phoneme_id: str) -> Phoneme:
        for p in self.phonemes:
            if p.id == phoneme_id:
                return p
        raise KeyError(phoneme_id)

    def has_phoneme(self, phoneme_id: str) -> bool:
        return any(p.id == phoneme_id for p in self.phonemes)

    def is_forbidden(self, phoneme_id: str, position: Position) -> bool:
        return any(
            r.phoneme_id == phoneme_id and position in r.forbidden_positions
            for r in self.restrictions
        )

    def consonants_allowed_in(self, position: Position) -> list[Phoneme]:
        return [c for c in self.consonants if not self.is_forbidden(c.id, position)]
