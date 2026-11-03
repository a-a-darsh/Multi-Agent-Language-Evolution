from enum import StrEnum

from pydantic import Field, model_validator

from app.models.common import DomainModel, new_id
from app.models.sounds import SOUNDS, PhonemeClass, Sound, nfc, sound, sound_for_grapheme

__all__ = ["MAX_CLUSTER", "Phoneme", "PhonemeClass", "Phonology", "Position", "PositionRestriction",
           "SIMPLE_PATTERNS", "SUPPORTED_PATTERNS", "pattern_order"]


class Position(StrEnum):
    ONSET = "onset"
    CODA = "coda"
    WORD_INITIAL = "word_initial"
    WORD_FINAL = "word_final"


# Syllable shapes the engine understands: up to three consonants on each side of the vowel
# (CCCVCCC, as in English "strengths"). Clusters must also obey sonority (see phonology_validator).
MAX_CLUSTER = 3
SUPPORTED_PATTERNS: frozenset[str] = frozenset(
    "C" * onset + "V" + "C" * coda for onset in range(MAX_CLUSTER + 1) for coda in range(MAX_CLUSTER + 1))
# The shapes the founding-language design agent chooses from. Clusters reach a founding
# language only through the user's own words; sound changes can create them later.
SIMPLE_PATTERNS: tuple[str, ...] = ("V", "CV", "VC", "CVC")


def pattern_order(pattern: str) -> tuple[int, int]:
    """Canonical order: by onset size, then coda size (V, VC, CV, CVC, CCV...)."""
    return pattern.index("V"), len(pattern) - pattern.index("V") - 1


class Phoneme(DomainModel):
    """One sound of a language, and how this language spells it.

    `id` is the IPA symbol (see `app.models.sounds`). Word forms store ids, so
    a sound change rewrites ids while spelling stays a separate choice: one
    language can write /ʃ/ as "sh" and another as "š".
    """

    id: str
    grapheme: str
    # Always equal to `id`; kept as its own field so stored and API data stay explicit.
    ipa: str
    phoneme_class: PhonemeClass
    # Relative sampling weight for root generation ("common" vs "rare" sounds).
    frequency: float = Field(default=1.0, gt=0)

    @classmethod
    def from_ipa(cls, ipa: str, grapheme: str | None = None, frequency: float = 1.0) -> "Phoneme":
        s = sound(ipa)
        return cls(id=s.ipa, grapheme=grapheme or s.default_grapheme, ipa=s.ipa,
                   phoneme_class=s.phoneme_class, frequency=frequency)

    @classmethod
    def from_grapheme(cls, grapheme: str, frequency: float = 1.0) -> "Phoneme":
        """The sound a letter spells by default, spelled that way: "y" -> /j/ written "y"."""
        s = sound_for_grapheme(grapheme)
        return cls.from_ipa(s.ipa, grapheme=s.default_grapheme, frequency=frequency)

    @property
    def sound(self) -> Sound:
        return sound(self.id)


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
    # Whether s-like sounds may break sonority at a cluster's edge: st-, sk-, -ks, -ts.
    allow_s_clusters: bool = False

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
            entry = SOUNDS.get(p.id)
            if entry is None:
                problems.append(f"unknown sound /{p.id}/")
                continue
            if p.ipa != p.id:
                problems.append(f"phoneme /{p.id}/ has ipa {p.ipa!r}; they must match")
            if entry.phoneme_class != p.phoneme_class:
                problems.append(f"/{p.id}/ must be class {entry.phoneme_class}")
            if not p.grapheme or p.grapheme != nfc(p.grapheme).lower() or any(
                    ch.isspace() or ch == "-" for ch in p.grapheme):
                problems.append(f"grapheme {p.grapheme!r} for /{p.id}/ must be lowercase NFC text "
                                "without spaces or hyphens")

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
        if any(p.endswith("C") for p in self.syllable_patterns) and not self.consonants_allowed_in(
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

    def by_grapheme(self, grapheme: str) -> Phoneme:
        for p in self.phonemes:
            if p.grapheme == grapheme:
                return p
        raise KeyError(grapheme)

    def spell(self, phoneme_ids) -> str:
        """Written form for messages; unknown ids are shown as /ipa/ instead of failing."""
        known = {p.id: p.grapheme for p in self.phonemes}
        return "".join(known.get(p, f"/{p}/") for p in phoneme_ids)

    @property
    def has_multi_letter_graphemes(self) -> bool:
        """Whether spelling can be ambiguous (s + h written "sh" vs /ʃ/ written "sh")."""
        return any(len(p.grapheme) > 1 for p in self.phonemes)

    def has_phoneme(self, phoneme_id: str) -> bool:
        return any(p.id == phoneme_id for p in self.phonemes)

    def is_forbidden(self, phoneme_id: str, position: Position) -> bool:
        return any(
            r.phoneme_id == phoneme_id and position in r.forbidden_positions
            for r in self.restrictions
        )

    def consonants_allowed_in(self, position: Position) -> list[Phoneme]:
        return [c for c in self.consonants if not self.is_forbidden(c.id, position)]
