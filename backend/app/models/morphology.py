from enum import StrEnum

from pydantic import Field

from app.models.common import DomainModel, new_id
from app.models.lexicon import PartOfSpeech


class MorphemeType(StrEnum):
    ROOT = "root"
    PREFIX = "prefix"
    SUFFIX = "suffix"


class MorphemeOrigin(StrEnum):
    USER_DEFINED = "user_defined"
    GENERATED = "generated"


class Morpheme(DomainModel):
    """A minimal meaningful unit: a root like `nal` (sea) or a suffix like `-ka`.

    Word forms are assembled from morphemes, so this stores the underlying
    phonemes. Sound changes will later operate on these.
    """

    id: str = Field(default_factory=new_id)
    language_id: str
    phonemes: list[str] = Field(min_length=1)
    type: MorphemeType
    gloss: str
    origin: MorphemeOrigin = MorphemeOrigin.GENERATED
    # The language version in which this morpheme entered the language; None = founding.
    # Its phonemes are as they were then; later sound changes are applied on top.
    introduced_in_version_id: str | None = None


class BoundaryRepair(DomainModel):
    """Deterministic fixes for illegal junctions between morphemes.

    Applied only where needed: `nal + ka` stays `nalka`, but if `nal + s`
    is illegal, a linking vowel "i" gives `nalis`. Never improvised by an LLM.
    """

    # Inserted between two consonants that cannot meet, e.g. nal + s -> nal-i-s.
    linking_vowel: str | None = None
    # Inserted between two vowels when hiatus is not allowed, e.g. sava + a -> sava-n-a.
    linking_consonant: str | None = None


class DerivationalRule(DomainModel):
    """A productive pattern, e.g. "suffix -ka turns a noun into 'person associated with it'"."""

    id: str
    name: str
    meaning: str
    morpheme_id: str
    input_parts_of_speech: list[PartOfSpeech] = Field(min_length=1)
    output_part_of_speech: PartOfSpeech
