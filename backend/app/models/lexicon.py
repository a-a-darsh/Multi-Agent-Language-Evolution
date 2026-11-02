from enum import StrEnum

from pydantic import Field, model_validator

from app.models.common import DomainModel, new_id


class PartOfSpeech(StrEnum):
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PRONOUN = "pronoun"
    NUMERAL = "numeral"
    ADPOSITION = "adposition"
    PARTICLE = "particle"
    INTERJECTION = "interjection"


class ConceptPriority(StrEnum):
    CORE = "core"
    CULTURAL = "cultural"


class SemanticConcept(DomainModel):
    """A language-independent meaning, identified by a stable id ("sea", not "Sea")."""

    id: str
    gloss: str
    part_of_speech: PartOfSpeech
    categories: list[str] = Field(default_factory=list)
    priority: ConceptPriority
    # For cultural concepts: which civilization traits make this concept likely.
    cultural_tags: list[str] = Field(default_factory=list)


class LexemeOrigin(StrEnum):
    USER_DEFINED = "user_defined"
    GENERATED_ROOT = "generated_root"
    DERIVED = "derived"


class LexicalSense(DomainModel):
    id: str = Field(default_factory=new_id)
    concept_id: str
    gloss: str


class Derivation(DomainModel):
    """Records that a lexeme was built from a base lexeme by a derivational rule."""

    id: str = Field(default_factory=new_id)
    output_lexeme_id: str
    base_lexeme_id: str
    rule_id: str
    applied_morpheme_ids: list[str] = Field(min_length=1)


class Lexeme(DomainModel):
    """A word as an abstract identity that persists across historical changes.

    `morpheme_ids` is the ordered underlying composition (e.g. [nal, -ka]).
    Spellings live in LexemeForm, one per language version, so `apa` and its
    later descendant `afa` remain the same lexeme.
    """

    id: str = Field(default_factory=new_id)
    language_id: str
    concept_id: str
    part_of_speech: PartOfSpeech
    origin: LexemeOrigin
    morpheme_ids: list[str] = Field(min_length=1)
    senses: list[LexicalSense] = Field(min_length=1)
    derivation: Derivation | None = None

    @model_validator(mode="after")
    def _derived_iff_has_derivation(self) -> "Lexeme":
        is_derived = self.origin == LexemeOrigin.DERIVED
        if is_derived != (self.derivation is not None):
            raise ValueError("derived lexemes must have a derivation, others must not")
        if self.derivation is not None and self.derivation.output_lexeme_id != self.id:
            raise ValueError("derivation.output_lexeme_id must equal the lexeme id")
        return self


class FormSegment(DomainModel):
    """One piece of a realized form: a morpheme, or a linking segment (morpheme_id None)."""

    phonemes: list[str] = Field(min_length=1)
    morpheme_id: str | None = None


class LexemeForm(DomainModel):
    """The realized form of a lexeme in one language version."""

    id: str = Field(default_factory=new_id)
    lexeme_id: str
    language_version_id: str
    phonemes: list[str] = Field(min_length=1)
    orthography: str
    # Segmentation for display and analysis, e.g. [nal][ka] or [nal][i][s].
    segments: list[FormSegment] = Field(default_factory=list)
