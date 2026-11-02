"""The schema the Language Design Agent must fill in.

This is an LLM-facing contract, kept simpler than the domain models: plain
letters instead of Phoneme objects, an affix's spelling instead of separate
Morpheme + DerivationalRule records. `design_validation.build_design`
converts it into domain models.

Field descriptions are part of the prompt: they are sent to the model inside
the JSON schema. Enum-like Literal types constrain what the model can emit.

Lesson learned: fields with defaults become *optional* in the JSON schema,
and with constrained decoding a model can simply skip them. Fields the model
must think about (features, restrictions, linking segments) therefore have
no defaults - they may be empty or null, but the model has to say so.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.models.grammar import AdpositionType, MorphologicalType, WordOrder
from app.models.lexicon import PartOfSpeech
from app.models.phonology import GRAPHEME_CATALOG, PhonemeClass, Position

VowelLetter = Literal[tuple(g for g, e in GRAPHEME_CATALOG.items() if e.phoneme_class == PhonemeClass.VOWEL)]
ConsonantLetter = Literal[tuple(g for g, e in GRAPHEME_CATALOG.items() if e.phoneme_class == PhonemeClass.CONSONANT)]
Letter = Literal[tuple(GRAPHEME_CATALOG)]
SyllablePattern = Literal["V", "CV", "VC", "CVC"]


class RestrictionDraft(BaseModel):
    letter: ConsonantLetter
    forbidden_positions: list[Position] = Field(
        min_length=1, description="Where this consonant may not appear: onset, coda, word_initial, word_final.")


class FeatureDraft(BaseModel):
    name: str = Field(description="Short name, e.g. 'Evidentiality'.")
    description: str = Field(description="What the feature means, in one or two plain sentences.")
    realization: str = Field(description="How it is expressed, e.g. 'verb suffix -si marks reported events'.")


class AffixDraft(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z_]{1,30}$",
                    description="Descriptive snake_case identifier, e.g. 'agent' or 'place_of'.")
    # min_length: the model once returned meaning='' for every affix, leaving the dictionary blank.
    name: str = Field(min_length=2, description="Human-readable name, e.g. 'Agent noun'.")
    meaning: str = Field(min_length=3, description="What it does, e.g. 'person associated with X' (sea -> sailor).")
    position: Literal["prefix", "suffix"]
    form: str = Field(description="Spelling of the affix using only the language's letters, usually 1-3 letters.")
    input_parts_of_speech: list[PartOfSpeech] = Field(min_length=1, description="Word classes it attaches to.")
    output_part_of_speech: PartOfSpeech = Field(description="Word class of the result.")


class LanguageDesignDraft(BaseModel):
    reasoning: str = Field(description=(
        "Think first, briefly: how the culture and the user's constraints shape the sound system and "
        "grammar, and how the affixes will attach cleanly to roots. 3-8 sentences."))
    name: str = Field(description="Name of the language, spelled with its own letters, e.g. 'Old Savari'.")

    consonants: list[ConsonantLetter] = Field(min_length=6, max_length=16, description=(
        "A distinctive subset of the available consonants that gives the language its character. "
        "Typically 8-14. Do not simply list every letter."))
    vowels: list[VowelLetter] = Field(min_length=2, max_length=5)
    common_sounds: list[Letter] = Field(
        description="Letters that should appear more often than others. May be empty.")
    syllable_patterns: list[SyllablePattern] = Field(
        min_length=1, description="Allowed syllable shapes. C = consonant, V = vowel. Include CV.")
    allow_hiatus: bool = Field(description="May two vowels be adjacent across syllables, as in 'ri.a'?")
    restrictions: list[RestrictionDraft] = Field(
        description="Position restrictions, e.g. no 'h' at the end of syllables. May be empty.")

    word_order: WordOrder
    morphological_type: MorphologicalType
    adposition_type: AdpositionType = Field(
        description="Postpositions usually go with verb-final (SOV) order, prepositions with VO orders.")
    has_grammatical_gender: bool = Field(description="Are nouns sorted into genders or noun classes?")
    has_case_marking: bool = Field(description="Do nouns take endings for their role (subject, object...)?")
    features: list[FeatureDraft] = Field(max_length=6, description=(
        "The few most notable grammatical features (at most 6). Every feature the user requested MUST "
        "have an entry here. May be empty only if the user requested none."))

    derivational_affixes: list[AffixDraft] = Field(min_length=3, max_length=8, description=(
        "Word-building affixes that create NEW words with new meanings from roots: agent (sea -> sailor), "
        "place, instrument, quality, small thing, collective, verb-from-noun... Not grammatical inflection "
        "such as plural, tense, or case - those belong in `features`."))
    linking_vowel: VowelLetter | None = Field(
        description="Vowel inserted when two consonants can't meet at an affix boundary, or null.")
    linking_consonant: ConsonantLetter | None = Field(
        description="Consonant inserted when two vowels meet but hiatus is not allowed, or null.")

    summary: str = Field(description="A friendly 2-4 sentence overview of the language for the user.")


# --- critic ------------------------------------------------------------------
#
# Lesson learned: a critic asked "any problems?" tends to rubber-stamp ("the
# design is acceptable"). So the critic must review every affix separately,
# apply it to an example first (show its work), and only then classify it.
# Code turns the classifications into findings; the model never gives a
# holistic verdict.


class AffixReview(BaseModel):
    affix_id: str = Field(max_length=40, description="The affix's exact id from the design.")
    example: str = Field(max_length=120, description=(
        "Apply the affix to an English base word and show the result, e.g. 'sea -> sailor' or "
        "'see -> see (marked: I witnessed it)'."))
    kind: Literal["derivational", "inflectional"] = Field(description=(
        "derivational: the result is a different word that would get its own dictionary entry. "
        "inflectional: the result is the same word marked for case, number, tense, aspect, mood, "
        "evidentiality, person, or agreement."))


class CritiqueFinding(BaseModel):
    target: str = Field(max_length=60, description="What the problem is about: 'grammar', 'feature:<name>', or 'other'.")
    problem_type: Literal["incoherent_grammar", "other"]
    explanation: str = Field(max_length=300, description="What is wrong, in one or two sentences.")
    suggestion: str = Field(max_length=200, description="A concrete fix, in one sentence.")


class Critique(BaseModel):
    affix_reviews: list[AffixReview] = Field(description="One review for EVERY derivational affix, in order.")
    inventory_looks_designed: bool = Field(description=(
        "false if the consonants look like a near-complete list of available letters in alphabetical "
        "order rather than a distinctive choice for this language."))
    other_findings: list[CritiqueFinding] = Field(max_length=3, description=(
        "Other real problems, e.g. grammar choices that contradict each other. Usually empty."))
