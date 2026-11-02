"""Analysis of user-defined words, done before any phonology is designed.

User words are authoritative. We work out what they require (phonemes,
syllable shapes, hiatus) so the Language Design Agent receives these as hard
facts, then check any proposed phonology against them.
"""

from collections import defaultdict
from collections.abc import Mapping

from pydantic import Field

from app.generation.phoneme_generator import UnsupportedSpellingError, parse_spelling
from app.models.common import DomainModel
from app.models.phonology import GRAPHEME_CATALOG, SUPPORTED_PATTERNS, Phoneme, Phonology, PhonemeClass
from app.models.validation import ValidationResult
from app.validation.phonology_validator import syllabify, validate_phoneme_sequence


class UserWord(DomainModel):
    concept_id: str
    spelling: str
    phonemes: list[str]
    syllable_patterns: list[str]


class PhonologyRequirements(DomainModel):
    words: list[UserWord] = Field(default_factory=list)
    required_phonemes: list[str] = Field(default_factory=list)
    required_patterns: list[str] = Field(default_factory=list)
    requires_hiatus: bool = False


def _permissive_phonology() -> Phonology:
    """Every supported phoneme and pattern - used to analyze words before an inventory exists."""
    return Phonology(
        phonemes=[Phoneme.from_grapheme(g) for g in GRAPHEME_CATALOG],
        syllable_patterns=sorted(SUPPORTED_PATTERNS),
        allow_hiatus=True,
    )


def analyze_user_words(custom_words: Mapping[str, str]) -> tuple[PhonologyRequirements, ValidationResult]:
    result = ValidationResult()
    permissive = _permissive_phonology()
    words: list[UserWord] = []

    for concept_id, spelling in custom_words.items():
        try:
            phonemes = parse_spelling(spelling)
        except UnsupportedSpellingError as e:
            result.add("UNSUPPORTED_GRAPHEME",
                       f"The word {spelling!r} for {concept_id!r} uses unsupported letters {e.unsupported}",
                       object_id=concept_id,
                       details={"spelling": spelling, "unsupported": e.unsupported})
            continue
        syllables = syllabify(phonemes, permissive)
        if syllables is None:
            result.add("UNSUPPORTED_SYLLABLE_STRUCTURE",
                       f"The word {spelling!r} for {concept_id!r} has a consonant cluster or shape "
                       f"the generator does not support yet (allowed shapes: V, CV, VC, CVC)",
                       object_id=concept_id, details={"spelling": spelling})
            continue
        words.append(UserWord(concept_id=concept_id, spelling=spelling, phonemes=phonemes,
                              syllable_patterns=[s.pattern for s in syllables]))

    by_form: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for w in words:
        by_form[tuple(w.phonemes)].append(w.concept_id)
    for form, concepts in by_form.items():
        if len(concepts) > 1:
            result.add("DUPLICATE_USER_FORM",
                       f"The same word {''.join(form)!r} was given for {concepts}",
                       details={"form": "".join(form), "concepts": concepts})

    requirements = PhonologyRequirements(
        words=words,
        required_phonemes=sorted({p for w in words for p in w.phonemes}),
        required_patterns=sorted({pat for w in words for pat in w.syllable_patterns}),
        requires_hiatus=any(
            _is_vowel(a) and _is_vowel(b) for w in words for a, b in zip(w.phonemes, w.phonemes[1:])
        ),
    )
    return requirements, result


def check_phonology_satisfies(phonology: Phonology, requirements: PhonologyRequirements) -> ValidationResult:
    """Every user word must be legal, unchanged, under the proposed phonology."""
    result = ValidationResult()
    for word in requirements.words:
        word_result = validate_phoneme_sequence(word.phonemes, phonology, object_id=word.concept_id)
        for issue in word_result.errors:
            result.add("USER_WORD_INCOMPATIBLE",
                       f"User word {word.spelling!r} ({word.concept_id}) is not legal: {issue.message}",
                       object_id=word.concept_id,
                       details={"spelling": word.spelling, "cause": issue.code, **issue.details})
    return result


def _is_vowel(phoneme_id: str) -> bool:
    return GRAPHEME_CATALOG[phoneme_id].phoneme_class == PhonemeClass.VOWEL
