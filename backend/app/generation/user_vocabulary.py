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
from app.models.phonology import SUPPORTED_PATTERNS, Phoneme, Phonology, PhonemeClass, pattern_order
from app.models.sounds import SOUNDS, sound
from app.models.validation import Severity, ValidationResult
from app.validation.phonology_validator import syllabify, validate_phoneme_sequence


class UserWord(DomainModel):
    concept_id: str
    spelling: str
    phonemes: list[str]  # IPA ids
    syllable_patterns: list[str]
    needs_s_clusters: bool = False


class PhonologyRequirements(DomainModel):
    words: list[UserWord] = Field(default_factory=list)
    required_phonemes: list[str] = Field(default_factory=list)
    required_patterns: list[str] = Field(default_factory=list)
    requires_hiatus: bool = False
    requires_s_clusters: bool = False

    @property
    def required_letters(self) -> list[str]:
        """The required sounds as letters (default spellings), which is what the design agent works in."""
        return sorted(sound(p).default_grapheme for p in self.required_phonemes)


def _permissive_phonology(allow_s_clusters: bool) -> Phonology:
    """Every supported phoneme and pattern - used to analyze words before an inventory exists."""
    return Phonology(
        phonemes=[Phoneme.from_ipa(ipa) for ipa in SOUNDS],
        syllable_patterns=sorted(SUPPORTED_PATTERNS, key=pattern_order),
        allow_hiatus=True,
        allow_s_clusters=allow_s_clusters,
    )


def analyze_user_words(custom_words: Mapping[str, str]) -> tuple[PhonologyRequirements, ValidationResult]:
    result = ValidationResult()
    permissive, with_s = _permissive_phonology(False), _permissive_phonology(True)
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
        # s-clusters (st-, -ks) are only switched on if a word needs them.
        syllables, needs_s = syllabify(phonemes, permissive), False
        if syllables is None and (syllables := syllabify(phonemes, with_s)) is not None:
            needs_s = True
        if syllables is None:
            problems = "; ".join(i.message for i in validate_phoneme_sequence(phonemes, with_s).errors)
            result.add("UNSUPPORTED_SYLLABLE_STRUCTURE",
                       f"The word {spelling!r} for {concept_id!r} has a consonant cluster or shape the "
                       f"engine does not support ({problems}). Clusters must get more vowel-like toward "
                       "the vowel (pl-, kr-, st-) and less after it (-lp, -nt, -ks), at most 3 consonants",
                       object_id=concept_id, details={"spelling": spelling})
            continue
        words.append(UserWord(concept_id=concept_id, spelling=spelling, phonemes=phonemes,
                              syllable_patterns=[s.pattern for s in syllables], needs_s_clusters=needs_s))

    by_form: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for w in words:
        by_form[tuple(w.phonemes)].append(w.concept_id)
    for form, concepts in by_form.items():
        if len(concepts) > 1:
            # Intentional polysemy (kijur = zinc, wealth): one word with several meanings.
            result.add("USER_POLYSEMY",
                       f"{permissive.spell(form)!r} will be one word meaning {', '.join(concepts)}",
                       details={"form": permissive.spell(form), "concepts": concepts}, severity=Severity.WARNING)

    requirements = PhonologyRequirements(
        words=words,
        required_phonemes=sorted({p for w in words for p in w.phonemes}),
        required_patterns=sorted({pat for w in words for pat in w.syllable_patterns}),
        requires_s_clusters=any(w.needs_s_clusters for w in words),
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
    return sound(phoneme_id).phoneme_class == PhonemeClass.VOWEL
