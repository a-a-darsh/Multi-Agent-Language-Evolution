"""Deterministic phonotactic validation.

A form is legal only if it can be split into syllables that each match an
allowed pattern and respect every positional restriction. Checking that each
letter is in the inventory is not enough: "strk" uses valid consonants but
cannot be syllabified under CV/CVC.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from app.models.phonology import Phonology, PhonemeClass, Position
from app.models.validation import ValidationResult


@dataclass(frozen=True)
class Syllable:
    phonemes: tuple[str, ...]
    pattern: str

    @property
    def onset(self) -> str | None:
        return self.phonemes[0] if self.pattern.startswith("C") else None

    @property
    def coda(self) -> str | None:
        return self.phonemes[-1] if self.pattern.endswith("VC") else None


def _all_parses(classes: str, patterns: Sequence[str], start: int = 0) -> Iterator[list[str]]:
    """Every way to tile the C/V string with the allowed patterns."""
    if start == len(classes):
        yield []
        return
    for pattern in patterns:
        if classes.startswith(pattern, start):
            for rest in _all_parses(classes, patterns, start + len(pattern)):
                yield [pattern, *rest]


def _to_syllables(phonemes: Sequence[str], parse: list[str]) -> list[Syllable]:
    syllables, i = [], 0
    for pattern in parse:
        syllables.append(Syllable(tuple(phonemes[i:i + len(pattern)]), pattern))
        i += len(pattern)
    return syllables


def _restriction_violations(syllables: list[Syllable], phonology: Phonology) -> list[str]:
    violations = []
    for s in syllables:
        if s.onset and phonology.is_forbidden(s.onset, Position.ONSET):
            violations.append(f"/{s.onset}/ cannot start a syllable")
        if s.coda and phonology.is_forbidden(s.coda, Position.CODA):
            violations.append(f"/{s.coda}/ cannot end a syllable")
    return violations


def _parse_candidates(phonemes: Sequence[str], phonology: Phonology) -> list[list[Syllable]]:
    """Structurally valid parses, best first.

    Ordering follows the maximal-onset principle (fewest codas), so "aka"
    is syllabified a.ka rather than ak.a.
    """
    classes = "".join(phonology.phoneme(p).phoneme_class.value for p in phonemes)
    parses = list(_all_parses(classes, phonology.syllable_patterns))
    parses.sort(key=lambda parse: sum(pat.endswith("VC") for pat in parse))
    return [_to_syllables(phonemes, parse) for parse in parses]


def syllabify(phonemes: Sequence[str], phonology: Phonology) -> list[Syllable] | None:
    """The preferred legal syllabification, or None if the form is illegal."""
    if not validate_phoneme_sequence(phonemes, phonology).ok:
        return None
    return next(
        s for s in _parse_candidates(phonemes, phonology)
        if not _restriction_violations(s, phonology)
    )


def validate_phoneme_sequence(
    phonemes: Sequence[str], phonology: Phonology, object_id: str | None = None
) -> ValidationResult:
    result = ValidationResult()
    form = "".join(phonemes)

    if not phonemes:
        result.add("EMPTY_FORM", "Form has no phonemes", object_id=object_id)
        return result

    unknown = sorted({p for p in phonemes if not phonology.has_phoneme(p)})
    if unknown:
        result.add("UNKNOWN_PHONEME", f"{form!r} uses phonemes not in the inventory: {unknown}",
                   object_id=object_id, details={"form": form, "phonemes": unknown})
        return result

    if phonology.is_forbidden(phonemes[0], Position.WORD_INITIAL):
        result.add("POSITION_RESTRICTION", f"/{phonemes[0]}/ cannot begin a word",
                   object_id=object_id, details={"form": form, "position": "word_initial"})
    if phonology.is_forbidden(phonemes[-1], Position.WORD_FINAL):
        result.add("POSITION_RESTRICTION", f"/{phonemes[-1]}/ cannot end a word",
                   object_id=object_id, details={"form": form, "position": "word_final"})

    if not phonology.allow_hiatus:
        for a, b in zip(phonemes, phonemes[1:]):
            if all(phonology.phoneme(p).phoneme_class == PhonemeClass.VOWEL for p in (a, b)):
                result.add("HIATUS_NOT_ALLOWED", f"{form!r} has adjacent vowels /{a}{b}/",
                           object_id=object_id, details={"form": form, "sequence": a + b})
                break

    candidates = _parse_candidates(phonemes, phonology)
    if not candidates:
        result.add("ILLEGAL_SYLLABLE_STRUCTURE",
                   f"{form!r} cannot be split into allowed syllables {phonology.syllable_patterns}",
                   object_id=object_id,
                   details={"form": form, "allowed_patterns": phonology.syllable_patterns})
    elif all(_restriction_violations(s, phonology) for s in candidates):
        for violation in _restriction_violations(candidates[0], phonology):
            result.add("POSITION_RESTRICTION", f"{form!r}: {violation}",
                       object_id=object_id, details={"form": form})
    return result
