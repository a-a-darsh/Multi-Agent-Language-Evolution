"""Deterministic phonotactic validation.

A form is legal only if it can be split into syllables that each match an
allowed pattern, keep every consonant cluster in sonority order, and respect
every positional restriction. Checking that each letter is in the inventory
is not enough: "strk" uses valid consonants but cannot be syllabified under
CV/CVC.

Clusters follow the sonority sequencing principle: consonants rise toward the
vowel and fall after it, so "plak" and "kalp" are fine but "lpak" and "kapl"
are not. With `allow_s_clusters`, an s-like sound may sit outside that order
next to a stop at the syllable edge (st-, sk-, -ks, -ts), as in many languages.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from app.generation.phoneme_generator import parse_spelling, render_phonemes
from app.models.phonology import Phonology, PhonemeClass, Position
from app.models.sounds import is_s_like, is_stop, sonority
from app.models.validation import ValidationResult


@dataclass(frozen=True)
class Syllable:
    phonemes: tuple[str, ...]
    pattern: str

    @property
    def onset(self) -> tuple[str, ...]:
        return self.phonemes[:self.pattern.index("V")]

    @property
    def coda(self) -> tuple[str, ...]:
        return self.phonemes[self.pattern.index("V") + 1:]


def _rising(cluster: tuple[str, ...]) -> bool:
    return all(sonority(a) < sonority(b) for a, b in zip(cluster, cluster[1:]))


def onset_is_legal(onset: tuple[str, ...], allow_s_clusters: bool) -> bool:
    if allow_s_clusters and len(onset) >= 2 and is_s_like(onset[0]) and is_stop(onset[1]):
        onset = onset[1:]  # s + stop (+ rising rest): st-, str-, sk-
    return _rising(onset)


def coda_is_legal(coda: tuple[str, ...], allow_s_clusters: bool) -> bool:
    if allow_s_clusters and len(coda) >= 2 and is_s_like(coda[-1]) and is_stop(coda[-2]):
        coda = coda[:-1]  # (falling rest +) stop + s: -ks, -ts, -nks
    return _rising(tuple(reversed(coda)))


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


def _restriction_violations(syllables: list[Syllable], phonology: Phonology) -> list[tuple[str, str]]:
    """(code, message) for everything wrong with one syllabification."""
    violations = []
    for s in syllables:
        for c in s.onset:
            if phonology.is_forbidden(c, Position.ONSET):
                violations.append(("POSITION_RESTRICTION", f"{_letter(c, phonology)} cannot start a syllable"))
        for c in s.coda:
            if phonology.is_forbidden(c, Position.CODA):
                violations.append(("POSITION_RESTRICTION", f"{_letter(c, phonology)} cannot end a syllable"))
        if len(s.onset) > 1 and not onset_is_legal(s.onset, phonology.allow_s_clusters):
            violations.append(("ILLEGAL_CLUSTER", f"{_letter_group(s.onset, phonology)} cannot begin a syllable "
                               "(consonants must get more vowel-like toward the vowel)"))
        if len(s.coda) > 1 and not coda_is_legal(s.coda, phonology.allow_s_clusters):
            violations.append(("ILLEGAL_CLUSTER", f"{_letter_group(s.coda, phonology)} cannot end a syllable "
                               "(consonants must get less vowel-like after the vowel)"))
    return violations


def _letter_group(phonemes: tuple[str, ...], phonology: Phonology) -> str:
    return f"/{phonology.spell(phonemes)}/"


def _letter(phoneme_id: str, phonology: Phonology) -> str:
    """How a sound is named in messages: by its spelling (which the LLM writes in), as before."""
    return f"/{phonology.spell([phoneme_id])}/"


def _parse_candidates(phonemes: Sequence[str], phonology: Phonology) -> list[list[Syllable]]:
    """Structurally valid parses, best first.

    Ordering follows the maximal-onset principle (fewest coda consonants), so
    "aka" is a.ka rather than ak.a, and "aplo" is a.plo rather than ap.lo.
    """
    classes = "".join(phonology.phoneme(p).phoneme_class.value for p in phonemes)
    parses = list(_all_parses(classes, phonology.syllable_patterns))
    parses.sort(key=lambda parse: sum(len(pat) - pat.index("V") - 1 for pat in parse))
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
    form = phonology.spell(phonemes)

    if not phonemes:
        result.add("EMPTY_FORM", "Form has no phonemes", object_id=object_id)
        return result

    unknown = sorted({p for p in phonemes if not phonology.has_phoneme(p)})
    if unknown:
        result.add("UNKNOWN_PHONEME", f"{form!r} uses phonemes not in the inventory: {unknown}",
                   object_id=object_id, details={"form": form, "phonemes": unknown})
        return result

    if phonology.is_forbidden(phonemes[0], Position.WORD_INITIAL):
        result.add("POSITION_RESTRICTION", f"{_letter(phonemes[0], phonology)} cannot begin a word",
                   object_id=object_id, details={"form": form, "position": "word_initial"})
    if phonology.is_forbidden(phonemes[-1], Position.WORD_FINAL):
        result.add("POSITION_RESTRICTION", f"{_letter(phonemes[-1], phonology)} cannot end a word",
                   object_id=object_id, details={"form": form, "position": "word_final"})

    if not phonology.allow_hiatus:
        for a, b in zip(phonemes, phonemes[1:]):
            if all(phonology.phoneme(p).phoneme_class == PhonemeClass.VOWEL for p in (a, b)):
                result.add("HIATUS_NOT_ALLOWED", f"{form!r} has adjacent vowels /{phonology.spell([a, b])}/",
                           object_id=object_id, details={"form": form, "sequence": phonology.spell([a, b])})
                break

    # With digraphs, s + h is written "sh" and would be read back as /ʃ/. A form is
    # only legal if its spelling reads back to exactly the same sounds.
    if phonology.has_multi_letter_graphemes:
        spelled = render_phonemes(phonemes, phonology)
        if parse_spelling(spelled, phonology) != list(phonemes):
            result.add("AMBIGUOUS_SPELLING",
                       f"{'+'.join(phonology.spell([p]) for p in phonemes)} would be written {spelled!r}, "
                       "which reads as different sounds",
                       object_id=object_id, details={"form": spelled, "phonemes": list(phonemes)})

    candidates = _parse_candidates(phonemes, phonology)
    if not candidates:
        result.add("ILLEGAL_SYLLABLE_STRUCTURE",
                   f"{form!r} cannot be split into allowed syllables {phonology.syllable_patterns}",
                   object_id=object_id,
                   details={"form": form, "allowed_patterns": phonology.syllable_patterns})
    elif all(_restriction_violations(s, phonology) for s in candidates):
        for code, violation in _restriction_violations(candidates[0], phonology):
            result.add(code, f"{form!r}: {violation}", object_id=object_id, details={"form": form})
    return result
