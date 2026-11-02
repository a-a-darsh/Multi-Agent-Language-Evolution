import hashlib
import random
import unicodedata
from collections.abc import Sequence

from app.models.phonology import GRAPHEME_CATALOG, Phoneme, Phonology, Position


class GenerationError(Exception):
    pass


class UnsupportedSpellingError(ValueError):
    def __init__(self, spelling: str, unsupported: list[str]):
        self.spelling = spelling
        self.unsupported = unsupported
        super().__init__(f"{spelling!r} contains unsupported letters: {unsupported}")


def derive_rng(seed: int, *labels: str) -> random.Random:
    """An independent, reproducible random stream for one purpose.

    Separate streams (e.g. "roots", "affixes") mean adding a new random step
    later doesn't change the output of existing ones for the same seed.
    Uses sha256 because Python's built-in hash() is randomized per process.
    """
    key = ":".join([str(seed), *labels]).encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def _pick(rng: random.Random, phonemes: Sequence[Phoneme]) -> str:
    return rng.choices(phonemes, weights=[p.frequency for p in phonemes])[0].id


def generate_syllable(phonology: Phonology, rng: random.Random, pattern: str | None = None) -> list[str]:
    pattern = pattern or rng.choice(phonology.syllable_patterns)
    if pattern not in phonology.syllable_patterns:
        raise GenerationError(f"pattern {pattern!r} is not allowed in this language")

    onsets = phonology.consonants_allowed_in(Position.ONSET)
    codas = phonology.consonants_allowed_in(Position.CODA)
    syllable = []
    for i, slot in enumerate(pattern):
        if slot == "V":
            syllable.append(_pick(rng, phonology.vowels))
        else:
            # A consonant before the vowel is an onset, after it a coda.
            syllable.append(_pick(rng, onsets if "V" in pattern[i:] else codas))
    return syllable


def render_phonemes(phonemes: Sequence[str], phonology: Phonology) -> str:
    return "".join(phonology.phoneme(p).grapheme for p in phonemes)


def parse_spelling(spelling: str, phonology: Phonology | None = None) -> list[str]:
    """Convert a written form into phoneme ids.

    With no phonology, accepts anything in the global grapheme catalog (used
    to analyze user words before the language's inventory exists). Only case
    and Unicode normalization are applied; the letters themselves are never changed.
    """
    normalized = unicodedata.normalize("NFC", spelling.strip()).lower()
    if phonology is None:
        known = {g: g for g in GRAPHEME_CATALOG}
    else:
        known = {p.grapheme: p.id for p in phonology.phonemes}
    longest = max(map(len, known))
    phonemes, unsupported, i = [], set(), 0
    while i < len(normalized):
        # Longest match first, so "sh" is read as one sound when the language has it.
        for size in range(min(longest, len(normalized) - i), 0, -1):
            piece = normalized[i:i + size]
            if piece in known:
                phonemes.append(known[piece])
                i += size
                break
        else:
            unsupported.add(normalized[i])
            i += 1
    if unsupported or not normalized:
        raise UnsupportedSpellingError(spelling, sorted(unsupported))
    return phonemes
