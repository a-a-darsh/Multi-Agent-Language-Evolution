import hashlib
import itertools
import math
import random
import unicodedata
from collections.abc import Sequence
from functools import cache

from app.models.phonology import Phoneme, Phonology, Position
from app.models.sounds import DEFAULT_SPELLINGS, INPUT_ALIASES


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


@cache
def _legal_clusters(consonants: tuple[tuple[str, float], ...], size: int, onset: bool,
                    allow_s_clusters: bool) -> tuple[tuple[tuple[str, ...], ...], tuple[float, ...]]:
    """Every sonority-legal cluster of `size` from these consonants, with sampling weights."""
    from app.validation.phonology_validator import coda_is_legal, onset_is_legal  # avoid an import cycle

    legal = onset_is_legal if onset else coda_is_legal
    weight = dict(consonants)
    clusters = [c for c in itertools.product(weight, repeat=size) if legal(c, allow_s_clusters)]
    return tuple(clusters), tuple(math.prod(weight[p] for p in c) for c in clusters)


def _pick_cluster(rng: random.Random, phonology: Phonology, consonants: Sequence[Phoneme], size: int,
                  onset: bool) -> list[str]:
    clusters, weights = _legal_clusters(tuple((p.id, p.frequency) for p in consonants), size, onset,
                                        phonology.allow_s_clusters)
    if not clusters:
        raise GenerationError(f"no legal {size}-consonant {'onset' if onset else 'coda'} exists in this language")
    return list(rng.choices(clusters, weights=weights)[0])


def generate_syllable(phonology: Phonology, rng: random.Random, pattern: str | None = None) -> list[str]:
    pattern = pattern or rng.choice(phonology.syllable_patterns)
    if pattern not in phonology.syllable_patterns:
        raise GenerationError(f"pattern {pattern!r} is not allowed in this language")

    onsets = phonology.consonants_allowed_in(Position.ONSET)
    codas = phonology.consonants_allowed_in(Position.CODA)
    nucleus = pattern.index("V")
    onset_size, coda_size = nucleus, len(pattern) - nucleus - 1
    syllable = []
    # Single consonants are drawn one at a time (unchanged since the MVP, so seeds reproduce
    # old roots); clusters are drawn whole from the clusters that obey sonority.
    if onset_size == 1:
        syllable.append(_pick(rng, onsets))
    elif onset_size > 1:
        syllable += _pick_cluster(rng, phonology, onsets, onset_size, onset=True)
    syllable.append(_pick(rng, phonology.vowels))
    if coda_size == 1:
        syllable.append(_pick(rng, codas))
    elif coda_size > 1:
        syllable += _pick_cluster(rng, phonology, codas, coda_size, onset=False)
    return syllable


def render_phonemes(phonemes: Sequence[str], phonology: Phonology) -> str:
    return "".join(phonology.phoneme(p).grapheme for p in phonemes)


def parse_spelling(spelling: str, phonology: Phonology | None = None) -> list[str]:
    """Convert a written form into phoneme ids (IPA).

    With a phonology, reads that language's own letters. Without one, reads the
    catalog's default spellings (used to analyze user words before the
    language's inventory exists). Only case, Unicode normalization, and typing
    aliases (' for ʼ) are applied; the letters themselves are never changed.
    """
    normalized = unicodedata.normalize("NFC", spelling.strip()).lower()
    if phonology is None:
        known = dict(DEFAULT_SPELLINGS)
    else:
        known = {p.grapheme: p.id for p in phonology.phonemes}
    # An alias only applies where it is not a real letter of the language.
    for alias, letter in INPUT_ALIASES.items():
        if alias not in known and letter in known:
            known[alias] = known[letter]
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
