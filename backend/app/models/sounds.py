"""The universal sound catalog: every phoneme the engine knows, with its features.

A phoneme is identified by its IPA symbol. Spelling is a separate, per-language
choice (`Phoneme.grapheme`); the catalog only supplies a *default* spelling.
Keeping identity and spelling apart is what lets a sound change rewrite /k/ to
/tʃ/ without touching orthography, and lets two languages spell /ʃ/ differently.

Features are what sound-change rules will match on ("velars before front
vowels"), so they are derived from the catalog, never stored per language:
fixing a feature here fixes it everywhere.

Default spellings follow two rules:
- Sounds of the original MVP alphabet keep their old spellings, so existing
  languages and user words read exactly as before (`founding=True`).
- Every sound added later is spelled with a single non-ASCII letter (ñ, ť, ā...),
  never a new ASCII digraph, so no existing ASCII spelling can be re-read as a
  new sound.
"""

import unicodedata
from dataclasses import asdict, dataclass
from enum import StrEnum


class PhonemeClass(StrEnum):
    CONSONANT = "C"
    VOWEL = "V"


class Place(StrEnum):
    BILABIAL = "bilabial"
    LABIODENTAL = "labiodental"
    DENTAL = "dental"
    ALVEOLAR = "alveolar"
    POSTALVEOLAR = "postalveolar"
    ALVEOLO_PALATAL = "alveolo_palatal"
    RETROFLEX = "retroflex"
    PALATAL = "palatal"
    VELAR = "velar"
    LABIAL_VELAR = "labial_velar"
    UVULAR = "uvular"
    GLOTTAL = "glottal"


class Manner(StrEnum):
    STOP = "stop"
    AFFRICATE = "affricate"
    FRICATIVE = "fricative"
    NASAL = "nasal"
    TRILL = "trill"
    APPROXIMANT = "approximant"


class Height(StrEnum):
    CLOSE = "close"
    CLOSE_MID = "close_mid"
    MID = "mid"
    OPEN_MID = "open_mid"
    NEAR_OPEN = "near_open"
    OPEN = "open"


class Backness(StrEnum):
    FRONT = "front"
    CENTRAL = "central"
    BACK = "back"


@dataclass(frozen=True)
class ConsonantFeatures:
    place: Place
    manner: Manner
    voiced: bool
    lateral: bool = False
    sibilant: bool = False


@dataclass(frozen=True)
class VowelFeatures:
    height: Height
    backness: Backness
    rounded: bool
    long: bool = False
    nasal: bool = False


@dataclass(frozen=True)
class Sound:
    ipa: str
    phoneme_class: PhonemeClass
    default_grapheme: str
    features: ConsonantFeatures | VowelFeatures
    # Part of the original alphabet offered to the founding-language design agent.
    founding: bool = False

    def feature_values(self) -> dict[str, str | bool]:
        """Flat feature map, e.g. {"class": "C", "place": "velar", "manner": "stop", "voiced": False}."""
        return {"class": self.phoneme_class.value, **asdict(self.features)}


# Sonority: how vowel-like a consonant is. Within a syllable, consonants rise in
# sonority toward the vowel and fall after it (pl- and -lp are fine, lp- and -pl are not).
_SONORITY = {Manner.STOP: 1, Manner.AFFRICATE: 2, Manner.FRICATIVE: 3, Manner.NASAL: 4,
             Manner.TRILL: 5, Manner.APPROXIMANT: 6}


def sonority(ipa: str) -> int:
    s = SOUNDS[ipa]
    if s.phoneme_class == PhonemeClass.VOWEL:
        return 7
    f = s.features
    return 5 if f.manner == Manner.APPROXIMANT and f.lateral else _SONORITY[f.manner]


def is_s_like(ipa: str) -> bool:
    """s, ʃ and similar: the sounds that may break sonority in st-, sk-, -ks clusters."""
    s = SOUNDS[ipa]
    return s.phoneme_class == PhonemeClass.CONSONANT and s.features.sibilant and s.features.manner == Manner.FRICATIVE


def is_stop(ipa: str) -> bool:
    s = SOUNDS[ipa]
    return s.phoneme_class == PhonemeClass.CONSONANT and s.features.manner == Manner.STOP


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


_C, _V = PhonemeClass.CONSONANT, PhonemeClass.VOWEL
P, M, H, B = Place, Manner, Height, Backness

# (ipa, default spelling, place, manner, voiced, extra flags, founding)
_CONSONANTS: list[tuple[str, str, Place, Manner, bool, dict, bool]] = [
    # stops
    ("p", "p", P.BILABIAL, M.STOP, False, {}, True),
    ("b", "b", P.BILABIAL, M.STOP, True, {}, True),
    ("t", "t", P.ALVEOLAR, M.STOP, False, {}, True),
    ("d", "d", P.ALVEOLAR, M.STOP, True, {}, True),
    ("ʈ", "ṭ", P.RETROFLEX, M.STOP, False, {}, False),
    ("ɖ", "ḍ", P.RETROFLEX, M.STOP, True, {}, False),
    ("c", "ť", P.PALATAL, M.STOP, False, {}, False),
    ("ɟ", "ď", P.PALATAL, M.STOP, True, {}, False),
    ("k", "k", P.VELAR, M.STOP, False, {}, True),
    ("ɡ", "g", P.VELAR, M.STOP, True, {}, True),
    ("q", "q", P.UVULAR, M.STOP, False, {}, True),
    ("ʔ", "ʼ", P.GLOTTAL, M.STOP, False, {}, False),
    # nasals
    ("m", "m", P.BILABIAL, M.NASAL, True, {}, True),
    ("n", "n", P.ALVEOLAR, M.NASAL, True, {}, True),
    ("ɳ", "ṇ", P.RETROFLEX, M.NASAL, True, {}, False),
    ("ɲ", "ñ", P.PALATAL, M.NASAL, True, {}, False),
    ("ŋ", "ng", P.VELAR, M.NASAL, True, {}, True),
    # fricatives
    ("β", "ƀ", P.BILABIAL, M.FRICATIVE, True, {}, False),
    ("f", "f", P.LABIODENTAL, M.FRICATIVE, False, {}, True),
    ("v", "v", P.LABIODENTAL, M.FRICATIVE, True, {}, True),
    ("θ", "th", P.DENTAL, M.FRICATIVE, False, {}, True),
    ("ð", "dh", P.DENTAL, M.FRICATIVE, True, {}, True),
    ("s", "s", P.ALVEOLAR, M.FRICATIVE, False, {"sibilant": True}, True),
    ("z", "z", P.ALVEOLAR, M.FRICATIVE, True, {"sibilant": True}, True),
    ("ɬ", "ł", P.ALVEOLAR, M.FRICATIVE, False, {"lateral": True}, False),
    ("ʃ", "sh", P.POSTALVEOLAR, M.FRICATIVE, False, {"sibilant": True}, True),
    ("ʒ", "zh", P.POSTALVEOLAR, M.FRICATIVE, True, {"sibilant": True}, True),
    ("ɕ", "ś", P.ALVEOLO_PALATAL, M.FRICATIVE, False, {"sibilant": True}, False),
    ("ʑ", "ź", P.ALVEOLO_PALATAL, M.FRICATIVE, True, {"sibilant": True}, False),
    ("ʂ", "ṣ", P.RETROFLEX, M.FRICATIVE, False, {"sibilant": True}, False),
    ("x", "x", P.VELAR, M.FRICATIVE, False, {}, True),
    ("ɣ", "ğ", P.VELAR, M.FRICATIVE, True, {}, False),
    ("h", "h", P.GLOTTAL, M.FRICATIVE, False, {}, True),
    # affricates
    ("ts", "c", P.ALVEOLAR, M.AFFRICATE, False, {"sibilant": True}, True),
    ("tʃ", "ch", P.POSTALVEOLAR, M.AFFRICATE, False, {"sibilant": True}, True),
    ("dʒ", "j", P.POSTALVEOLAR, M.AFFRICATE, True, {"sibilant": True}, True),
    ("tɕ", "ć", P.ALVEOLO_PALATAL, M.AFFRICATE, False, {"sibilant": True}, False),
    # liquids and glides
    ("r", "r", P.ALVEOLAR, M.TRILL, True, {}, True),
    ("l", "l", P.ALVEOLAR, M.APPROXIMANT, True, {"lateral": True}, True),
    ("ɭ", "ḷ", P.RETROFLEX, M.APPROXIMANT, True, {"lateral": True}, False),
    ("ʎ", "ľ", P.PALATAL, M.APPROXIMANT, True, {"lateral": True}, False),
    ("j", "y", P.PALATAL, M.APPROXIMANT, True, {}, True),
    ("w", "w", P.LABIAL_VELAR, M.APPROXIMANT, True, {}, True),
]

# (ipa, default spelling, height, backness, rounded, founding)
_BASE_VOWELS: list[tuple[str, str, Height, Backness, bool, bool]] = [
    ("i", "i", H.CLOSE, B.FRONT, False, True),
    ("y", "ü", H.CLOSE, B.FRONT, True, False),
    ("ɨ", "ï", H.CLOSE, B.CENTRAL, False, False),
    ("u", "u", H.CLOSE, B.BACK, True, True),
    ("e", "e", H.CLOSE_MID, B.FRONT, False, True),
    ("ø", "ö", H.CLOSE_MID, B.FRONT, True, False),
    ("o", "o", H.CLOSE_MID, B.BACK, True, True),
    ("ə", "ë", H.MID, B.CENTRAL, False, False),
    ("ɛ", "è", H.OPEN_MID, B.FRONT, False, False),
    ("ɔ", "ò", H.OPEN_MID, B.BACK, True, False),
    ("æ", "ä", H.NEAR_OPEN, B.FRONT, False, False),
    # The "a" of a five-vowel system is low *central* (strict IPA [a] is front). Treating it as
    # front would make rules like palatalization fire before a, which is rare.
    ("a", "a", H.OPEN, B.CENTRAL, False, True),
]

# Long and nasal variants exist for the five cardinal vowels, spelled with a macron / tilde.
_VARIANT_BASES = "aeiou"
_MACRON, _TILDE = "̄", "̃"


def _build_catalog() -> dict[str, Sound]:
    sounds: list[Sound] = []
    for ipa, grapheme, place, manner, voiced, flags, founding in _CONSONANTS:
        sounds.append(Sound(ipa, _C, grapheme, ConsonantFeatures(place, manner, voiced, **flags), founding))
    for ipa, grapheme, height, backness, rounded, founding in _BASE_VOWELS:
        sounds.append(Sound(ipa, _V, grapheme, VowelFeatures(height, backness, rounded), founding))
        if ipa in _VARIANT_BASES:
            sounds.append(Sound(ipa + "ː", _V, nfc(grapheme + _MACRON),
                                VowelFeatures(height, backness, rounded, long=True)))
            sounds.append(Sound(nfc(ipa + _TILDE), _V, nfc(grapheme + _TILDE),
                                VowelFeatures(height, backness, rounded, nasal=True)))

    catalog = {s.ipa: s for s in sounds}
    spellings = [s.default_grapheme for s in sounds]
    assert len(catalog) == len(sounds), "duplicate IPA symbol in the sound catalog"
    assert len(set(spellings)) == len(spellings), "duplicate default spelling in the sound catalog"
    return catalog


SOUNDS: dict[str, Sound] = _build_catalog()
# Default spelling -> IPA. Used to read words before a language (and its spelling) exists.
DEFAULT_SPELLINGS: dict[str, str] = {s.default_grapheme: s.ipa for s in SOUNDS.values()}
FOUNDING_SOUNDS: list[Sound] = [s for s in SOUNDS.values() if s.founding]

# Typing help: characters people type for a letter that is hard to type.
INPUT_ALIASES: dict[str, str] = {"'": "ʼ", "’": "ʼ"}


def sound(ipa: str) -> Sound:
    try:
        return SOUNDS[ipa]
    except KeyError:
        raise KeyError(f"unknown sound /{ipa}/") from None


def sound_for_grapheme(grapheme: str) -> Sound:
    """The sound a letter spells by default, e.g. "y" -> /j/."""
    try:
        return SOUNDS[DEFAULT_SPELLINGS[nfc(grapheme).lower()]]
    except KeyError:
        raise KeyError(f"no sound is spelled {grapheme!r} by default") from None
