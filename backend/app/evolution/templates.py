"""Named, well-attested kinds of sound change, as ready-made rules.

Each template returns an ordinary `SoundChange`, so the evolution agent can
use one as-is, adjust its parameters, or write a custom rule in the same format.
The templates encode typology (what usually happens), not word lists.
"""

from app.models.sound_change import ContextItem, SoundChange, SoundMatcher

VOWEL = SoundMatcher(features={"class": "V"})
CONSONANT = SoundMatcher(features={"class": "C"})
FRONT_VOWEL = SoundMatcher(features={"class": "V", "backness": "front"})
CLOSE_FRONT_VOWEL = SoundMatcher(ipa=["i", "iː", "ĩ", "y"])


def _seg(matcher: SoundMatcher) -> ContextItem:
    return ContextItem(kind="segment", matcher=matcher)


BOUNDARY = ContextItem(kind="boundary")
ANY_CONSONANTS = ContextItem(kind="consonants")


def palatalization(close_vowels_only: bool = False, include_voiced: bool = True) -> SoundChange:
    """Velar stops become postalveolar affricates before front vowels: k -> tʃ, ɡ -> dʒ (Latin c > Italian ci)."""
    features = {"place": "velar", "manner": "stop"}
    if not include_voiced:
        features["voiced"] = False
    return SoundChange(
        name="Velar palatalization",
        description="k and g become ch and j before " + ("i" if close_vowels_only else "front vowels"),
        target=SoundMatcher(features=features),
        output_features={"place": "postalveolar", "manner": "affricate"},
        right=[_seg(CLOSE_FRONT_VOWEL if close_vowels_only else FRONT_VOWEL)],
    )


def intervocalic_voicing() -> SoundChange:
    """Voiceless stops become voiced between vowels: p t k -> b d ɡ (Latin lacu > Spanish lago)."""
    return SoundChange(
        name="Intervocalic voicing",
        description="p, t, k become b, d, g between vowels",
        target=SoundMatcher(features={"manner": "stop", "voiced": False}),
        output_features={"voiced": True},
        left=[_seg(VOWEL)], right=[_seg(VOWEL)],
    )


def spirantization(voiced_only: bool = True) -> SoundChange:
    """Stops become fricatives between vowels: b d ɡ -> β ð ɣ (as in Spanish)."""
    mapping = {"b": "β", "d": "ð", "ɡ": "ɣ"}
    if not voiced_only:
        mapping |= {"p": "f", "t": "θ", "k": "x"}
    return SoundChange(
        name="Intervocalic spirantization",
        description="stops soften into fricatives between vowels",
        output_mapping=mapping, left=[_seg(VOWEL)], right=[_seg(VOWEL)],
    )


def final_vowel_loss(after_single_consonant: bool = False) -> SoundChange:
    """Word-final vowels drop in words of two or more syllables (Latin lupu > Catalan llop).

    `after_single_consonant` keeps the vowel after a cluster (ojenro stays, kijuri -> kijur),
    which is how many languages avoid ending words in hard clusters.
    """
    return SoundChange(
        name="Final vowel loss",
        description="vowels at the end of a word disappear, except in one-syllable words"
                    + (" and after two or more consonants" if after_single_consonant else ""),
        target=SoundMatcher(features={"class": "V", "long": False}), delete=True,
        # "an earlier vowel exists" = a vowel, then consonants, then the target
        left=[_seg(VOWEL), _seg(CONSONANT) if after_single_consonant else ANY_CONSONANTS], right=[BOUNDARY],
    )


def i_umlaut() -> SoundChange:
    """Back vowels move forward when the next syllable has i: u o a -> y ø e (Germanic *mūsiz > mice)."""
    return SoundChange(
        name="i-umlaut",
        description="u, o, a become ü, ö, e when the next vowel is i",
        output_mapping={"u": "y", "o": "ø", "a": "e"},
        right=[ANY_CONSONANTS, _seg(CLOSE_FRONT_VOWEL)],
    )


def final_devoicing() -> SoundChange:
    """Voiced obstruents become voiceless at the end of a word: d -> t, z -> s (German Rad [ʁaːt])."""
    # An explicit mapping, because only obstruents devoice: m, n, l, r have no voiceless partner.
    return SoundChange(
        name="Final devoicing",
        description="b, d, g, v, z and similar become voiceless at the end of a word",
        output_mapping={"b": "p", "d": "t", "ɡ": "k", "v": "f", "z": "s", "ʒ": "ʃ", "dʒ": "tʃ", "ð": "θ",
                        "β": "p", "ɣ": "x"},
        right=[BOUNDARY],
    )


def h_loss() -> SoundChange:
    """/h/ disappears everywhere (Latin habere > Spanish haber [aˈβeɾ])."""
    return SoundChange(name="Loss of h", description="h is no longer pronounced",
                       output_mapping={"h": None})


SONORANT = SoundMatcher(features={"class": "C", "manner": ["nasal", "trill", "approximant"]})
S_LIKE = SoundMatcher(features={"class": "C", "sibilant": True, "manner": "fricative"})


def cluster_epenthesis(vowel: str = "e", sonorants_only: bool = True) -> SoundChange:
    """A vowel breaks up a word-final consonant cluster: ojenr -> ojener (Latin ager, from *agr-).

    With `sonorants_only` (default) only clusters ending in a nasal, liquid or glide are broken,
    which are the ones that violate sonority; otherwise every final two-consonant cluster is.
    """
    return SoundChange(
        name="Final cluster epenthesis",
        description=f"{vowel} is inserted between two consonants at the end of a word"
                    + (" when the last is l, r, n, m, w or y" if sonorants_only else ""),
        insert=vowel,
        left=[_seg(CONSONANT)], right=[_seg(SONORANT if sonorants_only else CONSONANT), BOUNDARY],
    )


def prothesis(vowel: str = "e") -> SoundChange:
    """A vowel appears before word-initial s + consonant: skola -> eskola (Latin schola > Spanish escuela)."""
    return SoundChange(
        name="Prothesis",
        description=f"{vowel} is added before s + consonant at the start of a word",
        insert=vowel, left=[BOUNDARY], right=[_seg(S_LIKE), _seg(CONSONANT)],
    )


def merger(source: str, into: str) -> SoundChange:
    """Two sounds fall together everywhere: e.g. ɛ -> e. Mergers are irreversible."""
    return SoundChange(name=f"Merger of {source} into {into}",
                       description=f"{source} is pronounced as {into} everywhere",
                       output_mapping={source: into})


TEMPLATES = {
    "palatalization": palatalization,
    "intervocalic_voicing": intervocalic_voicing,
    "spirantization": spirantization,
    "final_vowel_loss": final_vowel_loss,
    "final_vowel_loss_after_single_consonant": lambda: final_vowel_loss(after_single_consonant=True),
    "i_umlaut": i_umlaut,
    "final_devoicing": final_devoicing,
    "h_loss": h_loss,
    "cluster_epenthesis": cluster_epenthesis,
    "prothesis": prothesis,
}
