import json
import unicodedata

import pytest
from pydantic import ValidationError

from app.generation.phoneme_generator import UnsupportedSpellingError, parse_spelling, render_phonemes
from app.generation.user_vocabulary import analyze_user_words
from app.models.language import LanguageState
from app.models.phonology import Phoneme, PhonemeClass
from app.models.sounds import (
    DEFAULT_SPELLINGS, FOUNDING_SOUNDS, SOUNDS, Backness, Manner, Place, sound, sound_for_grapheme,
)
from app.persistence.migrations import upgrade_language_state, upgrade_world
from app.validation.morphology_validator import validate_lexicon
from tests.factories import sample_language, sample_phonology


# --- catalog -------------------------------------------------------------------

def test_founding_alphabet_is_unchanged():
    """The design agent's alphabet and every old spelling must read exactly as before."""
    old = {"a": "a", "e": "e", "i": "i", "o": "o", "u": "u", "p": "p", "b": "b", "t": "t", "d": "d",
           "k": "k", "g": "ɡ", "q": "q", "m": "m", "n": "n", "f": "f", "v": "v", "s": "s", "z": "z",
           "h": "h", "x": "x", "l": "l", "r": "r", "w": "w", "y": "j", "j": "dʒ", "c": "ts", "sh": "ʃ",
           "ch": "tʃ", "zh": "ʒ", "th": "θ", "dh": "ð", "ng": "ŋ"}
    assert {s.default_grapheme: s.ipa for s in FOUNDING_SOUNDS} == old


def test_new_sounds_never_use_ascii_spellings():
    """Otherwise an existing ASCII word could be re-read as a new sound."""
    for s in SOUNDS.values():
        if not s.founding:
            assert not s.default_grapheme.isascii(), s
            assert len(s.default_grapheme) == 1, s  # one precomposed letter


def test_catalog_text_is_normalized():
    for s in SOUNDS.values():
        assert s.ipa == unicodedata.normalize("NFC", s.ipa)
        assert s.default_grapheme == unicodedata.normalize("NFC", s.default_grapheme).lower()


def test_features():
    k = sound("k").feature_values()
    assert k == {"class": "C", "place": Place.VELAR, "manner": Manner.STOP, "voiced": False,
                 "lateral": False, "sibilant": False}
    assert sound("tʃ").features.sibilant and sound("ɬ").features.lateral
    assert sound("aː").features.long and sound("ã").features.nasal
    assert sound("y").features.rounded and sound("y").features.backness == Backness.FRONT
    front = {s.ipa for s in SOUNDS.values() if s.phoneme_class == PhonemeClass.VOWEL
             and s.features.backness == Backness.FRONT and not s.features.long and not s.features.nasal}
    assert front == {"i", "y", "e", "ø", "ɛ", "æ"}


def test_ids_and_spellings_are_separate_namespaces():
    # The letter "y" spells /j/; the IPA /y/ is a front rounded vowel spelled "ü".
    assert sound_for_grapheme("y").ipa == "j"
    assert sound("y").default_grapheme == "ü"
    assert Phoneme.from_grapheme("c").id == "ts"
    assert Phoneme.from_ipa("c").grapheme == "ť"


# --- spelling ------------------------------------------------------------------

def test_diacritic_letters_parse_as_single_sounds():
    assert parse_spelling("ñāma") == ["ɲ", "aː", "m", "a"]
    assert parse_spelling("Ťõḷi") == ["c", "õ", "ɭ", "i"]
    # Decomposed input (n + combining tilde) reads the same as the precomposed letter.
    assert parse_spelling("ña") == ["ɲ", "a"]


def test_apostrophe_is_a_typing_alias_for_the_glottal_stop():
    assert parse_spelling("ka'a") == parse_spelling("kaʼa") == ["k", "a", "ʔ", "a"]


def test_each_language_has_its_own_spelling():
    # Same sound /ʃ/, spelled "š" here instead of the default "sh".
    phonology = sample_phonology(phonemes=[Phoneme.from_ipa("ʃ", grapheme="š"),
                                           *[Phoneme.from_grapheme(g) for g in "kria"]], restrictions=[])
    phonemes = parse_spelling("šira", phonology)
    assert phonemes == ["ʃ", "i", "r", "a"]
    assert render_phonemes(phonemes, phonology) == "šira"
    with pytest.raises(UnsupportedSpellingError):
        parse_spelling("shira", phonology)


@pytest.mark.parametrize("phoneme, message", [
    (Phoneme(id="ʃ", grapheme="Sh", ipa="ʃ", phoneme_class=PhonemeClass.CONSONANT), "lowercase"),
    (Phoneme(id="ʃ", grapheme="s-h", ipa="ʃ", phoneme_class=PhonemeClass.CONSONANT), "hyphens"),
    (Phoneme(id="ʃ", grapheme="sh", ipa="s", phoneme_class=PhonemeClass.CONSONANT), "must match"),
    (Phoneme(id="sh", grapheme="sh", ipa="sh", phoneme_class=PhonemeClass.CONSONANT), "unknown sound"),
])
def test_phonology_rejects_bad_phonemes(phoneme, message):
    with pytest.raises(ValidationError, match=message):
        sample_phonology(phonemes=[phoneme, Phoneme.from_grapheme("a")], restrictions=[])


def test_user_words_may_use_new_sounds():
    reqs, result = analyze_user_words({"water": "ñāma"})
    assert result.ok
    assert set(reqs.required_phonemes) == {"ɲ", "aː", "m", "a"}
    assert reqs.required_letters == sorted(["ñ", "ā", "m", "a"])


# --- migration -----------------------------------------------------------------

def _legacy(state_json: dict) -> dict:
    """Rewrite a current state into the old format, where a phoneme's id was its letter."""
    to_letter = {p["id"]: p["grapheme"] for p in state_json["version"]["phonology"]["phonemes"]}
    for p in state_json["version"]["phonology"]["phonemes"]:
        p["id"] = p["grapheme"]
    for r in state_json["version"]["phonology"]["restrictions"]:
        r["phoneme_id"] = to_letter[r["phoneme_id"]]
    for m in state_json["morphemes"]:
        m["phonemes"] = [to_letter[p] for p in m["phonemes"]]
    for f in state_json["forms"]:
        f["phonemes"] = [to_letter[p] for p in f["phonemes"]]
        for seg in f["segments"]:
            seg["phonemes"] = [to_letter[p] for p in seg["phonemes"]]
    return state_json


def _with_tricky_sounds(state: LanguageState) -> LanguageState:
    """Add sounds whose old letter id is another sound's new IPA id (y -> j, j -> dʒ) and a pre-digraph c."""
    ph = state.version.phonology
    extra = [Phoneme.from_grapheme("y"), Phoneme.from_grapheme("j"),
             Phoneme(id="tʃ", grapheme="c", ipa="tʃ", phoneme_class=PhonemeClass.CONSONANT)]
    version = state.version.model_copy(update={"phonology": ph.model_copy(update={"phonemes": [*ph.phonemes, *extra]})})
    root = next(m for m in state.morphemes if m.type.value == "root")
    morphemes = [m.model_copy(update={"phonemes": ["j", "a", "dʒ", "a", "tʃ"]}) if m.id == root.id else m
                 for m in state.morphemes]
    return state.model_copy(update={"version": version, "morphemes": morphemes})


def test_legacy_language_migrates_to_ipa_ids():
    current = _with_tricky_sounds(sample_language())
    legacy = _legacy(json.loads(current.model_dump_json()))
    assert "y" in [p["id"] for p in legacy["version"]["phonology"]["phonemes"]]

    upgraded = LanguageState.model_validate(upgrade_language_state(legacy))
    assert upgraded == current
    # The old c that meant /tʃ/ keeps its spelling.
    assert upgraded.version.phonology.phoneme("tʃ").grapheme == "c"


def test_migration_is_a_no_op_on_current_data():
    state = json.loads(sample_language().model_dump_json())
    assert upgrade_language_state(json.loads(json.dumps(state))) == state


def test_migrated_sample_language_still_validates():
    legacy = _legacy(json.loads(sample_language().model_dump_json()))
    upgraded = LanguageState.model_validate(upgrade_language_state(legacy))
    assert validate_lexicon(upgraded).ok


def test_world_sketches_migrate():
    world = {"language_sketches": [{"phonology": {"phonemes": [
        {"id": "g", "grapheme": "g", "ipa": "ɡ", "phoneme_class": "C"},
        {"id": "a", "grapheme": "a", "ipa": "a", "phoneme_class": "V"}], "restrictions": []},
        "words": [{"phonemes": ["g", "a"]}]}]}
    sketch = upgrade_world(world)["language_sketches"][0]
    assert [p["id"] for p in sketch["phonology"]["phonemes"]] == ["ɡ", "a"]
    assert sketch["words"][0]["phonemes"] == ["ɡ", "a"]


def test_default_spellings_cover_the_catalog():
    assert set(DEFAULT_SPELLINGS.values()) == set(SOUNDS)
