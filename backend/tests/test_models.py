import pytest
from pydantic import ValidationError

from app.models.generation import GenerationRequest
from app.models.language import LanguageState
from app.models.lexicon import LexemeOrigin, Lexeme, LexicalSense, PartOfSpeech
from app.models.phonology import Phoneme, PhonemeClass, Position, PositionRestriction
from tests.factories import sample_language, sample_phonology


def test_sample_language_round_trips_through_json():
    state = sample_language()
    restored = LanguageState.model_validate_json(state.model_dump_json())
    assert restored == state


def test_sample_language_derivation_links_to_base():
    state = sample_language()
    sailor = state.lexeme("lex_sailor")
    base = state.lexeme(sailor.derivation.base_lexeme_id)
    assert base.concept_id == "sea"
    assert state.form_of(sailor.id).orthography == "nalka"
    # The form is reconstructible from its morphemes, not just annotated.
    assembled = [p for m_id in sailor.morpheme_ids for p in state.morpheme(m_id).phonemes]
    assert assembled == state.form_of(sailor.id).phonemes


def test_phoneme_from_catalog():
    y = Phoneme.from_grapheme("y")
    assert y.ipa == "j" and y.phoneme_class == PhonemeClass.CONSONANT


@pytest.mark.parametrize("overrides, message", [
    ({"syllable_patterns": ["CCCCV"]}, "unsupported syllable patterns"),
    ({"phonemes": [Phoneme.from_grapheme("p")]}, "at least one vowel"),
    ({"phonemes": [Phoneme.from_grapheme("a"), Phoneme.from_grapheme("a")]}, "duplicate"),
    ({"restrictions": [PositionRestriction(phoneme_id="z", forbidden_positions=[Position.CODA])]},
     "unknown phoneme"),
])
def test_invalid_phonology_rejected(overrides, message):
    with pytest.raises(ValidationError, match=message):
        sample_phonology(**overrides)


def test_phonology_rejects_grapheme_with_wrong_class():
    bad = Phoneme(id="a", grapheme="a", ipa="a", phoneme_class=PhonemeClass.CONSONANT)
    with pytest.raises(ValidationError, match="must be class"):
        sample_phonology(phonemes=[bad, Phoneme.from_grapheme("e")])


def test_phonology_rejects_unfillable_coda():
    with pytest.raises(ValidationError, match="every consonant is forbidden in codas"):
        sample_phonology(
            phonemes=[Phoneme.from_grapheme(g) for g in "pa"],
            restrictions=[PositionRestriction(phoneme_id="p", forbidden_positions=[Position.CODA])],
        )


def test_derived_lexeme_requires_derivation():
    with pytest.raises(ValidationError, match="derivation"):
        Lexeme(language_id="x", concept_id="sailor", part_of_speech=PartOfSpeech.NOUN,
               origin=LexemeOrigin.DERIVED, morpheme_ids=["m"],
               senses=[LexicalSense(concept_id="sailor", gloss="sailor")])


def test_models_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        Phoneme(id="a", grapheme="a", ipa="a", phoneme_class=PhonemeClass.VOWEL, tone="high")


def test_user_specified_fields_tracks_explicit_inputs():
    request = GenerationRequest.model_validate({
        "civilization": {"geography": "coastal"},
        "language_preferences": {"word_order": "SOV", "custom_words": {"water": "mari"}},
    })
    assert request.user_specified_fields() == [
        "civilization.geography",
        "language_preferences.word_order",
        "language_preferences.custom_words",
    ]
    assert isinstance(request.seed, int)
