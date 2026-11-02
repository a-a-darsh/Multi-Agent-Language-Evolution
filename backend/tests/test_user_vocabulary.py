from app.data import load_example_requests
from app.generation.user_vocabulary import analyze_user_words, check_phonology_satisfies
from app.models.phonology import Phoneme
from tests.factories import sample_phonology


def test_requirements_inferred_from_words():
    reqs, result = analyze_user_words({"hello": "sava", "water": "mari"})
    assert result.ok
    assert reqs.required_phonemes == ["a", "i", "m", "r", "s", "v"]
    assert reqs.required_patterns == ["CV"]
    assert not reqs.requires_hiatus


def test_vowel_initial_and_hiatus_detected():
    reqs, result = analyze_user_words({"song": "aria"})
    assert result.ok
    assert reqs.requires_hiatus
    assert reqs.words[0].syllable_patterns == ["V", "CV", "V"]


def test_spelling_preserved_exactly():
    reqs, _ = analyze_user_words({"water": "Mari"})
    assert reqs.words[0].spelling == "Mari"


def test_unsupported_words_reported():
    _, result = analyze_user_words({"oasis": "strak", "sea": "nal!"})
    assert sorted(i.code for i in result.errors) == ["UNSUPPORTED_GRAPHEME", "UNSUPPORTED_SYLLABLE_STRUCTURE"]


def test_duplicate_user_forms_reported():
    _, result = analyze_user_words({"water": "mari", "sea": "mari"})
    assert [i.code for i in result.errors] == ["DUPLICATE_USER_FORM"]


def test_conflicting_fixture_is_caught():
    request = load_example_requests()["conflicting_constraints"]
    _, result = analyze_user_words(request.language_preferences.custom_words)
    assert [i.object_id for i in result.errors] == ["oasis"]


def test_phonology_must_accommodate_user_words():
    reqs, _ = analyze_user_words({"hello": "sava", "mother": "ama", "tool": "bek"})
    issues = check_phonology_satisfies(sample_phonology(), reqs).errors
    causes = {i.object_id: i.details["cause"] for i in issues}
    assert causes == {"mother": "ILLEGAL_SYLLABLE_STRUCTURE", "tool": "UNKNOWN_PHONEME"}

    fixed = sample_phonology(
        phonemes=[Phoneme.from_grapheme(g) for g in "ptkbmnslrvaeiou"],
        syllable_patterns=["V", "CV", "CVC"],
    )
    assert check_phonology_satisfies(fixed, reqs).ok
