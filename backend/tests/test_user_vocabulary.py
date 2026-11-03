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


def test_digraph_words_are_supported():
    reqs, result = analyze_user_words({"water": "shira", "song": "ngatha"})
    assert result.ok
    assert {"ʃ", "ŋ", "θ"} <= set(reqs.required_phonemes)
    assert {"sh", "ng", "th"} <= set(reqs.required_letters)
    assert reqs.words[0].syllable_patterns == ["CV", "CV"]


def test_vowel_initial_and_hiatus_detected():
    reqs, result = analyze_user_words({"song": "aria"})
    assert result.ok
    assert reqs.requires_hiatus
    assert reqs.words[0].syllable_patterns == ["V", "CV", "V"]


def test_spelling_preserved_exactly():
    reqs, _ = analyze_user_words({"water": "Mari"})
    assert reqs.words[0].spelling == "Mari"


def test_unsupported_words_reported():
    _, result = analyze_user_words({"oasis": "rtak", "sea": "nal!"})
    assert sorted(i.code for i in result.errors) == ["UNSUPPORTED_GRAPHEME", "UNSUPPORTED_SYLLABLE_STRUCTURE"]


def test_same_word_for_two_meanings_is_allowed():
    # Intentional polysemy (kijur = zinc, wealth): noted, not an error.
    _, result = analyze_user_words({"water": "mari", "sea": "mari"})
    assert result.ok and [i.code for i in result.issues] == ["USER_POLYSEMY"]


def test_conflicting_fixture_words_are_legal_but_need_s_clusters():
    # "strak" breaks sonority (s before t) only in the way s-clusters are allowed to.
    # The fixture's conflict ("never use s" vs. words with s) is for the design agent to resolve.
    request = load_example_requests()["conflicting_constraints"]
    reqs, result = analyze_user_words(request.language_preferences.custom_words)
    assert result.ok and reqs.requires_s_clusters
    assert "CCCVC" in reqs.required_patterns


def test_cluster_words_are_supported_when_they_follow_sonority():
    reqs, result = analyze_user_words({"bread": "plano", "mountain": "kalpa", "wolf": "vulk"})
    assert result.ok and not reqs.requires_s_clusters
    assert {"CCV", "CVCC"} <= set(reqs.required_patterns)
    _, bad = analyze_user_words({"x": "lpana"})
    assert [i.code for i in bad.errors] == ["UNSUPPORTED_SYLLABLE_STRUCTURE"]
    assert "cannot begin a syllable" in bad.errors[0].message


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
