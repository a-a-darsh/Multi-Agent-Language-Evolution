import pytest

from app.models.phonology import Phoneme, Position, PositionRestriction
from app.validation.phonology_validator import syllabify, validate_phoneme_sequence
from tests.factories import sample_phonology


def codes(phonemes, phonology):
    return [i.code for i in validate_phoneme_sequence(list(phonemes), phonology).errors]


@pytest.mark.parametrize("form", ["sava", "nal", "nalka", "tamirok"])
def test_legal_forms(form):
    assert codes(form, sample_phonology()) == []


def test_valid_letters_but_illegal_structure():
    # Every consonant exists, but CV/CVC can't parse a cluster.
    phonology = sample_phonology(phonemes=[Phoneme.from_grapheme(g) for g in "strka"], restrictions=[])
    assert codes("strk", phonology) == [
        "ILLEGAL_SYLLABLE_STRUCTURE"
    ]


def test_unknown_phoneme():
    assert codes("bala", sample_phonology()) == ["UNKNOWN_PHONEME"]


def test_empty_form():
    assert codes("", sample_phonology()) == ["EMPTY_FORM"]


def test_onsetless_syllables_need_v_patterns():
    assert codes("ama", sample_phonology()) == ["ILLEGAL_SYLLABLE_STRUCTURE"]
    assert codes("ama", sample_phonology(syllable_patterns=["V", "CV", "CVC"])) == []


def test_hiatus_requires_permission():
    phonology = sample_phonology(syllable_patterns=["V", "CV"])
    assert "HIATUS_NOT_ALLOWED" in codes("ria", phonology)
    assert codes("ria", sample_phonology(syllable_patterns=["V", "CV"], allow_hiatus=True)) == []


def test_coda_restriction():
    # Sample phonology forbids /v/ in codas.
    assert codes("tav", sample_phonology()) == ["POSITION_RESTRICTION"]
    assert codes("tava", sample_phonology()) == []


def test_restriction_passes_if_any_parse_satisfies_it():
    # "akat" parses as a.kat or ak.at; forbidding /k/ in codas leaves a.kat legal.
    phonology = sample_phonology(
        syllable_patterns=["V", "VC", "CV", "CVC"],
        restrictions=[PositionRestriction(phoneme_id="k", forbidden_positions=[Position.CODA])],
    )
    assert codes("akat", phonology) == []
    assert codes("ak", phonology) == ["POSITION_RESTRICTION"]


def test_word_edge_restrictions():
    phonology = sample_phonology(restrictions=[
        PositionRestriction(phoneme_id="r", forbidden_positions=[Position.WORD_INITIAL]),
        PositionRestriction(phoneme_id="m", forbidden_positions=[Position.WORD_FINAL]),
    ])
    assert codes("rasa", phonology) == ["POSITION_RESTRICTION"]
    assert codes("sam", phonology) == ["POSITION_RESTRICTION"]
    assert codes("sara", phonology) == []


def test_spelling_must_read_back_to_the_same_sounds():
    phonology = sample_phonology(phonemes=[Phoneme.from_grapheme(g) for g in ["s", "h", "sh", "a", "t"]],
                                 syllable_patterns=["V", "CV", "CVC"], restrictions=[])
    # tas.ha: s (coda) + h (onset) would be written "tasha" and read back as ta + sha.
    assert codes(["t", "a", "s", "h", "a"], phonology) == ["AMBIGUOUS_SPELLING"]
    assert codes(["t", "a", "sh", "a"], phonology) == []
    assert codes(["t", "a", "s", "t", "a"], phonology) == []


def test_syllabify_uses_maximal_onset():
    phonology = sample_phonology(syllable_patterns=["V", "VC", "CV", "CVC"])
    assert [s.pattern for s in syllabify(list("aka"), phonology)] == ["V", "CV"]
    assert ["".join(s.phonemes) for s in syllabify(list("nalka"), phonology)] == ["nal", "ka"]


def test_syllabify_returns_none_for_illegal():
    assert syllabify(list("ama"), sample_phonology()) is None
