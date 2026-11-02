import random

import pytest

from app.data import load_questions, load_semantic_inventory
from app.models.grammar import WordOrder
from app.models.questionnaire import InputType, Question
from app.questionnaire import QuestionnaireError, build_generation_request, select_questions

CONCEPTS = {c.id for c in load_semantic_inventory()}
BANK = load_questions()


def build(answers, questions=BANK):
    return build_generation_request(questions, answers, seed=1, known_concept_ids=CONCEPTS)


def test_select_questions_picks_five_reproducibly():
    first = select_questions(BANK, random.Random(3))
    second = select_questions(BANK, random.Random(3))
    assert len(first) == 5
    assert [q.id for q in first] == [q.id for q in second]


def test_select_questions_varies_with_seed():
    selections = {tuple(q.id for q in select_questions(BANK, random.Random(s))) for s in range(20)}
    assert len(selections) > 10


def test_select_questions_always_includes_required():
    required = Question(id="name", category="civilization", prompt="Name?",
                        input_type=InputType.FREE_TEXT, maps_to="civilization.name", required=True)
    for seed in range(10):
        chosen = select_questions([*BANK, required], random.Random(seed))
        assert "name" in {q.id for q in chosen} and len(chosen) == 5


def test_answers_map_into_request():
    request = build({
        "geography": "coastal",
        "cultural_traits": ["seafaring", "commercial"],
        "word_order": "SOV",
        "case_marking": True,
        "special_features": "  verbs mark evidentiality  ",
        "word_water": "mari",
        "custom_words": {"sea": "nal"},
    })
    assert request.civilization.geography == "coastal"
    assert request.language_preferences.word_order == WordOrder.SOV
    assert request.language_preferences.case_marking is True
    assert request.language_preferences.special_features == ["verbs mark evidentiality"]
    assert request.language_preferences.custom_words == {"water": "mari", "sea": "nal"}
    assert request.seed == 1


def test_skipped_and_blank_answers_mean_ai_decides():
    request = build({"geography": "", "civilization_description": "   "})
    assert request.civilization.geography is None
    assert request.user_specified_fields() == []


def test_all_problems_reported_together():
    with pytest.raises(QuestionnaireError) as exc:
        build({"geography": "moon", "custom_words": {"spaceship": "zor"}, "favorite_color": "red"})
    codes = sorted(i.code for i in exc.value.result.errors)
    assert codes == ["INVALID_ANSWER", "INVALID_ANSWER", "UNKNOWN_QUESTION"]


def test_conflicting_custom_words_are_rejected_not_overwritten():
    with pytest.raises(QuestionnaireError) as exc:
        build({"word_water": "mari", "custom_words": {"water": "suli"}})
    [issue] = exc.value.result.errors
    assert issue.code == "CONFLICTING_CUSTOM_WORD"
    assert issue.details["spellings"] == ["mari", "suli"]


def test_answer_to_question_not_asked_is_rejected():
    asked = [q for q in BANK if q.id == "geography"]
    with pytest.raises(QuestionnaireError):
        build({"word_order": "SOV"}, questions=asked)
