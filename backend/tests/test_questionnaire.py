import random

import pytest

from app.data import load_questions, load_semantic_inventory
from app.models.grammar import WordOrder
from app.models.questionnaire import InputType, Question
from app.questionnaire import (
    NewConcept,
    QuestionnaireError,
    build_generation_request,
    closest_option,
    select_questions,
)

CONCEPTS = {c.id for c in load_semantic_inventory()}
BANK = load_questions()


def build(answers, questions=BANK, custom=None, new=()):
    return build_generation_request(questions, answers, seed=1, known_concept_ids=CONCEPTS,
                                    custom_answers=custom, new_concepts=new)


ALWAYS = {"civilization_name", "era", "language_notes"}


def test_select_questions_picks_five_random_plus_always_asked():
    first = select_questions(BANK, random.Random(3))
    second = select_questions(BANK, random.Random(3))
    assert len(first) == 8 and ALWAYS <= {q.id for q in first}
    assert [q.id for q in first] == [q.id for q in second]
    assert first[0].id == "civilization_name" and first[-1].id == "language_notes"


def test_select_questions_varies_with_seed():
    selections = {tuple(q.id for q in select_questions(BANK, random.Random(s))) for s in range(20)}
    assert len(selections) > 10


def test_select_questions_always_includes_required():
    required = Question(id="name", category="civilization", prompt="Name?",
                        input_type=InputType.FREE_TEXT, maps_to="civilization.name", required=True)
    for seed in range(10):
        chosen = select_questions([*BANK, required], random.Random(seed))
        assert "name" in {q.id for q in chosen} and len(chosen) == 9


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


def test_name_and_notes():
    request = build({"civilization_name": " Savari ", "language_notes": "Soft and watery, lots of sea words"})
    assert request.civilization.name == "Savari"
    assert request.language_preferences.notes == "Soft and watery, lots of sea words"


@pytest.mark.parametrize("question, text, expected", [
    ("geography", "Island-hopping traders on a warm sea", "island"),
    ("geography", "They live among the mountains", "mountain"),
    ("word_order", "Mostly SOV, but flexible for emphasis", "SOV"),
    ("geography", "Floating cities in the clouds", None),
])
def test_describe_your_own_picks_a_clear_option(question, text, expected):
    q = next(q for q in BANK if q.id == question)
    assert closest_option(q, text) == expected


def test_custom_answers_reach_the_agents():
    request = build({}, custom={"geography": "Island-hopping traders", "word_order": "free word order",
                                "cultural_traits": "obsessed with poetry"})
    assert request.civilization.geography == "island"            # clearly named -> option chosen
    assert request.language_preferences.word_order is None       # not an option -> the AI decides
    assert "Island-hopping traders" in request.civilization.description
    assert "obsessed with poetry" in request.civilization.description
    assert "free word order" in request.language_preferences.notes
    assert request.custom_answers["civilization.geography"] == "Island-hopping traders"


@pytest.mark.parametrize("text, era, year", [
    ("around 1500 BCE", "bronze_age", -1500),
    ("They rise in 700 BC", "iron_age", -700),
    ("1200 AD, a kingdom of castles", "medieval", 1200),
    ("Late in the iron age", "iron_age", None),   # no year: era from the text, typical year later
])
def test_era_from_written_answer(text, era, year):
    request = build({}, custom={"era": text})
    assert request.civilization.era == era
    assert request.civilization.founding_year == year


def test_custom_answer_only_for_multiple_choice():
    with pytest.raises(QuestionnaireError):
        build({}, custom={"word_water": "something"})


def test_new_concepts_become_dictionary_meanings():
    request = build({}, new=[NewConcept(gloss="Moonlight festival", part_of_speech="noun", word="lunara"),
                             NewConcept(gloss="water", part_of_speech="noun", word="mari")])
    prefs = request.language_preferences
    assert prefs.custom_words == {"user_moonlight_festival": "lunara", "water": "mari"}  # existing concept reused
    [concept] = prefs.custom_concepts
    assert (concept.id, concept.gloss, concept.priority.value) == ("user_moonlight_festival", "Moonlight festival", "user")


def test_new_concept_conflicting_with_another_answer():
    with pytest.raises(QuestionnaireError) as exc:
        build({"word_water": "mari"}, new=[NewConcept(gloss="water", part_of_speech="noun", word="suli")])
    assert exc.value.result.errors[0].code == "CONFLICTING_CUSTOM_WORD"


def test_answer_to_question_not_asked_is_rejected():
    asked = [q for q in BANK if q.id == "geography"]
    with pytest.raises(QuestionnaireError):
        build({"word_order": "SOV"}, questions=asked)
