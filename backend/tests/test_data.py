from app.data import load_example_requests, load_questions, load_semantic_inventory
from app.models.generation import GenerationRequest
from app.models.lexicon import ConceptPriority
from app.models.questionnaire import InputType


def test_data_files_have_no_byte_order_mark():
    # Regression: a BOM written by a Windows editor broke every questionnaire request.
    from app.data import DATA_DIR
    for path in DATA_DIR.rglob("*.json"):
        assert not path.read_bytes().startswith(b"\xef\xbb\xbf"), path.name


def test_semantic_inventory_size_and_unique_ids():
    concepts = load_semantic_inventory()
    ids = [c.id for c in concepts]
    assert len(ids) == len(set(ids))
    core = [c for c in concepts if c.priority == ConceptPriority.CORE]
    cultural = [c for c in concepts if c.priority == ConceptPriority.CULTURAL]
    assert len(core) >= 100
    assert 30 <= len(cultural) <= 120
    assert all(c.cultural_tags for c in cultural)


def test_concept_eras_are_valid():
    from app.models.eras import ERAS
    era_question = next(q for q in load_questions() if q.id == "era")
    assert [o.value for o in era_question.options] == list(ERAS)
    for c in load_semantic_inventory():
        assert c.min_era is None or c.min_era in ERAS, c.id


def test_question_ids_unique_and_maps_to_resolve():
    questions = load_questions()
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        section, field, *rest = q.maps_to.split(".")
        section_model = GenerationRequest.model_fields[section].annotation
        assert field in section_model.model_fields, q.id
        assert bool(rest) == (q.input_type == InputType.CUSTOM_WORD), q.id


def test_custom_word_questions_target_known_concepts():
    concept_ids = {c.id for c in load_semantic_inventory()}
    for q in load_questions():
        if q.input_type == InputType.CUSTOM_WORD:
            assert q.maps_to.rsplit(".", 1)[-1] in concept_ids


def test_cultural_tags_match_questionnaire_options():
    option_values = {o.value for q in load_questions() for o in q.options}
    tags = {t for c in load_semantic_inventory() for t in c.cultural_tags}
    assert tags <= option_values, tags - option_values


def test_example_requests_load_and_reference_known_concepts():
    concept_ids = {c.id for c in load_semantic_inventory()}
    requests = load_example_requests()
    assert len(requests) == 5
    for request in requests.values():
        assert set(request.language_preferences.custom_words) <= concept_ids
