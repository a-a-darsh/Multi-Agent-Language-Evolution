"""Turns questionnaire answers into a validated GenerationRequest.

Each session asks a random subset of the question bank. Questions that were
not asked (or were skipped) leave their field empty, which means "AI decides".
"""

import random
from collections.abc import Iterable, Mapping
from typing import Any, get_args, get_origin

from app.models.generation import GenerationRequest
from app.models.questionnaire import InputType, Question
from app.models.validation import ValidationResult

DEFAULT_QUESTION_COUNT = 5


class QuestionnaireError(Exception):
    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__("; ".join(i.message for i in result.errors))


def select_questions(
    bank: Iterable[Question], rng: random.Random, count: int = DEFAULT_QUESTION_COUNT
) -> list[Question]:
    """Pick `count` questions: all required ones, the rest at random.

    Returned in bank order so the questionnaire reads naturally.
    """
    bank = list(bank)
    required = [q for q in bank if q.required]
    optional = [q for q in bank if not q.required]
    remaining = max(count - len(required), 0)
    chosen = {q.id for q in required} | {q.id for q in rng.sample(optional, min(remaining, len(optional)))}
    return [q for q in bank if q.id in chosen]


def build_generation_request(
    questions: Iterable[Question],
    answers: Mapping[str, Any],
    seed: int,
    known_concept_ids: set[str],
) -> GenerationRequest:
    """Validate answers against the asked questions and build the request.

    Raises QuestionnaireError listing every problem at once, so the UI can
    show them together instead of one per submit.
    """
    questions_by_id = {q.id: q for q in questions}
    result = ValidationResult()
    data: dict[str, Any] = {"civilization": {}, "language_preferences": {}, "seed": seed}

    for question_id in answers:
        if question_id not in questions_by_id:
            result.add("UNKNOWN_QUESTION", f"Answer given for unknown question {question_id!r}",
                       object_id=question_id)

    for question in questions_by_id.values():
        answer = _normalize(answers.get(question.id))
        if answer is None:
            if question.required:
                result.add("MISSING_REQUIRED_ANSWER", f"Question {question.id!r} is required",
                           object_id=question.id)
            continue
        problem = _check_answer(question, answer, known_concept_ids)
        if problem:
            result.add("INVALID_ANSWER", problem, object_id=question.id, details={"answer": answer})
            continue
        for concept_id, existing, new in _store(data, question, answer):
            result.add("CONFLICTING_CUSTOM_WORD",
                       f"Two different words were given for {concept_id!r}: {existing!r} and {new!r}",
                       object_id=concept_id, details={"spellings": [existing, new]})

    if not result.ok:
        raise QuestionnaireError(result)
    return GenerationRequest.model_validate(data)


def _normalize(answer: Any) -> Any:
    """Treat blank answers as skipped. Only surrounding whitespace is trimmed."""
    if isinstance(answer, str):
        answer = answer.strip()
    if answer in (None, "", [], {}):
        return None
    return answer


def _check_answer(question: Question, answer: Any, known_concept_ids: set[str]) -> str | None:
    option_values = {o.value for o in question.options}
    match question.input_type:
        case InputType.SINGLE_SELECT:
            if answer not in option_values:
                return f"{answer!r} is not an option for {question.id!r}"
        case InputType.MULTI_SELECT:
            if not isinstance(answer, list) or not set(answer) <= option_values:
                return f"{answer!r} contains values that are not options for {question.id!r}"
        case InputType.FREE_TEXT | InputType.CUSTOM_WORD:
            if not isinstance(answer, str):
                return f"{question.id!r} expects text"
        case InputType.YES_NO:
            if not isinstance(answer, bool):
                return f"{question.id!r} expects yes or no"
        case InputType.CUSTOM_VOCABULARY:
            if not isinstance(answer, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in answer.items()
            ):
                return f"{question.id!r} expects meaning -> word pairs"
            unknown = sorted(set(answer) - known_concept_ids)
            if unknown:
                return f"unknown meanings {unknown}"
    if question.input_type == InputType.CUSTOM_WORD:
        concept_id = question.maps_to.rsplit(".", 1)[-1]
        if concept_id not in known_concept_ids:
            return f"question {question.id!r} maps to unknown meaning {concept_id!r}"
    return None


def _store(data: dict[str, Any], question: Question, answer: Any) -> list[tuple[str, str, str]]:
    """Write the answer into `data`. Returns (concept, existing, new) word conflicts."""
    section, field, *rest = question.maps_to.split(".")
    target = data[section]
    if question.input_type in (InputType.CUSTOM_WORD, InputType.CUSTOM_VOCABULARY):
        # CUSTOM_WORD maps to e.g. "language_preferences.custom_words.water"
        pairs = {rest[0]: answer} if question.input_type == InputType.CUSTOM_WORD else answer
        words = target.setdefault(field, {})
        conflicts = []
        for concept_id, spelling in pairs.items():
            concept_id, spelling = concept_id.strip(), spelling.strip()
            if concept_id in words and words[concept_id] != spelling:
                conflicts.append((concept_id, words[concept_id], spelling))
            else:
                words[concept_id] = spelling
        return conflicts
    if _is_list_field(section, field) and not isinstance(answer, list):
        # A free-text answer feeding a list field, e.g. special_features.
        target.setdefault(field, []).append(answer)
    else:
        target[field] = answer
    return []


def _is_list_field(section: str, field: str) -> bool:
    section_model = GenerationRequest.model_fields[section].annotation
    annotation = section_model.model_fields[field].annotation
    return get_origin(annotation) is list or any(get_origin(a) is list for a in get_args(annotation))
