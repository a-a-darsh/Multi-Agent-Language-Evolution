"""Turns questionnaire answers into a validated GenerationRequest.

Each session asks the always-asked questions (e.g. the civilization's name and
final notes) plus a random subset of the rest. Questions that were not asked,
or were skipped, leave their field empty, which means "AI decides".

Multiple-choice questions also accept "describe your own" text. That text is
passed to the agents verbatim; when it clearly names one of the options (e.g.
"island-hopping traders" -> island), that option is also selected, because the
procedural world generator needs a category to work with.
"""

import random
import re
from collections.abc import Iterable, Mapping
from typing import Any, get_args, get_origin

from pydantic import BaseModel, Field

from app.models.eras import era_for_year, parse_year
from app.models.generation import GenerationRequest
from app.models.lexicon import ConceptPriority, PartOfSpeech, SemanticConcept
from app.models.questionnaire import InputType, Question
from app.models.validation import ValidationResult

DEFAULT_QUESTION_COUNT = 5
MAX_CUSTOM_ANSWER = 300
SELECTS = (InputType.SINGLE_SELECT, InputType.MULTI_SELECT)


class QuestionnaireError(Exception):
    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__("; ".join(i.message for i in result.errors))


class NewConcept(BaseModel):
    """A meaning the user invents in the vocabulary editor, with their word for it."""

    gloss: str = Field(min_length=1, max_length=60)
    part_of_speech: PartOfSpeech
    word: str = Field(min_length=1, max_length=40)


def select_questions(
    bank: Iterable[Question], rng: random.Random, count: int = DEFAULT_QUESTION_COUNT
) -> list[Question]:
    """Every required/always-asked question, plus `count` random others. Returned in bank order."""
    bank = list(bank)
    fixed = [q for q in bank if q.required or q.always_ask]
    optional = [q for q in bank if q not in fixed]
    chosen = {q.id for q in fixed} | {q.id for q in rng.sample(optional, min(count, len(optional)))}
    return [q for q in bank if q.id in chosen]


def concept_id_for(gloss: str) -> str:
    return "user_" + (re.sub(r"[^a-z0-9]+", "_", gloss.lower()).strip("_") or "concept")


def closest_option(question: Question, text: str) -> str | None:
    """The option a free-text answer clearly names, if any (longest match wins)."""
    best, best_len = None, 0
    for option in question.options:
        for candidate in {option.value, option.value.replace("_", " "), option.label}:
            # Prefix match at a word boundary: "mountains" names "mountain".
            if re.search(r"\b" + re.escape(candidate.lower()), text.lower()) and len(candidate) > best_len:
                best, best_len = option.value, len(candidate)
    return best


def build_generation_request(
    questions: Iterable[Question],
    answers: Mapping[str, Any],
    seed: int,
    known_concept_ids: set[str],
    custom_answers: Mapping[str, str] | None = None,
    new_concepts: Iterable[NewConcept] = (),
) -> GenerationRequest:
    """Validate everything against the asked questions and build the request.

    Raises QuestionnaireError listing every problem at once, so the UI can
    show them together instead of one per submit.
    """
    questions_by_id = {q.id: q for q in questions}
    result = ValidationResult()
    data: dict[str, Any] = {"civilization": {}, "language_preferences": {}, "seed": seed, "custom_answers": {}}
    custom_answers = {k: v.strip() for k, v in (custom_answers or {}).items() if v and v.strip()}
    new_concepts = list(new_concepts)

    for question_id in [*answers, *custom_answers]:
        if question_id not in questions_by_id:
            result.add("UNKNOWN_QUESTION", f"Answer given for unknown question {question_id!r}",
                       object_id=question_id)

    # New meanings from the vocabulary editor become concepts (or reuse an existing one).
    concepts: list[dict] = []
    new_words: dict[str, str] = {}
    if new_concepts and not any(q.input_type == InputType.CUSTOM_VOCABULARY for q in questions_by_id.values()):
        result.add("UNKNOWN_QUESTION", "New meanings were sent, but the vocabulary question wasn't asked")
    for nc in new_concepts:
        gloss = nc.gloss.strip()
        existing = re.sub(r"[^a-z0-9]+", "_", gloss.lower()).strip("_")
        concept_id = existing if existing in known_concept_ids else concept_id_for(gloss)
        if concept_id in new_words:
            result.add("DUPLICATE_CONCEPT", f"The meaning {gloss!r} was added twice", object_id=concept_id)
            continue
        new_words[concept_id] = nc.word
        if concept_id not in known_concept_ids:
            concepts.append(SemanticConcept(id=concept_id, gloss=gloss, part_of_speech=nc.part_of_speech,
                                            priority=ConceptPriority.USER).model_dump(mode="json"))
    known_concept_ids = known_concept_ids | set(new_words)

    civ_notes: list[str] = []
    language_notes: list[str] = []
    for question in questions_by_id.values():
        answer = _normalize(answers.get(question.id))
        custom = custom_answers.get(question.id)

        if custom is not None:
            if question.input_type not in SELECTS or not question.allow_custom:
                result.add("INVALID_ANSWER", f"{question.id!r} doesn't take a written answer", object_id=question.id)
                continue
            if len(custom) > MAX_CUSTOM_ANSWER:
                result.add("INVALID_ANSWER", f"Please keep this under {MAX_CUSTOM_ANSWER} characters",
                           object_id=question.id)
                continue
            data["custom_answers"][question.maps_to] = custom
            (civ_notes if question.maps_to.startswith("civilization.") else language_notes).append(
                f"{question.prompt} {custom}")
            if question.maps_to == "civilization.era" and (year := parse_year(custom)) is not None:
                # "around 1500 BCE" -> founded 1500 BCE, Bronze Age
                data["civilization"]["founding_year"] = year
                answer = answer or era_for_year(year)
            if answer is None and question.input_type == InputType.SINGLE_SELECT:
                answer = closest_option(question, custom)  # may stay None: then the AI decides

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
            _conflict(result, concept_id, existing, new)

    words = data["language_preferences"].setdefault("custom_words", {})
    for concept_id, word in new_words.items():
        if concept_id in words and words[concept_id] != word.strip():
            _conflict(result, concept_id, words[concept_id], word.strip())
        else:
            words[concept_id] = word.strip()
    if concepts:
        data["language_preferences"]["custom_concepts"] = concepts

    if civ_notes:
        civ = data["civilization"]
        civ["description"] = "\n".join(filter(None, [civ.get("description"), *civ_notes]))
    if language_notes:
        prefs = data["language_preferences"]
        prefs["notes"] = "\n".join(filter(None, [prefs.get("notes"), *language_notes]))

    if not result.ok:
        raise QuestionnaireError(result)
    return GenerationRequest.model_validate(data)


def _conflict(result: ValidationResult, concept_id: str, existing: str, new: str) -> None:
    result.add("CONFLICTING_CUSTOM_WORD",
               f"Two different words were given for {concept_id!r}: {existing!r} and {new!r}",
               object_id=concept_id, details={"spellings": [existing, new]})


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
        case InputType.FREE_TEXT | InputType.SHORT_TEXT | InputType.CUSTOM_WORD:
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
