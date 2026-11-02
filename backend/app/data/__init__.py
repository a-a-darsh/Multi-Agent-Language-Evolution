import json
from functools import cache
from pathlib import Path

from app.models.generation import GenerationRequest
from app.models.lexicon import SemanticConcept
from app.models.questionnaire import Question

DATA_DIR = Path(__file__).parent


def _read_json(name: str):
    # utf-8-sig tolerates a byte-order mark: Windows PowerShell 5's `Set-Content -Encoding utf8`
    # adds one, and plain utf-8 decoding then made every questionnaire request fail.
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8-sig"))


@cache
def load_questions() -> tuple[Question, ...]:
    return tuple(Question.model_validate(q) for q in _read_json("questions.json"))


@cache
def load_semantic_inventory() -> tuple[SemanticConcept, ...]:
    return tuple(SemanticConcept.model_validate(c) for c in _read_json("semantic_inventory.json"))


def load_example_requests() -> dict[str, GenerationRequest]:
    raw = _read_json("examples/requests.json")
    return {name: GenerationRequest.model_validate(r) for name, r in raw.items()}
