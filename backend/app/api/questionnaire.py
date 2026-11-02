import secrets

from fastapi import APIRouter
from pydantic import BaseModel

from app.data import load_questions, load_semantic_inventory
from app.generation.phoneme_generator import derive_rng
from app.models.lexicon import SemanticConcept
from app.models.questionnaire import Question
from app.questionnaire import select_questions

router = APIRouter()


class QuestionnaireResponse(BaseModel):
    seed: int
    questions: list[Question]


def questions_for_seed(seed: int) -> list[Question]:
    """The session's questions are derived from its seed, so the server can
    re-derive exactly which questions were asked when answers come back."""
    return select_questions(load_questions(), derive_rng(seed, "questionnaire"))


@router.get("/questionnaire")
def get_questionnaire(seed: int | None = None) -> QuestionnaireResponse:
    seed = seed if seed is not None else secrets.randbelow(2**31)
    return QuestionnaireResponse(seed=seed, questions=questions_for_seed(seed))


@router.get("/concepts")
def get_concepts() -> list[SemanticConcept]:
    return list(load_semantic_inventory())
