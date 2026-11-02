from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.errors import ApiError
from app.api.questionnaire import questions_for_seed
from app.data import load_semantic_inventory
from app.generation.user_vocabulary import analyze_user_words
from app.generation.world_generator import generate_world
from app.models.civilization import World
from app.models.generation import GenerationRequest
from app.questionnaire import NewConcept, QuestionnaireError, build_generation_request

router = APIRouter()


class AnswersBody(BaseModel):
    seed: int
    answers: dict[str, Any] = Field(default_factory=dict)
    # question id -> "describe your own" text for a multiple-choice question
    custom_answers: dict[str, str] = Field(default_factory=dict)
    # meanings invented in the vocabulary editor, each with the user's word
    new_concepts: list[NewConcept] = Field(default_factory=list)
    # share of the dictionary built from other words (the create page's slider)
    derived_ratio: float | None = Field(default=None, ge=0, le=0.5)


class WorldGenerationResponse(BaseModel):
    request: GenerationRequest
    world: World


def validated_request(body: AnswersBody) -> GenerationRequest:
    """Questionnaire answers -> GenerationRequest, rejecting anything the generators can't honor."""
    concept_ids = {c.id for c in load_semantic_inventory()}
    try:
        request = build_generation_request(questions_for_seed(body.seed), body.answers, body.seed, concept_ids,
                                           body.custom_answers, body.new_concepts)
    except QuestionnaireError as e:
        raise ApiError(422, "INVALID_ANSWERS", "Some answers need fixing", e.result.errors) from e
    _, word_result = analyze_user_words(request.language_preferences.custom_words)
    if not word_result.ok:
        raise ApiError(422, "INVALID_CUSTOM_WORDS", "Some of your words can't be used yet", word_result.errors)
    if body.derived_ratio is not None:
        request.language_preferences.derived_ratio = body.derived_ratio
    return request


@router.post("/worlds")
def create_world(body: AnswersBody) -> WorldGenerationResponse:
    request = validated_request(body)
    return WorldGenerationResponse(request=request, world=generate_world(request))
