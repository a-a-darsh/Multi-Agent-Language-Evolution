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
from app.questionnaire import QuestionnaireError, build_generation_request

router = APIRouter()


class WorldGenerationBody(BaseModel):
    seed: int
    answers: dict[str, Any] = Field(default_factory=dict)


class WorldGenerationResponse(BaseModel):
    request: GenerationRequest
    world: World


@router.post("/worlds")
def create_world(body: WorldGenerationBody) -> WorldGenerationResponse:
    concept_ids = {c.id for c in load_semantic_inventory()}
    try:
        request = build_generation_request(questions_for_seed(body.seed), body.answers, body.seed, concept_ids)
    except QuestionnaireError as e:
        raise ApiError(422, "INVALID_ANSWERS", "Some answers need fixing", e.result.errors) from e

    # User words aren't used by world generation, but the language generator
    # will reject them later, so surface problems now while the user is here.
    _, word_result = analyze_user_words(request.language_preferences.custom_words)
    if not word_result.ok:
        raise ApiError(422, "INVALID_CUSTOM_WORDS", "Some of your words can't be used yet", word_result.errors)

    return WorldGenerationResponse(request=request, world=generate_world(request))
