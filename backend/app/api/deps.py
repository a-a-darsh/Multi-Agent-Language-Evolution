"""Shared singletons for the API. Tests replace these via `app.dependency_overrides`."""

from functools import cache

from app.agents.pipeline import AgentLLMs, agent_llms_from_settings
from app.api.errors import ApiError
from app.llm.settings import SettingsStore
from app.persistence.database import make_session_factory
from app.persistence.repository import Repository
from app.services.generation_jobs import GenerationService


@cache
def get_repository() -> Repository:
    repository = Repository(make_session_factory())
    repository.mark_interrupted()
    return repository


def llms_from_saved_settings() -> tuple[AgentLLMs, dict]:
    settings = SettingsStore.default().load()
    if settings is None:
        raise ApiError(409, "LLM_NOT_CONFIGURED", "Choose an AI model in the setup screen first")
    return agent_llms_from_settings(settings), {"provider": settings.provider.value, "model": settings.model}


@cache
def get_generation_service() -> GenerationService:
    return GenerationService(get_repository(), llms_from_saved_settings)
