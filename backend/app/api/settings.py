from collections.abc import Iterator

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.errors import ApiError
from app.llm.providers import ProviderError, list_models
from app.llm.settings import DEFAULT_LOCAL_URL, LLMProvider, LLMSettings, LLMSettingsView, SettingsStore

router = APIRouter(prefix="/settings/llm")


def get_store() -> SettingsStore:
    return SettingsStore.default()


def get_http_client() -> Iterator[httpx.Client]:
    with httpx.Client(timeout=15) as client:
        yield client


class ConnectionBody(BaseModel):
    provider: LLMProvider
    # Omit to reuse the saved key for the same provider (e.g. just switching models).
    api_key: str | None = None
    base_url: str | None = None


class SaveBody(ConnectionBody):
    model: str


class ModelsResponse(BaseModel):
    models: list[str]


def _resolve_key(body: ConnectionBody, store: SettingsStore) -> str | None:
    if not body.provider.needs_api_key:
        return None
    if body.api_key and body.api_key.strip():
        return body.api_key.strip()
    saved = store.load()
    if saved and saved.provider == body.provider and saved.api_key:
        return saved.api_key.get_secret_value()
    raise ApiError(422, "MISSING_API_KEY", f"Enter an API key for {body.provider}")


def _models(body: ConnectionBody, api_key: str | None, client: httpx.Client) -> list[str]:
    base_url = body.base_url or (DEFAULT_LOCAL_URL if body.provider == LLMProvider.LOCAL else None)
    try:
        return list_models(body.provider, api_key, base_url, client)
    except ProviderError as e:
        raise ApiError(422, e.code, e.message) from e


@router.get("")
def get_settings(store: SettingsStore = Depends(get_store)) -> LLMSettingsView:
    return LLMSettingsView.of(store.load())


@router.post("/models")
def check_connection(body: ConnectionBody, store: SettingsStore = Depends(get_store),
                     client: httpx.Client = Depends(get_http_client)) -> ModelsResponse:
    return ModelsResponse(models=_models(body, _resolve_key(body, store), client))


@router.put("")
def save_settings(body: SaveBody, store: SettingsStore = Depends(get_store),
                  client: httpx.Client = Depends(get_http_client)) -> LLMSettingsView:
    api_key = _resolve_key(body, store)
    if body.model not in _models(body, api_key, client):
        raise ApiError(422, "UNKNOWN_MODEL", f"{body.provider} does not offer model {body.model!r}")
    settings = LLMSettings(provider=body.provider, model=body.model, base_url=body.base_url, api_key=api_key)
    store.save(settings)
    return LLMSettingsView.of(settings)
