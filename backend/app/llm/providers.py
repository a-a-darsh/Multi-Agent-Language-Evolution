"""Connection checks: verify credentials and list the models a provider offers.

Model lists come from each provider's API rather than a hardcoded list, so
they stay current.
"""

import httpx

from app.llm.settings import LLMProvider

ANTHROPIC_VERSION = "2023-06-01"

# OpenAI lists every model type; hide ones that can't do chat.
_OPENAI_NON_CHAT = ("embedding", "tts", "whisper", "dall-e", "moderation", "transcribe", "image",
                    "davinci", "babbage", "audio", "realtime", "search")


class ProviderError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _get(client: httpx.Client, url: str, provider: str, headers: dict[str, str] | None = None) -> dict:
    try:
        response = client.get(url, headers=headers)
    except httpx.HTTPError as e:
        raise ProviderError("PROVIDER_UNREACHABLE", f"Could not reach {provider} at {url}: {e}") from e
    if response.status_code in (401, 403):
        raise ProviderError("INVALID_API_KEY", f"{provider} rejected the API key")
    if response.is_error:
        raise ProviderError("PROVIDER_ERROR", f"{provider} returned HTTP {response.status_code}")
    return response.json()


def list_models(provider: LLMProvider, api_key: str | None, base_url: str | None,
                client: httpx.Client) -> list[str]:
    match provider:
        case LLMProvider.LOCAL:
            url = (base_url or "").rstrip("/")
            body = _get(client, f"{url}/api/tags", "Ollama")
            # Embedding models (used for RAG later) can't chat.
            models = [m["name"] for m in body.get("models", []) if "embed" not in m["name"]]
            if not models:
                raise ProviderError("NO_MODELS", "Ollama is running but has no models. Run `ollama pull <model>` first.")
        case LLMProvider.OPENAI:
            body = _get(client, "https://api.openai.com/v1/models", "OpenAI",
                        {"Authorization": f"Bearer {api_key}"})
            models = [m["id"] for m in body["data"] if not any(x in m["id"] for x in _OPENAI_NON_CHAT)]
        case LLMProvider.ANTHROPIC:
            body = _get(client, "https://api.anthropic.com/v1/models?limit=1000", "Anthropic",
                        {"x-api-key": api_key or "", "anthropic-version": ANTHROPIC_VERSION})
            models = [m["id"] for m in body["data"]]
        case LLMProvider.OPENROUTER:
            # The model list is public, so check the key separately first.
            _get(client, "https://openrouter.ai/api/v1/key", "OpenRouter",
                 {"Authorization": f"Bearer {api_key}"})
            body = _get(client, "https://openrouter.ai/api/v1/models", "OpenRouter")
            models = [m["id"] for m in body["data"]]
    return sorted(models)
