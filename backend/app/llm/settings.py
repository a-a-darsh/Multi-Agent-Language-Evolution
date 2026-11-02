"""Which LLM the app uses, and where that choice is stored.

API keys are stored in a local, git-ignored file and are never returned
to the frontend; the UI only sees a short hint like "sk-…a1b2".
"""

import json
import os
from enum import StrEnum
from pathlib import Path

from pydantic import SecretStr, model_validator

from app.models.common import DomainModel


class LLMProvider(StrEnum):
    LOCAL = "local"  # Ollama
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"

    @property
    def needs_api_key(self) -> bool:
        return self != LLMProvider.LOCAL


DEFAULT_LOCAL_URL = "http://localhost:11434"
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / ".local" / "llm_settings.json"


class LLMSettings(DomainModel):
    provider: LLMProvider
    model: str
    base_url: str | None = None
    api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _provider_requirements(self) -> "LLMSettings":
        if self.provider.needs_api_key and not self.api_key:
            raise ValueError(f"{self.provider} requires an API key")
        if self.provider == LLMProvider.LOCAL:
            self.base_url = self.base_url or DEFAULT_LOCAL_URL
            self.api_key = None
        return self


class LLMSettingsView(DomainModel):
    """What the frontend is allowed to see."""

    configured: bool
    provider: LLMProvider | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_hint: str | None = None

    @classmethod
    def of(cls, settings: LLMSettings | None) -> "LLMSettingsView":
        if settings is None:
            return cls(configured=False)
        return cls(configured=True, provider=settings.provider, model=settings.model,
                   base_url=settings.base_url,
                   api_key_hint=key_hint(settings.api_key.get_secret_value()) if settings.api_key else None)


def key_hint(key: str) -> str:
    return f"{key[:3]}…{key[-4:]}" if len(key) >= 12 else "••••"


class SettingsStore:
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def default(cls) -> "SettingsStore":
        return cls(Path(os.environ.get("LANGEVO_SETTINGS_PATH", DEFAULT_SETTINGS_PATH)))

    def load(self) -> LLMSettings | None:
        if not self.path.exists():
            return None
        return LLMSettings.model_validate_json(self.path.read_text(encoding="utf-8-sig"))

    def save(self, settings: LLMSettings) -> None:
        data = settings.model_dump(mode="json")
        # SecretStr serializes as "**********"; store the real value.
        data["api_key"] = settings.api_key.get_secret_value() if settings.api_key else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
