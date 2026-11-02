import secrets
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from app.models.common import DomainModel, new_id
from app.models.grammar import MorphologicalType, WordOrder
from app.models.lexicon import SemanticConcept
from app.models.validation import ValidationResult


class CivilizationProfile(DomainModel):
    """Civilization inputs. `None` / empty means "let the AI decide"."""

    name: str | None = None
    era: str | None = None
    founding_year: int | None = None
    geography: str | None = None
    economy: str | None = None
    social_organization: str | None = None
    cultural_traits: list[str] = Field(default_factory=list)
    description: str | None = None


class LanguagePreferences(DomainModel):
    word_order: WordOrder | None = None
    morphological_type: MorphologicalType | None = None
    grammatical_gender: bool | None = None
    case_marking: bool | None = None
    sound_preferences: str | None = None
    special_features: list[str] = Field(default_factory=list)
    # concept_id -> spelling, e.g. {"water": "mari"}. Authoritative: never altered.
    custom_words: dict[str, str] = Field(default_factory=dict)
    # Meanings the user invented (not in the semantic inventory), used as keys in custom_words.
    custom_concepts: list[SemanticConcept] = Field(default_factory=list)
    # Free-form notes on how the user wants the language, passed to every agent.
    notes: str | None = None
    # Target share of the dictionary built from other words (0 = all roots, max 0.5).
    derived_ratio: float | None = Field(default=None, ge=0, le=0.5)


class GenerationRequest(DomainModel):
    civilization: CivilizationProfile = Field(default_factory=CivilizationProfile)
    language_preferences: LanguagePreferences = Field(default_factory=LanguagePreferences)
    # Controls all local randomness. Generated if not supplied so it can be recorded.
    seed: int = Field(default_factory=lambda: secrets.randbelow(2**31))
    # "Describe your own" answers to multiple-choice questions, keyed by the field they
    # describe (e.g. "civilization.geography"). Kept verbatim for provenance; their text is
    # also folded into the civilization description / language notes the agents read.
    custom_answers: dict[str, str] = Field(default_factory=dict)

    def user_specified_fields(self) -> list[str]:
        """Dotted paths of every input the user explicitly provided.

        Anything not listed here is free for the AI to decide, and anything
        listed is a hard constraint.
        """
        specified = []
        for section_name in ("civilization", "language_preferences"):
            section = getattr(self, section_name)
            for field_name in type(section).model_fields:
                value = getattr(section, field_name)
                if value not in (None, [], {}, ""):
                    specified.append(f"{section_name}.{field_name}")
        specified += [f"custom:{path}" for path in self.custom_answers]
        return specified


class GenerationStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class GenerationRun(DomainModel):
    """Audit record of one generation: inputs, model config, and outcome."""

    id: str = Field(default_factory=new_id)
    request: GenerationRequest
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    status: GenerationStatus = GenerationStatus.PENDING
    validation: ValidationResult = Field(default_factory=ValidationResult)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
