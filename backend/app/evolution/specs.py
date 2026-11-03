"""How callers (the API, the evolution agent) describe a step of sound changes.

A change is a named template with optional parameters, or a full rule. An
exception names the change by its position in the step and the word by
lexeme id or concept id, since ids generated for templates aren't known to
the caller in advance.
"""

from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.evolution.sound_change_engine import SoundChangeResult, apply_sound_change
from app.evolution.templates import TEMPLATES
from app.models.language import LanguageState
from app.models.sound_change import SoundChange, SoundChangeException


class SpecError(ValueError):
    pass


class ChangeSpec(BaseModel):
    """Either a named template (with optional parameters) or a full custom rule."""

    template: str | None = Field(default=None, description="Template name, e.g. 'palatalization'.")
    params: dict[str, Any] = Field(default_factory=dict, description="Template parameters, e.g. {'vowel': 'a'}.")
    rule: SoundChange | None = Field(default=None, description="A custom rule, instead of a template.")

    @model_validator(mode="after")
    def _one(self) -> "ChangeSpec":
        if (self.template is None) == (self.rule is None):
            raise ValueError("give either `template` or `rule`")
        return self

    def build(self) -> SoundChange:
        if self.rule is not None:
            return self.rule
        if self.template not in TEMPLATES:
            raise SpecError(f"no template {self.template!r}; known: {', '.join(TEMPLATES)}. For any other "
                            "change, write a custom rule: {\"rule\": {\"name\": ..., \"target\": ..., ...}}")
        try:
            return TEMPLATES[self.template](**self.params)
        except (TypeError, ValidationError, ValueError) as e:
            raise SpecError(f"template {self.template}: {e}") from e


class ExceptionSpec(BaseModel):
    change_index: int = Field(ge=0, description="Which change of the step (0-based) the word skips.")
    lexeme_id: str | None = None
    concept_id: str | None = Field(default=None, description="The word, by its meaning id (e.g. 'water').")
    reason: str = Field(min_length=3, description="Why this word resisted, e.g. 'very frequent word'.")

    @model_validator(mode="after")
    def _one(self) -> "ExceptionSpec":
        if (self.lexeme_id is None) == (self.concept_id is None):
            raise ValueError("give either `lexeme_id` or `concept_id`")
        return self


def build_step(state: LanguageState, specs: list[ChangeSpec],
               exceptions: list[ExceptionSpec]) -> tuple[list[SoundChange], list[SoundChangeException]]:
    changes = [spec.build() for spec in specs]
    by_concept = {lx.concept_id: lx.id for lx in state.lexemes}
    built = []
    for e in exceptions:
        if e.change_index >= len(changes):
            raise SpecError(f"change_index {e.change_index} is out of range (the step has {len(changes)} changes)")
        lexeme_id = e.lexeme_id or by_concept.get(e.concept_id)
        if lexeme_id is None:
            raise SpecError(f"no word for concept {e.concept_id!r}")
        built.append(SoundChangeException(change_id=changes[e.change_index].id, lexeme_id=lexeme_id, reason=e.reason))
    return changes, built


def run_step(state: LanguageState, specs: list[ChangeSpec], exceptions: list[ExceptionSpec], years: int,
             spellings: dict[str, str] | None = None) -> SoundChangeResult:
    """Raises SpecError or SoundChangeError for requests that can't even be previewed."""
    changes, built = build_step(state, specs, exceptions)
    return apply_sound_change(state, changes, year=state.version.valid_from_year + years,
                              spellings=spellings or {}, exceptions=built)
