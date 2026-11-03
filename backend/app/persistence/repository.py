"""The only code that knows how languages and generations are stored."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.agents.pipeline import GenerationOutcome
from app.models.civilization import World
from app.models.common import DomainModel, new_id
from app.models.generation import GenerationRequest
from app.models.language import LanguageState
from app.models.lexicon import LexemeOrigin
from app.models.validation import ValidationIssue
from app.persistence.database import GenerationRow, LanguageRow
from app.persistence.migrations import upgrade_language_state, upgrade_world

GenerationStatus = Literal["queued", "running", "succeeded", "failed"]


class GenerationSummary(DomainModel):
    id: str
    status: GenerationStatus
    current_step: str | None = None
    steps: list[str] = Field(default_factory=list)
    language_id: str | None = None
    errors: list[ValidationIssue] = Field(default_factory=list)


class LanguageSummary(DomainModel):
    id: str
    name: str
    civilization_name: str
    entry_count: int
    derived_count: int
    created_at: str


class _Lenient(BaseModel):
    """Reads stored history without re-applying today's validation rules.

    The LLM-facing schemas get stricter over time (e.g. affix meanings gained a
    minimum length). Re-validating old records against them made saved languages
    fail to load with a 500, so history is read through these narrow, lenient
    models that keep only what the views display and ignore everything else.
    """

    model_config = ConfigDict(extra="ignore")


class RecordedStep(_Lenient):
    step: str
    issues: list[ValidationIssue] = Field(default_factory=list)


class RecordedCall(_Lenient):
    step: str
    model: str
    seconds: float
    error: str | None = None


class RecordedAgent(_Lenient):
    attempts: list[RecordedStep] = Field(default_factory=list)   # design agent
    rounds: list[RecordedStep] = Field(default_factory=list)     # dictionary agent
    llm_calls: list[RecordedCall] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)


class RecordedRequest(_Lenient):
    seed: int


class GenerationRecord(_Lenient):
    request: RecordedRequest
    design: RecordedAgent
    dictionary: RecordedAgent | None = None


class StoredLanguage(DomainModel):
    state: LanguageState
    world: World
    generation_id: str


class StaleLanguageError(Exception):
    """The language changed since it was read; the caller must reload and retry."""


class Repository:
    def __init__(self, sessions: sessionmaker):
        self.sessions = sessions

    # --- generations -----------------------------------------------------------

    def create_generation(self, request: GenerationRequest, model: dict) -> str:
        gid = new_id()
        with self.sessions.begin() as s:
            s.add(GenerationRow(id=gid, status="queued", request_json=request.model_dump_json(),
                                model_json=json.dumps(model)))
        return gid

    def record_step(self, gid: str, step: str) -> None:
        with self.sessions.begin() as s:
            row = s.get(GenerationRow, gid)
            row.status, row.current_step = "running", step
            row.steps_json = json.dumps([*json.loads(row.steps_json), step])

    def fail_generation(self, gid: str, errors: list[ValidationIssue], outcome: GenerationOutcome | None = None) -> None:
        with self.sessions.begin() as s:
            row = s.get(GenerationRow, gid)
            row.status = "failed"
            row.error_json = json.dumps([e.model_dump(mode="json") for e in errors])
            row.outcome_json = outcome.model_dump_json() if outcome else None

    def complete_generation(self, gid: str, outcome: GenerationOutcome, world: World) -> str:
        """Save the language and mark the job done in ONE transaction: a language is never
        visible unless its generation completed (the plan's transactional requirement)."""
        state = outcome.dictionary.state
        with self.sessions.begin() as s:
            s.add(LanguageRow(
                id=state.language.id, name=state.language.name, civilization_name=state.civilization.name,
                entry_count=len(state.lexemes),
                derived_count=sum(bool(lx.base_lexeme_ids) for lx in state.lexemes),  # affix or compound
                generation_id=gid, state_json=state.model_dump_json(), world_json=world.model_dump_json()))
            row = s.get(GenerationRow, gid)
            row.status, row.language_id, row.outcome_json = "succeeded", state.language.id, outcome.model_dump_json()
        return state.language.id

    def get_generation(self, gid: str) -> GenerationSummary | None:
        with self.sessions() as s:
            row = s.get(GenerationRow, gid)
            if row is None:
                return None
            errors = [ValidationIssue.model_validate(e) for e in json.loads(row.error_json or "[]")]
            return GenerationSummary(id=row.id, status=row.status, current_step=row.current_step,
                                     steps=json.loads(row.steps_json), language_id=row.language_id, errors=errors)

    def get_generation_record(self, gid: str) -> GenerationRecord | None:
        with self.sessions() as s:
            row = s.get(GenerationRow, gid)
            return GenerationRecord.model_validate_json(row.outcome_json) if row and row.outcome_json else None

    def mark_interrupted(self) -> None:
        """Jobs run in-process; any left 'running' after a restart were interrupted."""
        with self.sessions.begin() as s:
            for row in s.scalars(select(GenerationRow).where(GenerationRow.status.in_(["queued", "running"]))):
                row.status = "failed"
                row.error_json = json.dumps([ValidationIssue(
                    code="INTERRUPTED", message="The server restarted before this generation finished").model_dump()])

    # --- languages ---------------------------------------------------------------

    def list_languages(self) -> list[LanguageSummary]:
        with self.sessions() as s:
            rows = s.scalars(select(LanguageRow).order_by(LanguageRow.created_at.desc()))
            return [LanguageSummary(id=r.id, name=r.name, civilization_name=r.civilization_name,
                                    entry_count=r.entry_count, derived_count=r.derived_count,
                                    created_at=r.created_at.isoformat()) for r in rows]

    def save_language_state(self, state: LanguageState, expected_version_id: str) -> None:
        """Replace a language's state (e.g. after a sound change or a new word).

        `expected_version_id` is the version the caller started from. If the stored language has
        moved on since (someone else evolved it), nothing is written and StaleLanguageError is raised,
        so two writers can never silently overwrite each other's history.
        """
        with self.sessions.begin() as s:
            row = s.get(LanguageRow, state.language.id)
            if row is None:
                raise KeyError(state.language.id)
            stored_version = json.loads(row.state_json)["version"]["id"]
            if stored_version != expected_version_id:
                raise StaleLanguageError(f"language {state.language.id} is at version {stored_version}, "
                                         f"not {expected_version_id}")
            row.state_json = state.model_dump_json()
            row.entry_count = len(state.lexemes)
            row.derived_count = sum(bool(lx.base_lexeme_ids) for lx in state.lexemes)

    def get_language(self, language_id: str) -> StoredLanguage | None:
        with self.sessions() as s:
            row = s.get(LanguageRow, language_id)
            if row is None:
                return None
            return StoredLanguage(
                state=LanguageState.model_validate(upgrade_language_state(json.loads(row.state_json))),
                world=World.model_validate(upgrade_world(json.loads(row.world_json))),
                generation_id=row.generation_id)
