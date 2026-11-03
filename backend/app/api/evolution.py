"""Evolving a saved language: sound changes (preview / commit), new words, and version history.

Every write starts from the version the client saw (`expected_version_id`, optional) and is
refused with 409 if the language moved on in the meantime.
"""

import inspect
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, model_validator

from app.agents.evolution_agent import ToolUse, run_evolution_agent
from app.api.deps import get_repository, get_tool_llm
from app.api.errors import ApiError
from app.evolution.lexicon import CoinageError, coin_root, compound_word, derive_word
from app.evolution.sound_change_engine import SoundChangeError, SoundChangePreview
from app.evolution.specs import ChangeSpec, ExceptionSpec, SpecError, run_step
from app.evolution.templates import TEMPLATES
from app.llm.tool_calling import ToolCallingLLM
from app.models.language import LanguageState
from app.models.lexicon import PartOfSpeech
from app.models.sound_change import SoundChange
from app.models.validation import ValidationIssue
from app.persistence.repository import Repository, StaleLanguageError
from app.services.dictionary_view import DictionaryView, EntryDetail

router = APIRouter()


# --- templates -------------------------------------------------------------------

class TemplateInfo(BaseModel):
    name: str
    summary: str
    parameters: dict[str, Any]
    rule: SoundChange


@router.get("/sound-change-templates")
def list_templates() -> list[TemplateInfo]:
    out = []
    for name, make in TEMPLATES.items():
        params = {p.name: p.default for p in inspect.signature(make).parameters.values()
                  if p.default is not inspect.Parameter.empty}
        out.append(TemplateInfo(name=name, summary=(inspect.getdoc(make) or "").split("\n")[0],
                                parameters=params, rule=make()))
    return out


# --- sound changes -------------------------------------------------------------------

class StepRequest(BaseModel):
    changes: list[ChangeSpec] = Field(min_length=1)
    years: int = Field(default=100, gt=0, description="How long after the current version this step happens.")
    spellings: dict[str, str] = Field(default_factory=dict, description="Letters for new sounds, by IPA.")
    exceptions: list[ExceptionSpec] = Field(default_factory=list)
    expected_version_id: str | None = None


class StepResponse(BaseModel):
    ok: bool
    preview: SoundChangePreview
    version_id: str | None = None
    year: int | None = None


def _load_state(language_id: str, repo: Repository, expected_version_id: str | None) -> LanguageState:
    stored = repo.get_language(language_id)
    if stored is None:
        raise ApiError(404, "NOT_FOUND", "No such language")
    if expected_version_id and stored.state.version.id != expected_version_id:
        raise ApiError(409, "STALE_VERSION", "The language has changed since you loaded it; reload and try again")
    return stored.state


def _run_step(state: LanguageState, body: StepRequest):
    try:
        return run_step(state, body.changes, body.exceptions, body.years, body.spellings)
    except SpecError as e:
        raise ApiError(422, "BAD_STEP", str(e)) from e
    except SoundChangeError as e:
        raise ApiError(422, "SOUND_CHANGE_INVALID", str(e)) from e


def _save(repo: Repository, new_state: LanguageState, old_state: LanguageState) -> None:
    try:
        repo.save_language_state(new_state, expected_version_id=old_state.version.id)
    except StaleLanguageError as e:
        raise ApiError(409, "STALE_VERSION", "The language has changed since you loaded it; reload and try again") \
            from e


@router.post("/languages/{language_id}/sound-changes/preview")
def preview_sound_changes(language_id: str, body: StepRequest,
                          repo: Repository = Depends(get_repository)) -> StepResponse:
    result = _run_step(_load_state(language_id, repo, body.expected_version_id), body)
    return StepResponse(ok=result.preview.ok, preview=result.preview)


@router.post("/languages/{language_id}/sound-changes")
def commit_sound_changes(language_id: str, body: StepRequest,
                         repo: Repository = Depends(get_repository)) -> StepResponse:
    state = _load_state(language_id, repo, body.expected_version_id)
    result = _run_step(state, body)
    if result.state is None:
        raise ApiError(422, "SOUND_CHANGE_REFUSED",
                       f"{len(result.preview.illegal)} word(s) would break the sound rules; nothing was saved",
                       [ValidationIssue(code="ILLEGAL_RESULT", message=f"{w.concept_id}: {w.form!r} - "
                                        + "; ".join(w.problems), object_id=w.concept_id)
                        for w in result.preview.illegal])
    _save(repo, result.state, state)
    return StepResponse(ok=True, preview=result.preview, version_id=result.state.version.id,
                        year=result.state.version.valid_from_year)


# --- new words -----------------------------------------------------------------------

class NewWordRequest(BaseModel):
    concept_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")
    gloss: str = Field(min_length=1)
    part_of_speech: PartOfSpeech | None = None
    # A new root: give its spelling, or leave everything below empty to generate one.
    spelling: str | None = None
    # A derivation: an existing word plus a word-building rule.
    base_lexeme_id: str | None = None
    rule_id: str | None = None
    # A compound: two existing words, in order.
    parts: list[str] | None = Field(default=None, min_length=2, max_length=2)
    seed: int = 0
    expected_version_id: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> "NewWordRequest":
        if (self.base_lexeme_id is None) != (self.rule_id is None):
            raise ValueError("a derivation needs both base_lexeme_id and rule_id")
        if (self.base_lexeme_id or self.parts) and self.spelling:
            raise ValueError("a built word comes from its parts; don't give a spelling")
        if self.base_lexeme_id and self.parts:
            raise ValueError("give either a derivation (base_lexeme_id + rule_id) or a compound (parts)")
        if not self.base_lexeme_id and self.part_of_speech is None:
            raise ValueError("a new root or compound needs a part_of_speech")
        return self


@router.post("/languages/{language_id}/words", status_code=201)
def add_word(language_id: str, body: NewWordRequest, repo: Repository = Depends(get_repository)) -> EntryDetail:
    state = _load_state(language_id, repo, body.expected_version_id)
    try:
        if body.parts:
            new_state = compound_word(state, body.parts, body.concept_id, body.gloss, body.part_of_speech)
        elif body.base_lexeme_id:
            new_state = derive_word(state, body.base_lexeme_id, body.rule_id, body.concept_id, body.gloss,
                                    body.part_of_speech)
        else:
            new_state = coin_root(state, body.concept_id, body.gloss, body.part_of_speech, body.spelling, body.seed)
    except CoinageError as e:
        raise ApiError(422, "WORD_INVALID", str(e), e.result.errors) from e
    _save(repo, new_state, state)
    return DictionaryView(new_state).detail(new_state.lexemes[-1].id)


# --- history ---------------------------------------------------------------------------

class ChangeInfo(BaseModel):
    name: str
    description: str | None


class ExceptionInfo(BaseModel):
    concept_id: str
    change: str
    reason: str


class VersionInfo(BaseModel):
    id: str
    year: int
    changes: list[ChangeInfo]
    exceptions: list[ExceptionInfo]
    words_coined: list[str]
    sounds: list[str]
    syllable_patterns: list[str]


@router.get("/languages/{language_id}/history")
def get_history(language_id: str, repo: Repository = Depends(get_repository)) -> list[VersionInfo]:
    state = _load_state(language_id, repo, None)
    concept = {lx.id: lx.concept_id for lx in state.lexemes}
    out = []
    for index, v in enumerate([*state.previous_versions, state.version]):
        names = {c.id: c.name for c in v.sound_changes}
        coined = [lx.concept_id for lx in state.lexemes
                  if lx.coined_in_version_id == v.id and not (index == 0 and lx.coined_in_version_id is None)]
        out.append(VersionInfo(
            id=v.id, year=v.valid_from_year,
            changes=[ChangeInfo(name=c.name, description=c.description) for c in v.sound_changes],
            exceptions=[ExceptionInfo(concept_id=concept.get(e.lexeme_id, e.lexeme_id), change=names[e.change_id],
                                      reason=e.reason) for e in v.exceptions],
            words_coined=coined, sounds=[p.grapheme for p in v.phonology.phonemes],
            syllable_patterns=v.phonology.syllable_patterns))
    return out


# --- the evolution agent ---------------------------------------------------------------------

class AgentRequest(BaseModel):
    years: int = Field(default=200, gt=0, le=2000)
    context: str | None = Field(default=None, max_length=1500, description="Historical pressures on the language.")
    save: bool = False
    expected_version_id: str | None = None


class AgentResponse(BaseModel):
    status: str
    rationale: str | None
    changes: list[ChangeInfo]
    exceptions: list[ExceptionInfo]
    issues: list[str]
    model_calls: int
    seconds: float
    tool_log: list[ToolUse]
    saved: bool
    version_id: str | None
    year: int | None


@router.post("/languages/{language_id}/evolve-agent")
def evolve_with_agent(language_id: str, body: AgentRequest, repo: Repository = Depends(get_repository),
                      llm: ToolCallingLLM = Depends(get_tool_llm)) -> AgentResponse:
    """Let the evolution agent propose (and optionally save) one step. Runs synchronously: with a
    local model this takes about a minute."""
    state = _load_state(language_id, repo, body.expected_version_id)
    outcome = run_evolution_agent(state, llm, years=body.years, context=body.context)
    new = outcome.state
    if body.save and new is not None:
        _save(repo, new, state)
    concept = {lx.id: lx.concept_id for lx in state.lexemes}
    names = {c.id: c.name for c in outcome.changes}
    return AgentResponse(
        status=outcome.status, rationale=outcome.rationale,
        changes=[ChangeInfo(name=c.name, description=c.description) for c in outcome.changes],
        exceptions=[ExceptionInfo(concept_id=concept.get(e.lexeme_id, e.lexeme_id), change=names[e.change_id],
                                  reason=e.reason) for e in outcome.exceptions],
        issues=outcome.issues, model_calls=len(outcome.llm_calls),
        seconds=round(sum(c.seconds for c in outcome.llm_calls), 1), tool_log=outcome.tool_log,
        saved=body.save and new is not None, version_id=new.version.id if new else None,
        year=new.version.valid_from_year if new else None)
