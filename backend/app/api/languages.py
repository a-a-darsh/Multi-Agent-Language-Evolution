from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_generation_service, get_repository
from app.api.errors import ApiError
from app.api.worlds import AnswersBody, validated_request
from app.models.civilization import Civilization, World
from app.models.lexicon import LexemeOrigin, PartOfSpeech
from app.models.morphology import MorphemeType
from app.models.phonology import Position
from app.models.validation import ValidationIssue
from app.persistence.repository import GenerationSummary, LanguageSummary, Repository, StoredLanguage
from app.services.dictionary_view import DictionaryView, EntryDetail, EntryPage, EntrySummary
from app.services.generation_jobs import GenerationService

router = APIRouter()


# --- generation jobs -------------------------------------------------------------

class GenerationStarted(BaseModel):
    generation_id: str


@router.post("/generations", status_code=202)
def start_generation(body: AnswersBody,
                     service: GenerationService = Depends(get_generation_service)) -> GenerationStarted:
    return GenerationStarted(generation_id=service.start(validated_request(body)))


@router.get("/generations/{generation_id}")
def get_generation(generation_id: str, repo: Repository = Depends(get_repository)) -> GenerationSummary:
    summary = repo.get_generation(generation_id)
    if summary is None:
        raise ApiError(404, "NOT_FOUND", "No such generation")
    return summary


# --- languages -------------------------------------------------------------------

def _load(language_id: str, repo: Repository) -> StoredLanguage:
    stored = repo.get_language(language_id)
    if stored is None:
        raise ApiError(404, "NOT_FOUND", "No such language")
    return stored


class Sound(BaseModel):
    letter: str
    ipa: str
    common: bool


class Restriction(BaseModel):
    letter: str
    not_allowed: list[Position]


class Affix(BaseModel):
    rule_id: str
    name: str
    meaning: str
    form: str
    position: str
    takes: list[PartOfSpeech]
    makes: PartOfSpeech
    examples: list[EntrySummary]


class Feature(BaseModel):
    name: str
    description: str
    realization: str


class LanguageOverview(BaseModel):
    id: str
    name: str
    summary: str | None
    civilization: Civilization
    consonants: list[Sound]
    vowels: list[Sound]
    syllable_patterns: list[str]
    allow_hiatus: bool
    restrictions: list[Restriction]
    word_order: str
    morphological_type: str
    adposition_type: str | None
    features: list[Feature]
    affixes: list[Affix]
    linking_vowel: str | None
    linking_consonant: str | None
    counts: dict[str, int]
    warnings: list[ValidationIssue]
    generation_id: str


@router.get("/languages")
def list_languages(repo: Repository = Depends(get_repository)) -> list[LanguageSummary]:
    return repo.list_languages()


@router.get("/languages/{language_id}")
def get_language(language_id: str, repo: Repository = Depends(get_repository)) -> LanguageOverview:
    stored = _load(language_id, repo)
    state, view = stored.state, DictionaryView(stored.state)
    ph, g = state.version.phonology, state.version.grammar
    morphemes = {m.id: m for m in state.morphemes}

    affixes = []
    for rule in state.derivational_rules:
        m = morphemes[rule.morpheme_id]
        form = "".join(m.phonemes)
        examples = [view.summary(lx) for lx in state.lexemes if lx.derivation and lx.derivation.rule_id == rule.id][:4]
        affixes.append(Affix(rule_id=rule.id, name=rule.name, meaning=rule.meaning,
                             form=f"{form}-" if m.type == MorphemeType.PREFIX else f"-{form}",
                             position=m.type.value, takes=rule.input_parts_of_speech,
                             makes=rule.output_part_of_speech, examples=examples))

    counts = {o.value: sum(lx.origin == o for lx in state.lexemes) for o in LexemeOrigin}
    record = repo.get_generation_record(stored.generation_id)
    warnings = (record.design.warnings + (record.dictionary.warnings if record.dictionary else [])) if record else []
    repair = state.version.boundary_repair
    sound = lambda p: Sound(letter=p.grapheme, ipa=p.ipa, common=p.frequency > 1)  # noqa: E731
    return LanguageOverview(
        id=state.language.id, name=state.language.name, summary=state.summary, civilization=state.civilization,
        consonants=[sound(p) for p in ph.consonants], vowels=[sound(p) for p in ph.vowels],
        syllable_patterns=ph.syllable_patterns, allow_hiatus=ph.allow_hiatus,
        restrictions=[Restriction(letter=r.phoneme_id, not_allowed=r.forbidden_positions) for r in ph.restrictions],
        word_order=g.word_order.value, morphological_type=g.morphological_type.value,
        adposition_type=g.adposition_type.value if g.adposition_type else None,
        features=[Feature(name=f.name, description=f.description, realization=f.realization) for f in g.features],
        affixes=affixes, linking_vowel=repair.linking_vowel if repair else None,
        linking_consonant=repair.linking_consonant if repair else None,
        counts=counts, warnings=warnings, generation_id=stored.generation_id)


@router.get("/languages/{language_id}/world")
def get_world(language_id: str, repo: Repository = Depends(get_repository)) -> World:
    return _load(language_id, repo).world


@router.get("/languages/{language_id}/dictionary")
def search_dictionary(language_id: str, q: str = "", pos: PartOfSpeech | None = None,
                      origin: LexemeOrigin | None = None, page: int = 1, page_size: int = 50,
                      repo: Repository = Depends(get_repository)) -> EntryPage:
    view = DictionaryView(_load(language_id, repo).state)
    return view.search(q, pos, origin, max(page, 1), min(max(page_size, 1), 200))


@router.get("/languages/{language_id}/dictionary/{lexeme_id}")
def get_entry(language_id: str, lexeme_id: str, repo: Repository = Depends(get_repository)) -> EntryDetail:
    detail = DictionaryView(_load(language_id, repo).state).detail(lexeme_id)
    if detail is None:
        raise ApiError(404, "NOT_FOUND", "No such dictionary entry")
    return detail


# --- provenance --------------------------------------------------------------------

class LogStep(BaseModel):
    agent: str
    step: str
    issues: list[ValidationIssue]


class LogCall(BaseModel):
    agent: str
    step: str
    model: str
    seconds: float
    error: str | None


class GenerationLog(BaseModel):
    seed: int
    steps: list[LogStep]
    llm_calls: list[LogCall]


@router.get("/languages/{language_id}/generation")
def get_generation_log(language_id: str, repo: Repository = Depends(get_repository)) -> GenerationLog:
    stored = _load(language_id, repo)
    record = repo.get_generation_record(stored.generation_id)
    if record is None:
        raise ApiError(404, "NOT_FOUND", "No generation record")
    steps, calls = [], []
    for agent, recorded in (("design", record.design), ("dictionary", record.dictionary)):
        if recorded is None:
            continue
        steps += [LogStep(agent=agent, step=s.step, issues=s.issues) for s in recorded.attempts + recorded.rounds]
        calls += [LogCall(agent=agent, step=c.step, model=c.model, seconds=c.seconds, error=c.error)
                  for c in recorded.llm_calls]
    return GenerationLog(seed=record.request.seed, steps=steps, llm_calls=calls)
