"""Execute a morphological plan: assign roots, apply derivations, build the dictionary.

Always produces a complete dictionary. Any planned derivation that can't be
realized becomes an independent root instead and is reported as a failure, so
the planner can be asked to fix just those entries. Building is deterministic:
the same design, plan, and seed always give the same words.
"""

from collections.abc import Iterable

from pydantic import Field

from app.agents.dictionary_schema import CompoundEntry, PlanEntry, Planned, SenseExtension
from app.generation.morphological_engine import MorphologyEngine, MorphologyError
from app.generation.root_generator import RootRegistry
from app.generation.user_vocabulary import UserWord
from app.models.civilization import Civilization
from app.models.common import DomainModel
from app.models.design import LanguageDesign
from app.models.language import Language, LanguageState, LanguageVersion
from app.models.lexicon import ConceptPriority, LexicalSense, PartOfSpeech, SemanticConcept
from app.models.validation import Severity, ValidationIssue

MAX_DERIVATION_DEPTH = 2
# Words that never make sense as compound parts: function words ("trade + in" for market) and
# numbers or quantities, which the model used as filler endings (black = night + one, hot = fire + many).
GRAMMATICAL_POS = {PartOfSpeech.ADPOSITION, PartOfSpeech.PARTICLE, PartOfSpeech.PRONOUN,
                   PartOfSpeech.INTERJECTION, PartOfSpeech.NUMERAL}
NON_PART_CATEGORIES = {"numbers", "quantity", "deixis", "questions", "function", "pronouns", "relations"}
# A word that is part of very many compounds is being used as a filler, not for its meaning
# ("one" was in 12). Words like house or person are legitimately in several (market, palace...).
MAX_COMPOUNDS_PER_PART = 8


def is_basic_vocabulary(concept: SemanticConcept) -> bool:
    """Core concepts (body parts, nature, numbers, colors, basic actions, kinship...). Real languages
    don't build them as compounds (black is not night + one), though an affix may derive a few
    (bad = not + good, like English unhappy)."""
    return concept.priority == ConceptPriority.CORE


def _not_a_part(concept: SemanticConcept) -> bool:
    return concept.part_of_speech in GRAMMATICAL_POS or bool(NON_PART_CATEGORIES & set(concept.categories))
# Extra candidates for derivations that fall back to roots, and for re-rolls.
CANDIDATE_MARGIN = 80
MAX_REROLL_PASSES = 3


class BuildResult(DomainModel):
    state: LanguageState
    # Planned derivations that could not be realized, keyed by concept id. Each became a root.
    failures: dict[str, ValidationIssue] = Field(default_factory=dict)
    # Informational changes made by code (e.g. dropped entries for user-word concepts).
    notes: list[ValidationIssue] = Field(default_factory=list)


def _issue(code: str, concept_id: str, message: str, **details) -> ValidationIssue:
    return ValidationIssue(code=code, object_id=concept_id, message=message, details=details)


def check_plan(plan: Iterable[Planned], concepts: dict[str, SemanticConcept], design: LanguageDesign,
               user_concepts: set[str]) -> tuple[list[Planned], dict[str, ValidationIssue], list[ValidationIssue]]:
    """Plan-level checks that need no word forms, for derivations and compounds.
    Returns (usable entries, failures, notes)."""
    rules = {r.id: r for r in design.derivational_rules}
    usable: dict[str, Planned] = {}
    failures: dict[str, ValidationIssue] = {}
    notes: list[ValidationIssue] = []

    for e in plan:
        if e.concept_id in user_concepts:
            # The user's word is authoritative; dropping the derivation is the only valid fix.
            notes.append(_issue("USER_WORD_KEPT", e.concept_id, f"{e.concept_id!r} keeps the user's own word "
                                f"instead of being built as {e.describe()}"))
            continue
        if e.concept_id in usable or e.concept_id in failures:
            notes.append(_issue("DUPLICATE_PLAN_ENTRY", e.concept_id, f"{e.concept_id!r} was planned twice; "
                                "the first entry was used"))
            continue
        if isinstance(e, CompoundEntry) and e.concept_id in concepts and is_basic_vocabulary(concepts[e.concept_id]):
            # Only one correct fix (its own root), so code makes it without a repair round.
            notes.append(_issue("BASIC_WORD_KEPT_AS_ROOT", e.concept_id, f"{e.concept_id!r} is basic vocabulary, "
                                f"so it keeps its own root instead of being the compound {e.describe()}"))
            continue
        if isinstance(e, CompoundEntry):
            if e.concept_id not in concepts or any(b not in concepts for b in e.bases):
                failures[e.concept_id] = _issue("UNKNOWN_REFERENCE", e.concept_id,
                                                f"{e.concept_id!r} references an unknown concept")
            elif e.concept_id in e.bases:
                failures[e.concept_id] = _issue("SELF_DERIVATION", e.concept_id,
                                                f"{e.concept_id!r} cannot be a compound of itself")
            elif e.first_concept_id == e.second_concept_id:
                failures[e.concept_id] = _issue("SELF_COMPOUND", e.concept_id,
                                                f"{e.concept_id!r} joins {e.first_concept_id!r} with itself")
            elif grammatical := [b for b in e.bases if _not_a_part(concepts[b])]:
                failures[e.concept_id] = _issue(
                    "GRAMMATICAL_COMPOUND_PART", e.concept_id,
                    f"{e.concept_id} = {e.describe()} uses {', '.join(grammatical)} as a part. Numbers, quantities "
                    "and function words can't be compound parts; use parts that are things, actions or qualities, "
                    "or an independent root")
            else:
                usable[e.concept_id] = e
            continue
        concept, base, rule = concepts.get(e.concept_id), concepts.get(e.base_concept_id), rules.get(e.rule_id)
        if concept is None or base is None or rule is None:
            failures[e.concept_id] = _issue("UNKNOWN_REFERENCE", e.concept_id,
                                            f"{e.concept_id!r} references an unknown concept or rule")
        elif e.base_concept_id == e.concept_id:
            failures[e.concept_id] = _issue("SELF_DERIVATION", e.concept_id, f"{e.concept_id!r} cannot derive from itself")
        elif concept.part_of_speech != rule.output_part_of_speech:
            # What an affix MAKES is part of its meaning: the person affix makes a person (a noun), so
            # buy = trade + person is wrong. (What it attaches TO is only guidance: wave = water + movement.)
            failures[e.concept_id] = _issue(
                "AFFIX_WRONG_RESULT", e.concept_id,
                f"{e.concept_id} = {e.describe()}, but the {rule.id} affix makes a {rule.output_part_of_speech.value} "
                f"({rule.meaning}) and {e.concept_id} is a {concept.part_of_speech.value}. Use a fitting affix, a "
                "compound, or an independent root")
        else:
            usable[e.concept_id] = e

    # Same parts = the same word for two meanings.
    composed: dict[tuple[str, ...], str] = {}
    for concept_id, e in list(usable.items()):
        key = tuple(e.bases) + ((e.rule_id,) if isinstance(e, PlanEntry) else ("+",))
        if key in composed:
            failures[concept_id] = _issue(
                "SAME_COMPOSITION", concept_id,
                f"{concept_id} and {composed[key]} are both {e.describe()}, which would be the same word. "
                f"Build {concept_id} differently or give it an independent root")
            del usable[concept_id]
        else:
            composed[key] = concept_id

    # Overused parts: beyond MAX_COMPOUNDS_PER_PART, a part is filler. Later entries fail (deterministic order).
    uses: dict[str, int] = {}
    for concept_id, e in list(usable.items()):
        if not isinstance(e, CompoundEntry):
            continue
        over = [b for b in e.bases if uses.get(b, 0) >= MAX_COMPOUNDS_PER_PART]
        if over:
            failures[concept_id] = _issue(
                "OVERUSED_PART", concept_id,
                f"{concept_id} = {e.describe()}: {', '.join(over)} is already part of {MAX_COMPOUNDS_PER_PART} "
                "compounds. Use a part that says something specific about this concept, or an independent root")
            del usable[concept_id]
            continue
        for b in e.bases:
            uses[b] = uses.get(b, 0) + 1

    # Cycles and depth, following base links (derivation bases and compound parts) through usable entries.
    for concept_id in list(usable):
        if (problem := _graph_problem(concept_id, usable)) is not None:
            failures[concept_id] = problem
    return [e for c, e in usable.items() if c not in failures], failures, notes


def _graph_problem(concept_id: str, usable: dict[str, Planned]) -> ValidationIssue | None:
    def walk(current: str, path: list[str]) -> ValidationIssue | None:
        for base in usable[current].bases:
            if base in path:
                return _issue("DERIVATION_CYCLE", concept_id, f"{' -> '.join([*path, base])} is a cycle")
            if base in usable and (problem := walk(base, [*path, base])):
                return problem
        return None

    if (cycle := walk(concept_id, [concept_id])) is not None:
        return cycle
    if (depth := _depth(concept_id, usable)) > MAX_DERIVATION_DEPTH:
        return _issue("DERIVATION_TOO_DEEP", concept_id,
                      f"{concept_id} is {depth} building steps from a root (max {MAX_DERIVATION_DEPTH})")
    return None


def _depth(concept_id: str, usable: dict[str, Planned]) -> int:
    """Building steps from the roots; only call on acyclic entries."""
    return 1 + max((_depth(b, usable) for b in usable[concept_id].bases if b in usable), default=0)


def check_extensions(extensions: Iterable[SenseExtension], concepts: dict[str, SemanticConcept],
                     derived: set[str], user_concepts: set[str]
                     ) -> tuple[list[SenseExtension], dict[str, ValidationIssue], list[ValidationIssue]]:
    """Checks for shared words (polysemy). Returns (usable extensions, failures, notes)."""
    extensions = list(extensions)
    extended = {x.concept_id for x in extensions}
    usable: dict[str, SenseExtension] = {}
    failures: dict[str, ValidationIssue] = {}
    notes: list[ValidationIssue] = []
    for x in extensions:
        concept, base = concepts.get(x.concept_id), concepts.get(x.base_concept_id)
        if x.concept_id in user_concepts:
            notes.append(_issue("USER_WORD_KEPT", x.concept_id, f"{x.concept_id!r} keeps the user's own word "
                                f"instead of sharing the word for {x.base_concept_id!r}"))
        elif x.concept_id in derived:
            notes.append(_issue("DERIVATION_KEPT", x.concept_id, f"{x.concept_id!r} was both derived and given a "
                                "shared word; the derivation was kept"))
        elif x.concept_id in usable or x.concept_id in failures:
            notes.append(_issue("DUPLICATE_PLAN_ENTRY", x.concept_id, f"{x.concept_id!r} was given two shared "
                                "words; the first was used"))
        elif concept is None or base is None:
            failures[x.concept_id] = _issue("UNKNOWN_REFERENCE", x.concept_id,
                                            f"{x.concept_id!r} references an unknown concept")
        elif x.base_concept_id == x.concept_id:
            failures[x.concept_id] = _issue("SELF_DERIVATION", x.concept_id, f"{x.concept_id!r} can't share its own word")
        elif x.base_concept_id in extended:
            failures[x.concept_id] = _issue(
                "EXTENSION_CHAIN", x.concept_id,
                f"{x.concept_id} would share the word for {x.base_concept_id}, which itself borrows another "
                "concept's word. Share a word that belongs to its own concept, or use an independent root")
        else:
            # Different parts of speech are fine too (one word for "light" and "to shine").
            usable[x.concept_id] = x
    return list(usable.values()), failures, notes


def _depth_order(entries: list[Planned]) -> list[Planned]:
    """Bases and parts before the words built on them."""
    by_concept = {e.concept_id: e for e in entries}
    return sorted(entries, key=lambda e: _depth(e.concept_id, by_concept))


def build_dictionary(design: LanguageDesign, civilization: Civilization, concepts: list[SemanticConcept],
                     user_words: list[UserWord], plan: Iterable[Planned], seed: int,
                     founding_year: int | None = None,
                     extensions: Iterable[SenseExtension] = ()) -> BuildResult:
    plan = list(plan)
    by_id = {c.id: c for c in concepts}
    user_concepts = {w.concept_id for w in user_words}
    entries, failures, notes = check_plan(plan, by_id, design, user_concepts)
    shared, ext_failures, ext_notes = check_extensions(extensions, by_id, {e.concept_id for e in entries},
                                                       user_concepts)
    failures, notes = {**failures, **ext_failures}, notes + ext_notes
    # A shared word has no word of its own to build from. (Found live: "sky" shared the word for
    # "god", and another word was then built on "sky" -> KeyError.) Point to the real word instead.
    shares = {x.concept_id: x.base_concept_id for x in shared}
    for e in entries:
        if borrowed := next((b for b in e.bases if b in shares), None):
            failures[e.concept_id] = _issue(
                "BASE_IS_SHARED_WORD", e.concept_id,
                f"{e.concept_id} = {e.describe()}, but {borrowed} uses the word for {shares[borrowed]}. "
                f"Build it from {shares[borrowed]} instead, or give it an independent root")
    entries = [e for e in entries if e.concept_id not in failures]

    blocked: set[tuple[str, ...]] = set()
    for _ in range(MAX_REROLL_PASSES):
        result, collisions = _build_once(design, civilization, concepts, user_words, entries, seed,
                                         founding_year, blocked, shared)
        if not collisions:
            break
        # A derived word clashed with an unrelated generated root: give that root a
        # different form and rebuild. Unambiguous, so code does it without asking the LLM.
        blocked |= collisions
    result.failures = {**failures, **result.failures}
    result.notes = notes + result.notes
    return result


def concepts_with_words(state: LanguageState) -> set[str]:
    """Every concept the dictionary expresses, through a word's main or extra senses."""
    return {s.concept_id for lx in state.lexemes for s in lx.senses}


def _build_once(design, civilization, concepts, user_words, entries, seed, founding_year,
                blocked, shared) -> tuple[BuildResult, set[tuple[str, ...]]]:
    year = founding_year if founding_year is not None else civilization.founding_year
    language = Language(id=design.language_id, civilization_id=civilization.id, name=design.name, founding_year=year)
    version = LanguageVersion(language_id=language.id, valid_from_year=year, phonology=design.phonology,
                              grammar=design.grammar, boundary_repair=design.boundary_repair)
    registry = RootRegistry(design.phonology, seed=seed, stream="dictionary-roots")
    engine = MorphologyEngine(language_id=language.id, version_id=version.id, phonology=design.phonology,
                              registry=registry, affixes=design.affixes, rules=design.derivational_rules,
                              boundary_repair=design.boundary_repair, max_depth=MAX_DERIVATION_DEPTH)
    by_id = {c.id: c for c in concepts}
    lexeme_of: dict[str, str] = {}
    failures: dict[str, ValidationIssue] = {}

    notes: list[ValidationIssue] = []

    # 1. User words first: they are fixed and must never collide with anything generated.
    #    The same spelling for two meanings is intentional polysemy: one word, two senses.
    by_form: dict[tuple[str, ...], str] = {}
    for w in user_words:
        c = by_id[w.concept_id]
        owner = by_form.get(tuple(w.phonemes))
        if owner is None:
            lexeme_of[c.id] = engine.add_root(c.id, c.gloss, c.part_of_speech, w.phonemes, user_defined=True).id
            by_form[tuple(w.phonemes)] = c.id
        else:
            _add_sense(engine, lexeme_of[owner], c)
            lexeme_of[c.id] = lexeme_of[owner]
            notes.append(_issue("USER_POLYSEMY", c.id, f"{''.join(w.phonemes)!r} means both {owner!r} and "
                                f"{c.id!r}, as you wrote it"))

    # 2. Generated roots for every concept that isn't derived or sharing another word.
    derived = {e.concept_id for e in entries}
    sharing = {x.concept_id for x in shared}
    root_concepts = [c for c in concepts if c.id not in lexeme_of and c.id not in derived and c.id not in sharing]
    for form in blocked:
        registry.reserve(form, owner_id="blocked", owner_kind="blocked")
    candidates = iter(registry.generate_candidates(len(root_concepts) + len(entries) + CANDIDATE_MARGIN))
    roots_by_form: dict[tuple[str, ...], str] = {}

    def assign_root(c: SemanticConcept) -> None:
        candidate = next(candidates)
        while registry.is_reserved(candidate.phonemes):  # skip any form taken since generation
            candidate = next(candidates)
        lexeme = engine.add_generated_root(c.id, c.gloss, c.part_of_speech, candidate.id)
        lexeme_of[c.id] = lexeme.id
        roots_by_form[tuple(candidate.phonemes)] = c.id

    for c in root_concepts:
        assign_root(c)

    # 3. Derivations and compounds, bases first. A failure becomes a root immediately, so words
    #    built on it still work.
    collisions: set[tuple[str, ...]] = set()
    for e in _depth_order(entries):
        c = by_id[e.concept_id]
        try:
            if isinstance(e, CompoundEntry):
                lexeme_of[c.id] = engine.compound([lexeme_of[b] for b in e.bases], c.id, c.gloss,
                                                  part_of_speech=c.part_of_speech).id
            else:
                lexeme_of[c.id] = engine.derive(lexeme_of[e.base_concept_id], e.rule_id, c.id, c.gloss,
                                                part_of_speech=c.part_of_speech).id
        except MorphologyError as err:
            issue = err.result.errors[0]
            # Use the phoneme list, not the spelled form: "sh" is one phoneme but two letters.
            clashing = tuple(issue.details.get("phonemes", ()))
            if issue.code == "DUPLICATE_FORM" and clashing in roots_by_form:
                collisions.add(clashing)
            failures[c.id] = issue.model_copy(update={
                "message": f"{c.id} = {e.describe()}: {issue.message}", "object_id": c.id})
            assign_root(c)

    # 4. Shared words: the concept becomes an extra sense of its base's word (after derivations,
    #    so a derived word can carry a second meaning too).
    for x in shared:
        _add_sense(engine, lexeme_of[x.base_concept_id], by_id[x.concept_id])
        lexeme_of[x.concept_id] = lexeme_of[x.base_concept_id]

    state = LanguageState(civilization=civilization, language=language, version=version, summary=design.summary,
                          morphemes=list(engine.morphemes.values()), derivational_rules=list(engine.rules.values()),
                          lexemes=list(engine.lexemes.values()), forms=list(engine.forms.values()))
    return BuildResult(state=state, failures=failures, notes=notes), collisions


def _add_sense(engine: MorphologyEngine, lexeme_id: str, concept: SemanticConcept) -> None:
    engine.lexemes[lexeme_id].senses.append(LexicalSense(concept_id=concept.id, gloss=concept.gloss))


def failure_warnings(failures: dict[str, ValidationIssue]) -> list[ValidationIssue]:
    """Unrepaired failures, phrased as what happened to the word."""
    return [f.model_copy(update={"severity": Severity.WARNING,
                                 "message": f"{f.message} -> given its own root instead"})
            for f in failures.values()]
