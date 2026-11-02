"""The Morphological Planner: decides which words are built from which, as a LangGraph workflow.

The LLM chooses semantic relationships ("sailor = sea + agent"); the
dictionary builder realizes them with the deterministic morphology engine.

    START -> select_concepts -> plan -> build -> critique -> finalize -> END
                                         ^  |       |
                                         |  v       v
                                         repair <---+     (failing entries only, MAX_REPAIRS shared)

Differences from the design agent worth noticing:
- Partial repair: only failing concepts go back to the LLM, through a schema
  that only accepts those ids.
- Graceful degradation: `build` always yields a complete dictionary (failed
  derivations become roots), so running out of repairs means warnings, never failure.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agents.dictionary_schema import PlanEntry, SenseExtension, plan_schema, repair_schema, review_schema
from app.agents.runner import StepCallback, invoke_with_progress
from app.data import load_semantic_inventory
from app.generation.concept_selection import select_concepts
from app.generation.dictionary_builder import BuildResult, build_dictionary, failure_warnings
from app.generation.user_vocabulary import UserWord
from app.llm.structured import LLMCallError, LLMCallRecord, StructuredLLM, recorded_call
from app.models.civilization import Civilization
from app.models.common import DomainModel
from app.models.design import LanguageDesign
from app.models.language import LanguageState
from app.models.lexicon import SemanticConcept
from app.models.validation import Severity, ValidationIssue
from app.validation.morphology_validator import validate_lexicon

MAX_REPAIRS = 2
MAX_EXPANSIONS = 2
DEFAULT_DERIVED_RATIO = 0.2
# Stop expanding once this close to the target (a few critic rejections are expected).
TARGET_TOLERANCE = 0.85


class PlanningRound(DomainModel):
    step: str
    entries: list[PlanEntry] = []
    shared: list[SenseExtension] = []
    issues: list[ValidationIssue] = []


class PlannerState(TypedDict, total=False):
    design: LanguageDesign
    civilization: Civilization
    user_words: list[UserWord]
    seed: int
    extra_concepts: list[SemanticConcept]       # meanings the user invented
    # The user's notes. Not `notes`: that key is the accumulating list of validation
    # notes, and redeclaring it would silently drop its reducer.
    user_notes: str | None
    concepts: list[SemanticConcept]
    plan: dict[str, PlanEntry]                 # concept_id -> entry; absent = independent root
    extensions: dict[str, SenseExtension]      # concept_id -> shares the word of another concept
    build: BuildResult
    issues: list[ValidationIssue]              # what the next repair must fix
    critique_issues: list[ValidationIssue]     # the latest critique's findings, reported at the end
    derived_ratio: float                       # the user's target share of derived words
    target_derived: int                        # ...as a word count for this dictionary
    expansions_used: int
    llm_down: bool                             # the model was unreachable: don't ask it for more
    # Signatures ("base+rule") of derivations the critic already approved, so each
    # round only reviews what's new or changed (a 50% dictionary has ~80 derivations).
    approved: dict[str, str]
    repairs_used: int
    notes: Annotated[list[ValidationIssue], operator.add]
    rounds: Annotated[list[PlanningRound], operator.add]
    llm_calls: Annotated[list[LLMCallRecord], operator.add]
    status: Literal["succeeded", "failed"]
    warnings: list[ValidationIssue]


class PlanningOutcome(DomainModel):
    status: Literal["succeeded", "failed"]
    state: LanguageState | None = None
    plan: list[PlanEntry] = []
    extensions: list[SenseExtension] = []
    warnings: list[ValidationIssue] = []
    issues: list[ValidationIssue] = []
    rounds: list[PlanningRound] = []
    llm_calls: list[LLMCallRecord] = []


# --- prompts -------------------------------------------------------------------

PLANNER_PROMPT = """You are a linguist building the founding dictionary of a fictional language. \
Every concept will get a word. Your job: decide which concepts are DERIVED from another concept \
with one of the language's affixes, so the dictionary has real word families instead of unrelated words. \
Code will build the actual word forms; you only choose relationships.

Rules:
- Derive a concept only when base + affix naturally gives its meaning: sailor = sea + agent, \
harbor = ship + place, fisher = fish + agent, darkness = black + quality. Everything else gets its own root.
- Decide in this order: the concept, then the base whose MEANING it comes from, then the affix whose \
meaning connects them. The parts of speech listed for each affix are only its typical use: any affix \
may attach to any word when the meaning works (wave = water + movement).
- The base must be another concept in the list. A derived word can be the base of another \
derivation, but at most 2 steps from a root. No cycles.
- Concepts marked (user word) keep the user's word and cannot be derived, but they can be bases.
- Let the culture guide which families are rich. Aim for the number of derivations given below.
- Optionally, a few concepts can SHARE another concept's word (extended_senses) when THIS culture \
links the two meanings, e.g. a people whose money is zinc might use one word for zinc and wealth. \
Use this sparingly, only for strong links in this culture. \
Examples here illustrate the idea; they are not suggestions."""


def _concept_lines(concepts: list[SemanticConcept], user_concepts: set[str]) -> str:
    return "\n".join(f"- {c.id} ({c.part_of_speech.value}): {c.gloss}" + (" (user word)" if c.id in user_concepts else "")
                     for c in concepts)


def _rule_lines(design: LanguageDesign) -> str:
    affixes = {m.id: m for m in design.affixes}
    lines = []
    for r in design.derivational_rules:
        m = affixes[r.morpheme_id]
        form = f"{''.join(m.phonemes)}-" if m.type.value == "prefix" else f"-{''.join(m.phonemes)}"
        lines.append(f"- {r.id} ({form}): {r.meaning}. Usually on a {'/'.join(p.value for p in r.input_parts_of_speech)}, "
                     f"usually makes a {r.output_part_of_speech.value}")
    return "\n".join(lines)


def _context(state: PlannerState) -> str:
    civ, design = state["civilization"], state["design"]
    user = {w.concept_id for w in state["user_words"]}
    description = f"\n{civ.description}" if civ.description else ""
    notes = f"\n\n## The user's notes on the language\n{state['user_notes']}" if state.get("user_notes") else ""
    era = f"{civ.era.replace('_', ' ')}, " if civ.era else ""
    return (f"## Civilization\n{era}{civ.geography}, {civ.economy}, {civ.social_organization}; "
            f"traits: {', '.join(civ.cultural_traits) or 'none'}{description}{notes}\n\n"
            f"## Language: {design.name}\n{design.summary}\n\n"
            f"## Derivational affixes\n{_rule_lines(design)}\n\n"
            f"## Concepts ({len(state['concepts'])})\n{_concept_lines(state['concepts'], user)}\n\n"
            f"## Target\nPlan about {state['target_derived']} derivations "
            f"({round(state['derived_ratio'] * 100)}% of the dictionary). Only use derivations whose meaning "
            "really follows; fewer good ones are better than forced ones.")


# --- the graph -----------------------------------------------------------------

def build_planner_graph(llm: StructuredLLM, max_repairs: int = MAX_REPAIRS, critic_llm: StructuredLLM | None = None):
    def ids(state: PlannerState) -> tuple[list[str], list[str]]:
        return [c.id for c in state["concepts"]], [r.id for r in state["design"].derivational_rules]

    def select(state: PlannerState) -> PlannerState:
        required = [w.concept_id for w in state["user_words"]]
        inventory = [*load_semantic_inventory(), *state.get("extra_concepts", [])]
        concepts = select_concepts(inventory, state["civilization"], required)
        return {"concepts": concepts, "repairs_used": 0, "expansions_used": 0, "plan": {}, "extensions": {},
                "approved": {}, "target_derived": round(state["derived_ratio"] * len(concepts))}

    def plan(state: PlannerState) -> PlannerState:
        if state["target_derived"] == 0:
            return {"rounds": [PlanningRound(step="plan")]}  # 0%: every word is its own root, no LLM call
        concept_ids, rule_ids = ids(state)
        user = {w.concept_id for w in state["user_words"]}
        schema = plan_schema([c for c in concept_ids if c not in user], concept_ids, rule_ids,
                             max_derivations=state["target_derived"] + 10)
        messages = [SystemMessage(PLANNER_PROMPT), HumanMessage(_context(state))]
        result, record, error = recorded_call(llm, "plan", messages, schema)
        if error:
            # Without a plan every concept is an independent root: still a valid dictionary.
            code = "LLM_UNAVAILABLE" if isinstance(error, LLMCallError) else "LLM_OUTPUT_INVALID"
            return {"llm_calls": [record], "plan": {}, "llm_down": isinstance(error, LLMCallError), "notes": [ValidationIssue(
                code=code, severity=Severity.WARNING,
                message=f"No derivation plan ({error}); every word was given its own root")]}
        entries: dict[str, PlanEntry] = {}
        for d in result.derivations:
            entries.setdefault(d.concept_id, PlanEntry(**d.model_dump()))  # if listed twice, keep the first
        shared: dict[str, SenseExtension] = {}
        for x in result.extended_senses:
            shared.setdefault(x.concept_id, SenseExtension(**x.model_dump()))
        return {"llm_calls": [record], "plan": entries, "extensions": shared,
                "rounds": [PlanningRound(step="plan", entries=list(entries.values()), shared=list(shared.values()))]}

    def build(state: PlannerState) -> PlannerState:
        # Deterministic: realize the plan. Always produces a complete dictionary.
        result = build_dictionary(state["design"], state["civilization"], state["concepts"],
                                  state["user_words"], state["plan"].values(), state["seed"],
                                  extensions=state["extensions"].values())
        issues = list(result.failures.values())
        return {"build": result, "issues": issues, "notes": result.notes,
                "rounds": [PlanningRound(step="build", issues=issues)]}

    def critique(state: PlannerState) -> PlannerState:
        # Per-derivation plausibility. Narrow, per-item judgments; the verdict is computed by code.
        failed, approved = state["build"].failures, state["approved"]
        realized = [e for e in state["plan"].values()
                    if e.concept_id not in failed and approved.get(e.concept_id) != _signature(e)]
        shared = [x for x in state["extensions"].values()
                  if x.concept_id not in failed and approved.get(x.concept_id) != _signature(x)]
        if critic_llm is None or not (realized or shared):
            return {"issues": [], "critique_issues": []}
        schema = review_schema([e.concept_id for e in realized] + [x.concept_id for x in shared])
        listing = "\n".join([f"- {e.concept_id} = {e.base_concept_id} + {e.rule_id}" for e in realized] +
                            [f"- {x.concept_id}: same word as {x.base_concept_id} ({x.explanation})" for x in shared])
        messages = [SystemMessage(CRITIC_PROMPT),
                    HumanMessage(f"## Affixes\n{_rule_lines(state['design'])}\n\n## Derivations to review\n{listing}")]
        result, record, error = recorded_call(critic_llm, "critique", messages, schema)
        if error:
            return {"llm_calls": [record], "issues": [], "critique_issues": [], "notes": [ValidationIssue(
                code="CRITIC_UNAVAILABLE", severity=Severity.WARNING, message=f"Derivations were not reviewed: {error}")]}
        planned = {e.concept_id: f"{e.concept_id} = {e.base_concept_id} + {e.rule_id}" for e in realized}
        planned |= {x.concept_id: f"{x.concept_id} sharing the word for {x.base_concept_id}" for x in shared}
        issues = [ValidationIssue(
            code="IMPLAUSIBLE_DERIVATION", object_id=r.concept_id, severity=Severity.WARNING,
            message=f"{planned[r.concept_id]} means '{r.literal_meaning}', which is a poor fit for '{r.concept_id}'")
            for r in result.reviews if not r.plausible and r.concept_id in planned]
        rejected = {i.object_id for i in issues}
        newly_approved = {item.concept_id: _signature(item) for item in [*realized, *shared]
                          if item.concept_id not in rejected}
        return {"llm_calls": [record], "issues": issues, "critique_issues": issues,
                "approved": {**approved, **newly_approved},
                "rounds": [PlanningRound(step="critique", issues=issues)]}

    def expand(state: PlannerState) -> PlannerState:
        # Below the target: ask for MORE derivations, only among words that still have their own root.
        taken = set(state["plan"]) | set(state["extensions"]) | set(state["build"].failures)
        taken |= {w.concept_id for w in state["user_words"]}
        concept_ids, rule_ids = ids(state)
        candidates = [c for c in concept_ids if c not in taken]
        wanted = state["target_derived"] - _derived_count(state["build"])
        schema = plan_schema(candidates, concept_ids, rule_ids, max_derivations=wanted + 5)
        messages = [SystemMessage(PLANNER_PROMPT), HumanMessage(_context(state)), HumanMessage(
            f"So far {_derived_count(state['build'])} words are derived, and the target is "
            f"{state['target_derived']}. Plan about {wanted} MORE derivations. Only concepts that still have "
            "their own word are offered; existing derivations stay as they are. Leave extended_senses empty. "
            "Skip any that would be forced: a good derivation matters more than reaching the number.")]
        result, record, error = recorded_call(llm, "expand", messages, schema)
        update: PlannerState = {"llm_calls": [record], "expansions_used": state["expansions_used"] + 1}
        if error:
            return {**update, "llm_down": isinstance(error, LLMCallError)}
        new_plan = dict(state["plan"])
        added = [PlanEntry(**d.model_dump()) for d in result.derivations if d.concept_id not in new_plan]
        new_plan.update({e.concept_id: e for e in added})
        return {**update, "plan": new_plan, "rounds": [PlanningRound(step="expand", entries=added)]}

    def repair(state: PlannerState) -> PlannerState:
        failing = list(dict.fromkeys(i.object_id for i in state["issues"]))
        schema = repair_schema(failing, *ids(state))
        problems = "\n".join(f"- [{i.code}] {i.message}" for i in state["issues"])
        messages = [SystemMessage(PLANNER_PROMPT), HumanMessage(_context(state)), HumanMessage(
            f"These planned words have problems:\n{problems}\n\nFor each of these concepts only, either "
            "derive it differently (a base whose MEANING it really comes from, with a rule that fits), share "
            "another concept's word if the culture truly links them, or give it an independent root. "
            "Everything else is fine and stays as it is.")]
        result, record, error = recorded_call(llm, "repair", messages, schema)
        update: PlannerState = {"llm_calls": [record], "repairs_used": state["repairs_used"] + 1}
        if error:
            return update  # plan unchanged; the next build reports the same failures
        new_plan, new_shared = dict(state["plan"]), dict(state["extensions"])
        for fix in result.fixes:
            new_plan.pop(fix.concept_id, None)
            new_shared.pop(fix.concept_id, None)
            if fix.action == "derive" and fix.base_concept_id and fix.rule_id:
                new_plan[fix.concept_id] = PlanEntry(concept_id=fix.concept_id, base_concept_id=fix.base_concept_id,
                                                     rule_id=fix.rule_id)
            elif fix.action == "extend" and fix.base_concept_id:
                new_shared[fix.concept_id] = SenseExtension(concept_id=fix.concept_id,
                                                            base_concept_id=fix.base_concept_id)
        return {**update, "plan": new_plan, "extensions": new_shared,
                "rounds": [PlanningRound(step="repair", entries=[new_plan[c] for c in failing if c in new_plan],
                                         shared=[new_shared[c] for c in failing if c in new_shared])]}

    def finalize(state: PlannerState) -> PlannerState:
        result, plan, shared = state["build"], dict(state["plan"]), dict(state["extensions"])
        implausible = state.get("critique_issues", [])
        if implausible:
            # Per-item plausibility judgments proved accurate, and an independent root is
            # always a safe fallback, so derivations the critic still rejects are demoted.
            for issue in implausible:
                plan.pop(issue.object_id, None)
                shared.pop(issue.object_id, None)
            result = build_dictionary(state["design"], state["civilization"], state["concepts"],
                                      state["user_words"], plan.values(), state["seed"], extensions=shared.values())
        check = validate_lexicon(result.state, max_depth=2)
        if not check.ok:  # should never happen; guards against bugs in the builder
            return {"status": "failed", "issues": check.errors}
        demoted = [i.model_copy(update={"message": f"{i.message} -> given its own root instead"}) for i in implausible]
        notes = list({(n.code, n.message): n for n in state.get("notes", [])}.values())
        derived, total = _derived_count(result), len(result.state.lexemes)
        if derived < state["target_derived"] * TARGET_TOLERANCE:
            notes.append(ValidationIssue(
                code="DERIVATION_TARGET_MISSED", severity=Severity.WARNING,
                message=(f"{derived} of {total} words ({round(100 * derived / total)}%) are derived, short of the "
                         f"{round(state['derived_ratio'] * 100)}% target: the planner couldn't find more "
                         "derivations that passed review without forcing them")))
        return {"status": "succeeded", "build": result, "plan": plan, "extensions": shared,
                "warnings": failure_warnings(result.failures) + demoted + notes}

    def has_budget(state: PlannerState) -> bool:
        return state["repairs_used"] < max_repairs

    def after_build(state: PlannerState) -> str:
        # Out of budget with failures left: still critique what was built, so implausible
        # derivations are at least reported (an earlier version skipped straight to finalize).
        return "repair" if state["issues"] and has_budget(state) else "critique"

    def needs_more(state: PlannerState) -> bool:
        return (state["expansions_used"] < MAX_EXPANSIONS and not state.get("llm_down")
                and _derived_count(state["build"]) < state["target_derived"] * TARGET_TOLERANCE)

    def after_critique(state: PlannerState) -> str:
        if state["issues"] and has_budget(state):
            return "repair"
        return "expand" if needs_more(state) else "finalize"

    graph = StateGraph(PlannerState)
    for name, fn in [("select_concepts", select), ("plan", plan), ("build", build), ("critique", critique),
                     ("repair", repair), ("expand", expand), ("finalize", finalize)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "select_concepts")
    graph.add_edge("select_concepts", "plan")
    graph.add_edge("plan", "build")
    graph.add_conditional_edges("build", after_build, ["repair", "critique", "finalize"])
    graph.add_conditional_edges("critique", after_critique, ["repair", "expand", "finalize"])
    graph.add_edge("repair", "build")
    graph.add_edge("expand", "build")
    graph.add_edge("finalize", END)
    return graph.compile()


def _signature(item: PlanEntry | SenseExtension) -> str:
    if isinstance(item, SenseExtension):
        return f"shared:{item.base_concept_id}"
    return f"{item.base_concept_id}+{item.rule_id}"


def _derived_count(build: BuildResult) -> int:
    return sum(lx.derivation is not None for lx in build.state.lexemes)


CRITIC_PROMPT = """You review planned word derivations in a fictional language. For each derivation, \
write what base + affix literally means, then judge whether speakers could naturally use that literal \
meaning for the concept. 'person associated with the sea' for sailor: plausible. 'place of fire' for \
kitchen: plausible. 'person associated with bread' for king: not plausible. Many languages use metaphor, \
so accept reasonable extensions; reject only forced or wrong connections. Some entries are SHARED \
words (one word with two meanings): judge whether this culture could plausibly use one word for both, \
e.g. zinc and wealth where zinc is money."""


def run_morphological_planning(design: LanguageDesign, civilization: Civilization, user_words: list[UserWord],
                               seed: int, llm: StructuredLLM, critic_llm: StructuredLLM | None = None,
                               max_repairs: int = MAX_REPAIRS, on_step: StepCallback | None = None,
                               extra_concepts: list[SemanticConcept] = (),
                               user_notes: str | None = None,
                               derived_ratio: float | None = None) -> PlanningOutcome:
    graph = build_planner_graph(llm, max_repairs, critic_llm)
    final = invoke_with_progress(graph, {
        "design": design, "civilization": civilization, "user_words": user_words, "seed": seed,
        "extra_concepts": list(extra_concepts), "notes": [], "rounds": [], "llm_calls": [],
        "derived_ratio": DEFAULT_DERIVED_RATIO if derived_ratio is None else derived_ratio,
    } | ({"user_notes": user_notes} if user_notes else {}), on_step)
    succeeded = final["status"] == "succeeded"
    return PlanningOutcome(
        status=final["status"], state=final["build"].state if succeeded else None,
        plan=list(final["plan"].values()), extensions=list(final.get("extensions", {}).values()),
        warnings=final.get("warnings", []),
        issues=[] if succeeded else final["issues"], rounds=final["rounds"], llm_calls=final["llm_calls"])
