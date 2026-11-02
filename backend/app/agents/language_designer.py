"""The Language Design Agent, as a LangGraph workflow.

LangGraph in four ideas, as used here:

1. STATE - a TypedDict that every step reads and updates (`DesignState`).
   A node returns only the keys it changes; LangGraph merges them in.
   Keys annotated with a *reducer* (`Annotated[list, operator.add]`) are
   appended to instead of overwritten - that's how attempts and LLM call
   records accumulate across loop iterations.

2. NODES - plain functions `state -> partial state`. Some call the LLM
   (design, repair); others are ordinary deterministic code (analyze,
   validate, finalize). An "agent" is just a graph where some nodes think.

3. EDGES - fixed transitions (`add_edge`) and *conditional* ones
   (`add_conditional_edges`) where a routing function looks at the state
   and returns the name of the next node. The repair loop is a conditional
   edge pointing backwards.

4. COMPILE + INVOKE - `graph.compile()` checks the wiring and returns a
   runnable; `.invoke(initial_state)` runs it to END and returns the final state.

    START -> analyze_inputs -> design -> validate -> critique -> finalize -> END
                  |                       |   ^        |
                  v                       v   |        v
                 fail <------------------ repair <-----+   (at most MAX_REPAIRS, shared)

Generator-critic: `design`/`repair` generate, `validate` checks structure
with code, and `critique` checks meaning with a second LLM role (plus
deterministic quality checks). Structural errors are fatal once the repair
budget is spent; critic findings are advisory, so a design that still has
them is accepted with warnings rather than thrown away.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agents.design_schema import Critique, LanguageDesignDraft
from app.agents.design_validation import (
    DesignBuildError,
    apply_user_requirements,
    build_design,
    check_design,
    check_quality,
    critique_issues,
)
from app.agents.runner import StepCallback, invoke_with_progress
from app.generation.phoneme_generator import derive_rng
from app.generation.user_vocabulary import PhonologyRequirements, analyze_user_words
from app.generation.world_generator import complete_civilization
from app.llm.structured import LLMCallError, LLMCallRecord, StructuredLLM, recorded_call
from app.models.civilization import Civilization
from app.models.common import DomainModel, new_id
from app.models.design import LanguageDesign
from app.models.eras import format_year
from app.models.generation import GenerationRequest
from app.models.phonology import GRAPHEME_CATALOG, PhonemeClass
from app.models.validation import Severity, ValidationIssue

MAX_REPAIRS = 2


class DesignAttempt(DomainModel):
    """One draft and what validation said about it."""

    step: str
    draft: LanguageDesignDraft | None = None
    issues: list[ValidationIssue] = []


class DesignState(TypedDict, total=False):
    # Inputs
    request: GenerationRequest
    civilization: Civilization
    requirements: PhonologyRequirements
    language_id: str
    # Working memory
    draft: LanguageDesignDraft | None
    issues: list[ValidationIssue]          # problems from the most recent check (what repair must fix)
    warnings: list[ValidationIssue]        # soft findings from validate, passed on to critique
    # Last structurally valid draft and its soft findings, kept in case a
    # critic-requested repair makes things worse.
    fallback_draft: LanguageDesignDraft | None
    fallback_issues: list[ValidationIssue]
    notes: Annotated[list[ValidationIssue], operator.add]   # persistent warnings (e.g. critic unavailable)
    advisory: list[ValidationIssue]        # critic opinions that are reported but don't trigger repair
    # Separate budgets: structural fixes once used up the shared budget, so accurate
    # critic findings (all-inflectional affixes) could never be acted on.
    repairs_used: int                      # structural (validation errors)
    quality_repairs_used: int              # critic / quality findings
    fatal: bool                            # stop now, no retry (e.g. LLM unreachable)
    # Accumulating history (reducers append rather than replace)
    attempts: Annotated[list[DesignAttempt], operator.add]
    llm_calls: Annotated[list[LLMCallRecord], operator.add]
    # Output
    design: LanguageDesign | None
    status: Literal["succeeded", "failed"]


class DesignOutcome(DomainModel):
    status: Literal["succeeded", "failed"]
    civilization: Civilization
    design: LanguageDesign | None = None
    issues: list[ValidationIssue] = []
    # Quality findings still unresolved when the repair budget ran out.
    warnings: list[ValidationIssue] = []
    attempts: list[DesignAttempt] = []
    llm_calls: list[LLMCallRecord] = []


# --- prompts -------------------------------------------------------------------

_VOWELS = "".join(g for g, e in GRAPHEME_CATALOG.items() if e.phoneme_class == PhonemeClass.VOWEL)
_CONSONANTS = " ".join(f"{g}=/{e.ipa}/" for g, e in GRAPHEME_CATALOG.items()
                       if e.phoneme_class == PhonemeClass.CONSONANT)

SYSTEM_PROMPT = f"""You are an expert linguist designing the founding state of a fictional language \
for a simulation. Code will later generate roots and build derived words from your design, so it \
must be precise, internally consistent, and simple enough to work mechanically.

Hard rules:
- The user's explicit choices and words are constraints. Never contradict them. If a user word \
needs a sound or syllable shape, include it.
- Use only these letters. Each letter is exactly one sound.
  Vowels: {_VOWELS}
  Consonants: {_CONSONANTS}
- Syllable shapes are limited to V, CV, VC, CVC. There are no consonant clusters inside a syllable, \
but a coda consonant may be followed by the next syllable's onset (CVC.CV is fine).
- Affixes are attached directly to roots. Make sure they fit: a suffix starting with a vowel next to \
a vowel-final root creates two adjacent vowels, which is only legal if allow_hiatus is true or a \
linking consonant is defined. A suffix starting with a consonant after a consonant-final root needs \
CVC syllables, or a linking vowel.
- Keep the grammar coherent and modest: choose features that naturally go together. Every grammatical \
feature the user asks for must appear in `features`.
- Choose a distinctive sound inventory rather than every available letter, and make affix forms \
distinct from the user's own words.

Let the civilization's culture and environment shape the sound and feel of the language and which \
derivational affixes it has, as tendencies rather than stereotypes. Fill in `reasoning` first."""


def _describe_request(state: DesignState) -> str:
    civ, request, reqs = state["civilization"], state["request"], state["requirements"]
    prefs = request.language_preferences
    lines = [
        "## Civilization",
        f"- Era: {civ.era.replace('_', ' ') if civ.era else 'unknown'}, founded {format_year(civ.founding_year)}",
        f"- Geography: {civ.geography}",
        f"- Economy: {civ.economy}",
        f"- Society: {civ.social_organization}",
        f"- Cultural traits: {', '.join(civ.cultural_traits) or 'none'}",
    ]
    if civ.description:
        lines.append(f"- Description: {civ.description}")

    lines.append("\n## The user's language choices (must be followed)")
    chosen = {
        "Word order": prefs.word_order, "Morphological type": prefs.morphological_type,
        "Grammatical gender": prefs.grammatical_gender, "Case marking": prefs.case_marking,
        "Sound preferences": prefs.sound_preferences,
    }
    explicit = [f"- {k}: {v}" for k, v in chosen.items() if v is not None]
    explicit += [f"- Requested feature: {f}" for f in prefs.special_features]
    lines += explicit or ["- None. You decide everything."]
    if prefs.notes:
        lines.append(f"\n## The user's own notes on the language (follow them where possible)\n{prefs.notes}")
    unchosen = [k for k, v in chosen.items() if v is None]
    if unchosen:
        lines.append(f"\nYou decide: {', '.join(unchosen)}.")

    if reqs.words:
        lines.append("\n## The user's own words (must remain legal exactly as spelled)")
        lines += [f"- {w.concept_id}: {w.spelling} (syllables: {'.'.join(w.syllable_patterns)})" for w in reqs.words]
        lines.append(f"These require the letters {', '.join(reqs.required_phonemes)} and the syllable shapes "
                     f"{', '.join(reqs.required_patterns)}" + ("; and adjacent vowels (allow_hiatus = true)."
                                                               if reqs.requires_hiatus else "."))
    return "\n".join(lines)


def _repair_message(state: DesignState) -> str:
    # Problems go first and are phrased as instructions. An earlier version said
    # "keep everything else the same" after the JSON, and the model just
    # copied its previous answer back unchanged.
    problems = "\n".join(f"{n}. [{i.code}] {i.message}" for n, i in enumerate(state["issues"], 1))
    previous = state.get("draft")
    if previous is None:
        return (f"Your previous reply could not be read as a valid design:\n{problems}\n\n"
                "Return a complete design that follows the schema exactly.")
    return (f"Your previous design was rejected. You must fix these problems:\n{problems}\n\n"
            f"For reference, the rejected design was:\n```json\n{previous.model_dump_json(indent=1)}\n```\n\n"
            "Return a complete new design in which every problem above is fixed. Your answer must differ "
            "from the rejected design wherever a problem points.")


CRITIC_PROMPT = """You are a careful reviewer of fictional-language designs. Another model wrote the \
design below; code has already confirmed it is structurally valid. You check meaning, which code cannot.

For EACH entry in `derivational_affixes`, apply it to an English example, then classify it:
- derivational: makes a different word with its own dictionary entry. Examples: agent \
(sea -> sailor), place (bake -> bakery), instrument (cut -> knife), diminutive (house -> hut), \
quality (dark -> darkness), collective (tree -> forest), noun-to-verb, noun-to-adjective.
- inflectional: the same word, marked for a grammatical category. Examples: case (nominative, \
accusative, dative, genitive, locative...), plural, tense, aspect, mood, evidentiality (witnessed / \
reported), person, agreement.
Judge by what the affix does, not by how it is labeled.

Also say whether the consonant inventory looks deliberately chosen, and report any grammar choices \
that contradict each other. Unusual but possible combinations are fine."""


_CRITIC_FIELDS = {"name", "consonants", "vowels", "syllable_patterns", "word_order", "morphological_type",
                  "adposition_type", "features", "derivational_affixes"}


def _critic_message(state: DesignState) -> str:
    design_json = state["draft"].model_dump_json(include=_CRITIC_FIELDS, indent=1)
    return (f"The language is for this civilization and these user choices:\n\n{_describe_request(state)}\n\n"
            f"## Design to review\n```json\n{design_json}\n```")


# --- the graph ---------------------------------------------------------------

def build_design_graph(llm: StructuredLLM, max_repairs: int = MAX_REPAIRS, critic_llm: StructuredLLM | None = None):
    """Wire up the workflow. LLMs are injected so tests can use scripted fakes.

    `critic_llm=None` skips the LLM critic (deterministic quality checks still run).
    """

    def analyze_inputs(state: DesignState) -> DesignState:
        # Deterministic node: work out what user words require before asking the LLM anything.
        request = state["request"]
        requirements, result = analyze_user_words(request.language_preferences.custom_words)
        # Use the world's civilization when given, so the language and the world share one object.
        civilization = state.get("civilization") or complete_civilization(
            request.civilization, derive_rng(request.seed, "world", "focal"))
        return {"requirements": requirements, "civilization": civilization, "issues": result.errors,
                "language_id": new_id(), "repairs_used": 0, "quality_repairs_used": 0, "fatal": False}

    def call_llm(state: DesignState, step: str, messages: list[BaseMessage]) -> DesignState:
        draft, record, error = recorded_call(llm, step, messages, LanguageDesignDraft)
        update: DesignState = {"draft": draft, "llm_calls": [record]}
        if error:
            # A malformed reply is retryable; an unreachable model is not.
            code = "LLM_UNAVAILABLE" if isinstance(error, LLMCallError) else "LLM_OUTPUT_INVALID"
            update["issues"] = [ValidationIssue(code=code, message=str(error))]
            update["fatal"] = isinstance(error, LLMCallError)
        return update

    def design(state: DesignState) -> DesignState:
        # LLM node: first attempt.
        return call_llm(state, "design", [SystemMessage(SYSTEM_PROMPT), HumanMessage(_describe_request(state))])

    def repair(state: DesignState) -> DesignState:
        # LLM node: same conversation, plus its previous answer and exactly what was wrong.
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(_describe_request(state)),
                    HumanMessage(_repair_message(state))]
        structural = any(i.severity == Severity.ERROR for i in state["issues"])
        counter = "repairs_used" if structural else "quality_repairs_used"
        return {**call_llm(state, "repair", messages), counter: state[counter] + 1}

    def validate(state: DesignState) -> DesignState:
        # Deterministic node: the LLM proposes, code decides.
        step = "repair" if state["repairs_used"] or state["quality_repairs_used"] else "design"
        draft = state.get("draft")
        if draft is None:  # the LLM call itself failed; its issue is already in state
            return {"attempts": [DesignAttempt(step=step, issues=state["issues"])]}
        # Unambiguous fixes are made by code before judging the draft.
        draft, adjustments = apply_user_requirements(draft, state["requirements"])
        warnings: list[ValidationIssue] = []
        try:
            built = build_design(draft, state["language_id"])
        except DesignBuildError as e:
            issues = e.result.errors
        else:
            result = check_design(built, draft, state["request"], state["requirements"])
            result.extend(check_quality(built, state["requirements"]))
            issues = result.errors
            warnings = [i for i in result.issues if i.severity == Severity.WARNING]
        return {"draft": draft, "issues": issues, "warnings": warnings, "notes": adjustments,
                "attempts": [DesignAttempt(step=step, draft=draft, issues=issues)]}

    def critique(state: DesignState) -> DesignState:
        # Second LLM role. Its findings join the deterministic quality warnings;
        # together they become the next repair's to-do list.
        findings = list(state.get("warnings", []))
        update: DesignState = {}
        if critic_llm is not None:
            messages = [SystemMessage(CRITIC_PROMPT), HumanMessage(_critic_message(state))]
            result, record, error = recorded_call(critic_llm, "critique", messages, Critique)
            update["llm_calls"] = [record]
            if error:
                # The critic is advisory: if it fails, note it and carry on with the deterministic findings.
                update["notes"] = [ValidationIssue(code="CRITIC_UNAVAILABLE", severity=Severity.WARNING,
                                                   message=f"The design was not reviewed: {error}")]
            else:
                actionable, advisory = critique_issues(result, state["draft"])
                findings += actionable
                update["advisory"] = advisory  # latest review's opinions; reported, never acted on
        return {**update, "issues": findings, "fallback_draft": state["draft"], "fallback_issues": findings,
                "attempts": [DesignAttempt(step="critique", issues=findings)]}

    def finalize(state: DesignState) -> DesignState:
        issues = state.get("issues", [])
        if any(i.severity == Severity.ERROR for i in issues):
            # The last (critic-requested) repair broke the structure: keep the last valid version.
            draft = state["fallback_draft"]
            leftovers = state["fallback_issues"] + [ValidationIssue(
                code="REPAIR_REVERTED", severity=Severity.WARNING,
                message="The final revision broke the design's structure, so the previous valid version was kept")]
        else:
            draft, leftovers = state["draft"], issues  # anything left is soft: keep as warnings
        # Notes accumulate on every validate pass; report each distinct one once.
        notes = list({(n.code, n.message): n for n in state.get("notes", [])}.values())
        return {"draft": draft, "design": build_design(draft, state["language_id"]), "status": "succeeded",
                "warnings": leftovers + notes + state.get("advisory", [])}

    def fail(state: DesignState) -> DesignState:
        return {"status": "failed"}

    # Routing functions: look at state, return the name of the next node.
    def after_analyze(state: DesignState) -> str:
        return "fail" if state["issues"] else "design"

    def after_validate(state: DesignState) -> str:
        if not state["issues"]:
            return "critique"
        if state.get("fatal") or state["repairs_used"] >= max_repairs:
            # Out of tries. If an earlier draft was structurally valid, fall back to it.
            return "finalize" if state.get("fallback_draft") else "fail"
        return "repair"

    def after_critique(state: DesignState) -> str:
        # Soft findings have their own budget; running out of it accepts rather than fails.
        if state["issues"] and state["quality_repairs_used"] < max_repairs:
            return "repair"
        return "finalize"

    graph = StateGraph(DesignState)
    for name, fn in [("analyze_inputs", analyze_inputs), ("design", design), ("validate", validate),
                     ("critique", critique), ("repair", repair), ("finalize", finalize), ("fail", fail)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "analyze_inputs")
    graph.add_conditional_edges("analyze_inputs", after_analyze, ["design", "fail"])
    graph.add_edge("design", "validate")
    graph.add_conditional_edges("validate", after_validate, ["critique", "repair", "finalize", "fail"])
    graph.add_conditional_edges("critique", after_critique, ["repair", "finalize"])
    graph.add_edge("repair", "validate")
    graph.add_edge("finalize", END)
    graph.add_edge("fail", END)
    return graph.compile()


def run_language_design(request: GenerationRequest, llm: StructuredLLM, max_repairs: int = MAX_REPAIRS,
                        critic_llm: StructuredLLM | None = None, civilization: Civilization | None = None,
                        on_step: StepCallback | None = None) -> DesignOutcome:
    graph = build_design_graph(llm, max_repairs, critic_llm)
    inputs = {"request": request, "attempts": [], "llm_calls": [], "notes": []}
    if civilization is not None:
        inputs["civilization"] = civilization
    final = invoke_with_progress(graph, inputs, on_step)
    succeeded = final["status"] == "succeeded"
    return DesignOutcome(
        status=final["status"], civilization=final["civilization"], design=final.get("design"),
        issues=[] if succeeded else final["issues"], warnings=final.get("warnings", []) if succeeded else [],
        attempts=final["attempts"], llm_calls=final["llm_calls"],
    )
