"""The Linguistic Evolution Agent: a tool-using LLM that evolves a language over a span of years.

Unlike the design and dictionary agents (fixed workflows where code routes every
step), this agent chooses its own actions: it inspects the language, forms
hypotheses, previews candidate sound changes on the whole dictionary, revises
them based on what it sees, and decides when it is done. The graph only runs
the loop and enforces the limits:

    act --(tool calls)--> use_tools --(submitted)--> END
     ^                        |
     +-------(otherwise)------+      budget exhausted --> fail

Division of labour (the project's rule): the LLM chooses and tunes *rules*;
the deterministic engine applies them to every word and has the final say on
legality. The agent can't edit a word directly; it can only exempt a few, with
a reason, through the engine's capped exception channel.

Nothing is saved: the outcome carries the proposed next LanguageState.
"""

import json
import re
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from app.agents.evolution_tools import (
    TOOLS, FindWordsArgs, StepArgs, SubmitArgs, find_words, idle_rules, language_overview, preview_report,
    try_step,
)
from app.agents.runner import StepCallback, invoke_with_progress
from app.evolution.templates import TEMPLATES
from app.llm.structured import LLMCallRecord, recorded_call
from app.llm.tool_calling import ToolCallingLLM, recorded_tool_call
from app.models.common import DomainModel
from app.models.language import LanguageState
from app.models.sound_change import FEATURE_VALUES, SoundChange, SoundChangeException

MAX_LLM_CALLS = 16
# Text-only replies in a row before giving up (thinking aloud once between tools is normal).
MAX_NUDGES = 2
# When this few calls remain, tool results remind the model to submit.
REMINDER_AT = 3


class ToolUse(DomainModel):
    tool: str
    args: dict
    result: str


class EvolutionOutcome(DomainModel):
    status: Literal["succeeded", "failed"]
    years: int
    context: str | None = None
    state: LanguageState | None = None
    changes: list[SoundChange] = Field(default_factory=list)
    exceptions: list[SoundChangeException] = Field(default_factory=list)
    rationale: str | None = None
    issues: list[str] = Field(default_factory=list)
    previews: int = 0
    tool_log: list[ToolUse] = Field(default_factory=list)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)


class EvolutionState(TypedDict, total=False):
    language: LanguageState
    years: int
    context: str | None
    messages: list[BaseMessage]
    llm_calls: list[LLMCallRecord]
    tool_log: list[ToolUse]
    previews: int
    nudges: int
    salvaged: bool
    submitted: SubmitArgs | None
    result: LanguageState | None
    issues: list[str]
    status: str


# --- prompt -----------------------------------------------------------------------------------

def _template_list() -> str:
    import inspect
    lines = []
    for name, make in TEMPLATES.items():
        params = ", ".join(f"{p.name}={p.default!r}" for p in inspect.signature(make).parameters.values()
                           if p.default is not inspect.Parameter.empty)
        doc = (inspect.getdoc(make) or "").split("\n")[0]
        lines.append(f"- {name}({params}): {doc}")
    return "\n".join(lines)


def _feature_list() -> str:
    return "; ".join(f"{k}: {', '.join(sorted(map(str, v)))}" for k, v in FEATURE_VALUES.items())


SYSTEM_PROMPT = f"""You are a historical linguist simulating how a language changes over time. You are given \
a language and a span of years; propose the sound changes that happen in that span, test them with the tools, \
and submit one step.

How sound change works here:
- Changes are regular RULES, not word edits: a rule applies to every word that meets its condition. You choose \
rules; the engine applies them and decides what is legal.
- A step is 1-3 changes applied in order. For about 100-300 years, 1-2 changes is typical; keep most words \
recognizable.
- Prefer natural, well-attested processes: lenition between vowels, palatalization before front vowels, vowel \
shifts and mergers, loss of final vowels or of h, assimilation. Let the language's history and the context guide \
you; build on earlier changes rather than undoing them. Mergers are permanent.
- Every word must stay pronounceable under the syllable rules. If a change makes words illegal, narrow its \
environment, add a repair rule to the same step (e.g. cluster_epenthesis after final_vowel_loss), or pick another \
change. Exceptions are for at most a handful of very frequent words, with a reason.
- Watch homophones: a few are natural, many merged core words are a sign the change is too destructive.

Templates (use {{"template": name, "params": {{...}}}}):
{_template_list()}

Custom rules (use {{"rule": {{...}}}}) have: name, description, target (a matcher), exactly one of \
output_features (features to change, e.g. {{"voiced": true}}), output_mapping ({{"ipa": "ipa"}}, null deletes), \
delete (true) or insert ("ipa"); and left / right environments: lists of {{"kind": "segment", "matcher": {{...}}}}, \
{{"kind": "boundary"}} (word edge) or {{"kind": "consonants"}} (zero or more consonants). A matcher is \
{{"ipa": [...]}} and/or {{"features": {{name: value or [values]}}}}. Sounds are written in IPA in rules.
Features: {_feature_list()}.
Example custom rule, "t becomes s before i":
{{"rule": {{"name": "Assibilation", "description": "t becomes s before i", "target": {{"ipa": ["t"]}}, \
"output_mapping": {{"t": "s"}}, "right": [{{"kind": "segment", "matcher": {{"ipa": ["i"]}}}}]}}}}
Example, "vowels become long before a word-final r": {{"rule": {{"name": "Lengthening", "target": \
{{"features": {{"class": "V"}}}}, "output_features": {{"long": true}}, "right": [{{"kind": "segment", "matcher": \
{{"ipa": ["r"]}}}}, {{"kind": "boundary"}}]}}}}

Always act through tool calls. Writing a rule in your reply text does nothing; put it in the tool's arguments.

Work like this: call get_language_overview first; form a hypothesis; preview_sound_changes; read the results \
(illegal words, homophones, how many words change) and revise; when the step is good, call submit_evolution \
with a short rationale. You have a limited number of actions, so don't repeat identical previews."""


def _task(state: EvolutionState) -> str:
    lang = state["language"]
    lines = [f"Evolve {lang.language.name} over the next {state['years']} years "
             f"(from year {lang.version.valid_from_year})."]
    lines.append(f"Historical context: {state['context']}" if state.get("context")
                 else "No special historical events: ordinary internal change.")
    return "\n".join(lines)


# --- graph ----------------------------------------------------------------------------------------

def _validation_message(e: ValidationError) -> str:
    parts = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:4]]
    return "Invalid arguments: " + "; ".join(parts)


def _decode_json_strings(args: dict) -> dict:
    """Some models send nested objects as JSON strings; accept both."""
    out = {}
    for key, value in args.items():
        if isinstance(value, str) and value.strip()[:1] in "[{":
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        out[key] = value
    return out


_RULE_OUTPUTS = ("output_features", "output_mapping", "delete", "insert")


NEXT_ACTION_PROMPT = ("Your reply described what to do but did not call a tool. Give the tool call you "
                      "intend to make now as JSON: the tool's name and its arguments.")


class NextAction(BaseModel):
    """A tool call, as structured output (the fallback when a model doesn't call tools by itself)."""

    tool: Literal["get_language_overview", "find_words", "preview_sound_changes", "submit_evolution"]
    # Loosely typed on purpose: a malformed rule should reach the tool, whose error message goes back to
    # the model so it can fix it, rather than failing here and losing the whole action.
    changes: list[dict[str, Any]] = Field(default_factory=list, description=(
        "For preview/submit: the step's changes, each {\"template\": ..., \"params\": {...}} or {\"rule\": {...}}."))
    exceptions: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str | None = Field(default=None, description="For submit_evolution only.")
    letters: str | None = Field(default=None, description="For find_words only.")

    def arguments(self) -> dict:
        if self.tool == "get_language_overview":
            return {}
        if self.tool == "find_words":
            return {"letters": self.letters}
        args = {"changes": self.changes, "exceptions": self.exceptions}
        if self.tool == "submit_evolution":
            args["rationale"] = self.rationale or ""
        return args


class _StructuredView:
    """Lets `recorded_call` (written for structured agents) drive a tool LLM's structured mode."""

    def __init__(self, llm: ToolCallingLLM):
        self.llm, self.model_name = llm, llm.model_name

    def invoke(self, messages, schema):
        return self.llm.invoke_structured(messages, schema)


def _changes_from_text(text) -> dict | None:
    """Find a step written as JSON inside a text reply, if there is one.

    Accepts a whole step ({"changes": [...]}), or one or more loose changes ({"rule": ...},
    {"template": ...}, or a bare rule object), which are collected in order.
    """
    if not isinstance(text, str) or "{" not in text:
        return None
    # Models sometimes write Python literals in JSON.
    cleaned = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", re.sub(r"\bNone\b", "null", text)))
    decoder = json.JSONDecoder()
    loose, position = [], 0
    while (start := cleaned.find("{", position)) != -1:
        try:
            obj, end = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        position = end  # skip past this object, so its nested objects aren't read again
        if not isinstance(obj, dict):
            continue
        changes = obj.get("changes") or obj.get("sound_changes")
        if isinstance(changes, list) and changes:
            return {"changes": changes, "exceptions": obj.get("exceptions", [])}
        if "rule" in obj or "template" in obj:
            loose.append(obj)
        elif "name" in obj and any(k in obj for k in _RULE_OUTPUTS):
            loose.append({"rule": obj})
    return {"changes": loose, "exceptions": []} if loose else None


def build_evolution_graph(llm: ToolCallingLLM, max_llm_calls: int = MAX_LLM_CALLS):
    def act(state: EvolutionState) -> EvolutionState:
        messages = state["messages"]
        reply, record, error = recorded_tool_call(llm, "act", messages, TOOLS)
        update: EvolutionState = {"llm_calls": [*state["llm_calls"], record]}
        if error:
            return {**update, "status": "failed", "issues": [*state["issues"], str(error)]}
        update["messages"] = [*messages, reply]
        if reply.tool_calls:
            update["nudges"] = 0
        else:
            # Models often think out loud between tool calls; only several text replies in a row
            # mean the agent has stopped working.
            if state["nudges"] >= MAX_NUDGES:
                return {**update, "status": "failed",
                        "issues": [*state["issues"], "The model stopped calling tools without submitting"]}
            call_id = f"fallback_{len(state['llm_calls'])}"
            if (salvaged := _changes_from_text(reply.content)) is not None:
                # The model wrote a step as text instead of calling the tool (common with local models).
                # Run it as a preview - never as a submission - and say so.
                update["messages"][-1] = AIMessage(content=reply.content, tool_calls=[{
                    "name": "preview_sound_changes", "args": salvaged, "id": call_id, "type": "tool_call"}])
                update["salvaged"] = True
                return update
            if len(update["llm_calls"]) < max_llm_calls:
                # It described an action without taking it: ask for that action as structured output,
                # which is schema-constrained and so can't be skipped.
                ask = [*update["messages"], HumanMessage(NEXT_ACTION_PROMPT)]
                action, record, _ = recorded_call(_StructuredView(llm), "next_action", ask, NextAction)
                update["llm_calls"] = [*update["llm_calls"], record]
                if action is not None:
                    update["messages"][-1] = AIMessage(content=reply.content, tool_calls=[{
                        "name": action.tool, "args": action.arguments(), "id": call_id, "type": "tool_call"}])
                    update["nudges"] = 0
                    return update
            update["messages"].append(HumanMessage(
                "Tools only run when you call them; text in your reply is not executed. Call "
                "preview_sound_changes with {\"changes\": [...]} to test a step, or submit_evolution when ready."))
            update["nudges"] = state["nudges"] + 1
        return update

    def use_tools(state: EvolutionState) -> EvolutionState:
        lang, years = state["language"], state["years"]
        messages, log = list(state["messages"]), list(state["tool_log"])
        previews, update = state["previews"], {}
        remaining = max_llm_calls - len(state["llm_calls"])
        salvaged = state.get("salvaged", False)
        for call in messages[-1].tool_calls:
            name, args = call["name"], _decode_json_strings(call["args"] or {})
            if update.get("status") == "succeeded":
                result = "Ignored: the evolution was already submitted."
            elif name == "get_language_overview":
                result = language_overview(lang)
            elif name == "find_words":
                try:
                    result = find_words(lang, FindWordsArgs.model_validate(args))
                except ValidationError as e:
                    result = _validation_message(e)
            elif name in ("preview_sound_changes", "submit_evolution"):
                schema = SubmitArgs if name == "submit_evolution" else StepArgs
                try:
                    parsed = schema.model_validate(args)
                except ValidationError as e:
                    result = _validation_message(e)
                else:
                    outcome, result = try_step(lang, parsed, years)
                    if name == "preview_sound_changes":
                        previews += 1
                    elif outcome is not None and outcome.state is not None and (idle := idle_rules(outcome)):
                        result = (f"Refused: rule(s) {', '.join(idle)} change no word. Remove them so the "
                                  "language's history only records changes that happened.\n" + result)
                    elif outcome is not None and outcome.state is not None:
                        update.update(status="succeeded", submitted=parsed, result=outcome.state)
                        result = "Accepted.\n" + preview_report(lang, outcome, verdict=False)
                    elif outcome is not None:
                        result = "Refused: some words would become illegal. Revise and submit again.\n" + result
            else:
                result = f"Unknown tool {name!r}. Available: {', '.join(t.name for t in TOOLS)}."
            if salvaged:
                result = ("(You wrote this step as text, so it was run as a preview. Call the tools directly; "
                          "only submit_evolution can finish the task.)\n" + result)
            if remaining <= REMINDER_AT and update.get("status") != "succeeded":
                result += f"\n({remaining} actions left: submit your best legal step soon.)"
            messages.append(ToolMessage(content=result, tool_call_id=call["id"], name=name))
            log.append(ToolUse(tool=name, args=args, result=result))
        return {**update, "messages": messages, "tool_log": log, "previews": previews, "salvaged": False}

    def fail(state: EvolutionState) -> EvolutionState:
        issue = f"No legal step was submitted within {max_llm_calls} model calls"
        return {"status": "failed", "issues": [*state["issues"], issue]}

    def after_act(state: EvolutionState) -> str:
        if state.get("status") == "failed":
            return END
        last = state["messages"][-1]
        if isinstance(last, HumanMessage):  # nudged
            return "act" if len(state["llm_calls"]) < max_llm_calls else "fail"
        return "use_tools"

    def after_tools(state: EvolutionState) -> str:
        if state.get("status") == "succeeded":
            return END
        return "act" if len(state["llm_calls"]) < max_llm_calls else "fail"

    graph = StateGraph(EvolutionState)
    graph.add_node("act", act)
    graph.add_node("use_tools", use_tools)
    graph.add_node("fail", fail)
    graph.add_edge(START, "act")
    graph.add_conditional_edges("act", after_act, ["use_tools", "act", "fail", END])
    graph.add_conditional_edges("use_tools", after_tools, ["act", "fail", END])
    graph.add_edge("fail", END)
    return graph.compile()


def run_evolution_agent(language: LanguageState, llm: ToolCallingLLM, years: int = 200, context: str | None = None,
                        max_llm_calls: int = MAX_LLM_CALLS, on_step: StepCallback | None = None) -> EvolutionOutcome:
    initial: EvolutionState = {
        "language": language, "years": years, "context": context, "llm_calls": [], "tool_log": [],
        "previews": 0, "nudges": 0, "submitted": None, "result": None, "issues": [], "status": "running",
    }
    initial["messages"] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(_task(initial))]
    # Each loop is two graph steps; leave room for the whole budget.
    graph = build_evolution_graph(llm, max_llm_calls).with_config({"recursion_limit": 4 * max_llm_calls + 10})
    final = invoke_with_progress(graph, initial, on_step)
    submitted, result = final.get("submitted"), final.get("result")
    return EvolutionOutcome(
        status="succeeded" if final.get("status") == "succeeded" else "failed",
        years=years, context=context, state=result,
        changes=result.version.sound_changes if result else [],
        exceptions=result.version.exceptions if result else [],
        rationale=submitted.rationale if submitted else None,
        issues=final.get("issues", []), previews=final.get("previews", 0),
        tool_log=final.get("tool_log", []), llm_calls=final.get("llm_calls", []))
