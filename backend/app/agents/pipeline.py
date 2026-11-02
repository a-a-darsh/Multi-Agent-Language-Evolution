"""Full language generation: design agent, then morphological planner.

A parent LangGraph whose nodes each run a whole sub-workflow. Each agent keeps
its own state schema; the parent passes only what the next one needs (the
validated design, civilization, user words). This keeps agents independent
and testable, and is the same shape later multi-agent orchestration will use.

    START -> design ──failed──> END
                └──ok──> dictionary -> END

Only the home civilization is passed in: neighbors and events affect how a
language changes over time, never its founding state.
"""

from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.language_designer import DesignOutcome, run_language_design
from app.agents.morphological_planner import PlanningOutcome, run_morphological_planning
from app.agents.runner import StepCallback
from app.generation.user_vocabulary import analyze_user_words
from app.llm.chat_model import build_chat_model
from app.llm.settings import LLMSettings
from app.llm.structured import LangChainStructuredLLM, StructuredLLM
from app.models.civilization import Civilization
from app.models.common import DomainModel
from app.models.generation import GenerationRequest


@dataclass
class AgentLLMs:
    """One model per role. Same underlying model, tuned per job."""

    designer: StructuredLLM
    planner: StructuredLLM
    critic: StructuredLLM | None


def agent_llms_from_settings(settings: LLMSettings, use_critic: bool = True) -> AgentLLMs:
    def make(temperature: float, max_tokens: int) -> StructuredLLM:
        return LangChainStructuredLLM(build_chat_model(settings, temperature, max_tokens), settings.model)

    return AgentLLMs(
        designer=make(0.7, 6000),                       # creative
        planner=make(0.3, 6000),                        # analytical
        critic=make(0.0, 3000) if use_critic else None,  # consistent judgments
    )


class GenerationOutcome(DomainModel):
    status: Literal["succeeded", "failed"]
    request: GenerationRequest
    design: DesignOutcome
    dictionary: PlanningOutcome | None = None


class PipelineState(TypedDict, total=False):
    request: GenerationRequest
    design: DesignOutcome
    dictionary: PlanningOutcome


def build_pipeline(llms: AgentLLMs, civilization: Civilization | None = None, on_step: StepCallback | None = None):
    def report(stage: str) -> StepCallback | None:
        return (lambda node: on_step(f"{stage}:{node}")) if on_step else None

    def design(state: PipelineState) -> PipelineState:
        return {"design": run_language_design(state["request"], llms.designer, critic_llm=llms.critic,
                                              civilization=civilization, on_step=report("design"))}

    def dictionary(state: PipelineState) -> PipelineState:
        request, outcome = state["request"], state["design"]
        requirements, _ = analyze_user_words(request.language_preferences.custom_words)
        prefs = request.language_preferences
        return {"dictionary": run_morphological_planning(
            outcome.design, outcome.civilization, requirements.words, request.seed, llms.planner, llms.critic,
            on_step=report("dictionary"), extra_concepts=prefs.custom_concepts, user_notes=prefs.notes,
            derived_ratio=prefs.derived_ratio)}

    graph = StateGraph(PipelineState)
    graph.add_node("design", design)
    graph.add_node("dictionary", dictionary)
    graph.add_edge(START, "design")
    graph.add_conditional_edges("design", lambda s: "dictionary" if s["design"].status == "succeeded" else END,
                                ["dictionary", END])
    graph.add_edge("dictionary", END)
    return graph.compile()


def run_generation(request: GenerationRequest, llms: AgentLLMs, civilization: Civilization | None = None,
                   on_step: StepCallback | None = None) -> GenerationOutcome:
    final = build_pipeline(llms, civilization, on_step).invoke({"request": request})
    dictionary = final.get("dictionary")
    ok = dictionary is not None and dictionary.status == "succeeded"
    return GenerationOutcome(status="succeeded" if ok else "failed", request=request,
                             design=final["design"], dictionary=dictionary)
