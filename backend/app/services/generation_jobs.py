"""Background generation jobs.

A full generation takes minutes with a local model, far too long for one HTTP
request. So POST returns a job id immediately, the pipeline runs on a worker
thread, every finished graph node is recorded as progress, and the browser
polls for status. One worker: a single local GPU can't run two at once anyway.
"""

import logging
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor

from app.agents.pipeline import AgentLLMs, run_generation
from app.generation.world_generator import generate_world
from app.models.generation import GenerationRequest
from app.models.validation import ValidationIssue
from app.persistence.repository import Repository

log = logging.getLogger(__name__)


class GenerationService:
    def __init__(self, repository: Repository, llm_factory: Callable[[], tuple[AgentLLMs, dict]],
                 executor: Executor | None = None):
        """`llm_factory` returns the agent models plus a description of them for provenance."""
        self.repository = repository
        self.llm_factory = llm_factory
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="generation")

    def start(self, request: GenerationRequest) -> str:
        llms, model_info = self.llm_factory()  # fail fast (e.g. no LLM configured) before queuing
        gid = self.repository.create_generation(request, model_info)
        self.executor.submit(self._run, gid, request, llms)
        return gid

    def _run(self, gid: str, request: GenerationRequest, llms: AgentLLMs) -> None:
        try:
            world = generate_world(request)
            self.repository.record_step(gid, "world")
            outcome = run_generation(request, llms, civilization=world.focal,
                                     on_step=lambda step: self.repository.record_step(gid, step))
            if outcome.status == "succeeded":
                self.repository.complete_generation(gid, outcome, world)
            else:
                errors = outcome.design.issues if outcome.dictionary is None else outcome.dictionary.issues
                self.repository.fail_generation(gid, errors, outcome)
        except Exception as e:  # never leave a job stuck in "running"
            log.exception("generation %s crashed", gid)
            self.repository.fail_generation(gid, [ValidationIssue(code="INTERNAL_ERROR", message=str(e))])
