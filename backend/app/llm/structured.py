"""Structured LLM calls with recording and replay.

Agents depend on the small `StructuredLLM` protocol, not on LangChain
directly. That gives three implementations:

- LangChainStructuredLLM: the real thing, via `with_structured_output`.
- ScriptedLLM: returns pre-written outputs, for deterministic tests.
- ReplayLLM: returns outputs recorded from an earlier run, to reproduce it.

Every call produces an LLMCallRecord so generations are auditable.
"""

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ValidationError

from app.models.common import DomainModel

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(Exception):
    """The model responded, but not with valid structured output. Worth retrying."""


class LLMCallError(Exception):
    """The model could not be reached or refused the request. Not worth retrying."""


class LLMCallRecord(DomainModel):
    step: str
    model: str
    messages: list[dict[str, str]]
    output: dict[str, Any] | None = None
    error: str | None = None
    seconds: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StructuredLLM(Protocol):
    model_name: str

    def invoke(self, messages: Sequence[BaseMessage], schema: type[T]) -> T: ...


def message_dicts(messages: Sequence[BaseMessage]) -> list[dict[str, str]]:
    return [{"role": m.type, "content": str(m.content)} for m in messages]


class LangChainStructuredLLM:
    def __init__(self, chat_model: BaseChatModel, model_name: str):
        self.chat_model = chat_model
        self.model_name = model_name

    def invoke(self, messages: Sequence[BaseMessage], schema: type[T]) -> T:
        # with_structured_output converts the Pydantic class to a JSON schema,
        # tells the provider to follow it, and parses the reply back into `schema`.
        runnable = self.chat_model.with_structured_output(schema)
        try:
            result = runnable.invoke(list(messages))
        except (ValidationError, ValueError) as e:
            # Includes LangChain's OutputParserException (a ValueError subclass).
            raise LLMOutputError(f"The model's reply did not match the schema: {e}") from e
        except Exception as e:
            raise LLMCallError(f"Calling {self.model_name} failed: {type(e).__name__}: {e}") from e
        if result is None:
            raise LLMOutputError("The model returned no structured output")
        return result if isinstance(result, schema) else schema.model_validate(result)


class ScriptedLLM:
    """Returns (or raises) each scripted item in turn. Records what it was asked."""

    def __init__(self, responses: Sequence[BaseModel | dict | Exception], model_name: str = "scripted"):
        self.responses = list(responses)
        self.model_name = model_name
        self.received: list[list[BaseMessage]] = []

    def invoke(self, messages: Sequence[BaseMessage], schema: type[T]) -> T:
        self.received.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        # Dicts are handy for run-time (dynamic) schemas; either way the output is validated.
        return schema.model_validate(response if isinstance(response, dict) else response.model_dump())


class ReplayLLM:
    """Replays the outputs of recorded calls, in order, to reproduce a past run."""

    def __init__(self, records: Sequence[LLMCallRecord]):
        self.records = list(records)
        self.model_name = f"replay:{records[0].model}" if records else "replay"

    def invoke(self, messages: Sequence[BaseMessage], schema: type[T]) -> T:
        record = self.records.pop(0)
        if record.output is None:
            raise LLMOutputError(record.error or "recorded call had no output")
        return schema.model_validate(record.output)


def recorded_call(llm: StructuredLLM, step: str, messages: Sequence[BaseMessage],
                  schema: type[T]) -> tuple[T | None, LLMCallRecord, Exception | None]:
    """Invoke and always produce a record, whether the call succeeded or not."""
    start = time.perf_counter()
    try:
        output = llm.invoke(messages, schema)
        error = None
    except (LLMOutputError, LLMCallError) as e:
        output, error = None, e
    record = LLMCallRecord(step=step, model=llm.model_name, messages=message_dicts(messages),
                           output=output.model_dump(mode="json") if output else None,
                           error=str(error) if error else None, seconds=round(time.perf_counter() - start, 2))
    return output, record, error
