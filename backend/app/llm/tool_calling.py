"""Tool-calling LLM calls, with recording and scripted replies for tests.

The structured-output agents (design, dictionary) ask the model to fill in one
schema. A tool-using agent instead lets the model choose its next action: each
reply may call one or more tools, whose results are sent back as ToolMessages.

Agents depend on the small `ToolCallingLLM` protocol:
- LangChainToolLLM: the real thing, via `bind_tools` (works with Ollama,
  OpenAI, Anthropic and OpenRouter models that support tool calling).
- ScriptedToolLLM: returns pre-written replies, for deterministic tests.

`invoke_structured` is the fallback for models that sometimes describe a tool
call in text instead of making it (local models via Ollama, which also
ignores `tool_choice`): structured output is decoded against a JSON schema,
so it cannot "forget" to produce the call.
"""

import json
import time
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from app.llm.structured import LangChainStructuredLLM, LLMCallError, LLMCallRecord, LLMOutputError

T = TypeVar("T", bound=BaseModel)


class ToolSpec(BaseModel):
    name: str
    description: str
    args: type[BaseModel]

    def as_openai_tool(self) -> dict[str, Any]:
        schema = self.args.model_json_schema()
        schema.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": schema}}


class ToolCallingLLM(Protocol):
    model_name: str

    def invoke(self, messages: Sequence[BaseMessage], tools: Sequence[ToolSpec]) -> AIMessage: ...

    def invoke_structured(self, messages: Sequence[BaseMessage], schema: type[T]) -> T: ...


class LangChainToolLLM:
    def __init__(self, chat_model: BaseChatModel, model_name: str):
        self.chat_model = chat_model
        self.model_name = model_name

    def invoke(self, messages: Sequence[BaseMessage], tools: Sequence[ToolSpec]) -> AIMessage:
        try:
            reply = self.chat_model.bind_tools([t.as_openai_tool() for t in tools]).invoke(list(messages))
        except Exception as e:
            raise LLMCallError(f"Calling {self.model_name} failed: {type(e).__name__}: {e}") from e
        return reply if isinstance(reply, AIMessage) else AIMessage(content=str(reply.content))

    def invoke_structured(self, messages: Sequence[BaseMessage], schema: type[T]) -> T:
        return LangChainStructuredLLM(self.chat_model, self.model_name).invoke(messages, schema)


class ScriptedToolLLM:
    """Returns each scripted reply in turn. A reply is an AIMessage, an Exception, or a shorthand dict:
    {"tool": name, "args": {...}} for one tool call, {"calls": [(name, args), ...]} for several,
    or {"text": "..."} for a plain answer."""

    def __init__(self, replies: Sequence[AIMessage | dict | Exception], model_name: str = "scripted-tools"):
        self.replies = list(replies)
        self.model_name = model_name
        self.received: list[list[BaseMessage]] = []
        self._ids = 0

    def invoke_structured(self, messages: Sequence[BaseMessage], schema: type[T]) -> T:
        """Scripted as {"structured": {...}}; anything else means the model produced nothing usable."""
        self.received.append(list(messages))
        reply = self.replies.pop(0) if self.replies else None
        if isinstance(reply, dict) and "structured" in reply:
            return schema.model_validate(reply["structured"])
        if reply is not None:
            self.replies.insert(0, reply)  # not meant for this call
        raise LLMOutputError("no scripted structured reply")

    def invoke(self, messages: Sequence[BaseMessage], tools: Sequence[ToolSpec]) -> AIMessage:
        self.received.append(list(messages))
        if not self.replies:
            raise AssertionError("ScriptedToolLLM ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, AIMessage):
            return reply
        if "text" in reply:
            return AIMessage(content=reply["text"])
        calls = reply.get("calls") or [(reply["tool"], reply.get("args", {}))]
        tool_calls = []
        for name, args in calls:
            self._ids += 1
            tool_calls.append({"name": name, "args": args, "id": f"call_{self._ids}", "type": "tool_call"})
        return AIMessage(content=reply.get("content", ""), tool_calls=tool_calls)


def _message_dict(m: BaseMessage) -> dict[str, str]:
    content = m.content if isinstance(m.content, str) else json.dumps(m.content)
    if isinstance(m, AIMessage) and m.tool_calls:
        calls = "; ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in m.tool_calls)
        content = f"{content}\n[tool calls] {calls}".strip()
    return {"role": m.type, "content": content}


def recorded_tool_call(llm: ToolCallingLLM, step: str, messages: Sequence[BaseMessage],
                       tools: Sequence[ToolSpec]) -> tuple[AIMessage | None, LLMCallRecord, Exception | None]:
    """Invoke and always produce a record. Only the newest message is stored per call; the whole
    conversation is the concatenation of the records, which keeps run files small."""
    start = time.perf_counter()
    try:
        reply, error = llm.invoke(messages, tools), None
    except LLMCallError as e:
        reply, error = None, e
    output = None
    if reply is not None:
        output = {"content": reply.content if isinstance(reply.content, str) else json.dumps(reply.content),
                  "tool_calls": [{"name": c["name"], "args": c["args"]} for c in reply.tool_calls]}
    record = LLMCallRecord(step=step, model=llm.model_name, messages=[_message_dict(m) for m in messages[-1:]],
                           output=output, error=str(error) if error else None,
                           seconds=round(time.perf_counter() - start, 2))
    return reply, record, error
