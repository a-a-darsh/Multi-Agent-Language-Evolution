"""Run a compiled LangGraph and report progress as nodes finish.

`graph.stream(..., stream_mode=["updates", "values"])` yields
("updates", {node_name: changes}) after every node and ("values", state)
with the full state; the last "values" item is the final state. That is
what powers the GUI's live progress display.
"""

from collections.abc import Callable
from typing import Any

StepCallback = Callable[[str], None]


def invoke_with_progress(graph, inputs: dict[str, Any], on_step: StepCallback | None = None) -> dict[str, Any]:
    if on_step is None:
        return graph.invoke(inputs)
    final: dict[str, Any] = dict(inputs)
    for mode, chunk in graph.stream(inputs, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node in chunk:
                on_step(node)
        else:
            final = chunk
    return final
