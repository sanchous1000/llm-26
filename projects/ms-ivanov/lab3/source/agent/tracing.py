"""Langfuse logging helpers: observations, readable message dumps and costs.

``Tracing`` wraps the Langfuse client so that the agent code never depends on whether
logging is enabled: without a client every observation is a no-op.
"""

import json
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from source.agent.config import ModelConfig

logger = logging.getLogger(__name__)
MAX_LOGGED_CHARS = 20_000  # per tool result in a logged prompt
STATUS_LEVELS = {  # turn status -> Langfuse observation level
    "ok": "DEFAULT",
    "step_limit": "WARNING",
    "timeout": "WARNING",
    "error": "ERROR",
}


class _NullObservation:
    id = None
    trace_id = None

    def update(self, **_: Any) -> "_NullObservation":
        return self

    def end(self, **_: Any) -> None:
        return None

    def start_observation(self, **_: Any) -> "_NullObservation":
        return self


class Tracing:
    """Langfuse client or nothing; ``observe`` nests via the OpenTelemetry context."""

    def __init__(self, client: Any | None = None):
        self.client = client

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @contextmanager
    def observe(self, name: str, as_type: str = "span", **fields: Any) -> Iterator[Any]:
        if self.client is None:
            yield _NullObservation()
            return
        with self.client.start_as_current_observation(
            name=name, as_type=as_type, **fields
        ) as observation:
            yield observation

    def attributes(self, **fields: Any):
        """Trace-level attributes (session, user, tags) for all observations inside."""
        if self.client is None:
            return nullcontext()
        from langfuse import propagate_attributes  # noqa: PLC0415 - optional dependency

        return propagate_attributes(**fields)

    def start(
        self,
        name: str,
        as_type: str = "span",
        *,
        trace_id: str | None = None,
        parent_id: str | None = None,
        **fields: Any,
    ) -> Any:
        """An observation that is not the current context (end it with ``.end()``).

        ``trace_id``/``parent_id`` attach it to an observation created elsewhere
        (another process passes them, e.g. the OpenCode runner to the LLM proxy).
        """
        if self.client is None:
            return _NullObservation()
        context = None
        if trace_id:
            context = {"trace_id": trace_id}
            if parent_id:
                context["parent_span_id"] = parent_id
        return self.client.start_observation(
            trace_context=context, name=name, as_type=as_type, **fields
        )

    def current_trace_id(self) -> str | None:
        return self.client.get_current_trace_id() if self.client else None

    def flush(self) -> None:
        if self.client is not None:
            self.client.flush()


def connect_langfuse(enabled: bool = True) -> Tracing:
    """Tracing with the client configured by LANGFUSE_* variables, if they are set."""
    if not enabled:
        return Tracing()
    try:
        from langfuse import Langfuse  # noqa: PLC0415

        client = Langfuse()
        if not client.auth_check():
            logger.warning("Langfuse keys rejected; tracing is off")
            return Tracing()
        return Tracing(client)
    except Exception as error:  # noqa: BLE001 - tracing must never stop the agent
        logger.warning("Langfuse is unavailable (%s); tracing is off", error)
        return Tracing()


def chat_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """The prompt as OpenAI-style chat messages (what the model saw), for the UI."""
    chat: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            chat.extend(_request_messages(message))
        elif isinstance(message, ModelResponse):
            chat.append(response_message(message))
    return chat


def _request_messages(message: ModelRequest) -> list[dict[str, Any]]:
    result = []
    if message.instructions:
        result.append({"role": "system", "content": message.instructions})
    for part in message.parts:
        if isinstance(part, SystemPromptPart):
            result.append({"role": "system", "content": part.content})
        elif isinstance(part, UserPromptPart):
            result.append({"role": "user", "content": part.content})
        elif isinstance(part, ToolReturnPart):
            result.append(
                {
                    "role": "tool",
                    "name": part.tool_name,
                    "tool_call_id": part.tool_call_id,
                    "content": _clip(_as_text(part.content)),
                }
            )
        elif isinstance(part, RetryPromptPart):
            result.append(
                {
                    "role": "tool",
                    "name": part.tool_name,
                    "content": _as_text(part.content),
                }
            )
    return result


def response_message(message: ModelResponse) -> dict[str, Any]:
    text = "".join(p.content for p in message.parts if isinstance(p, TextPart))
    reasoning = "".join(p.content for p in message.parts if isinstance(p, ThinkingPart))
    calls = [
        {"id": p.tool_call_id, "name": p.tool_name, "arguments": p.args}
        for p in message.parts
        if isinstance(p, ToolCallPart)
    ]
    result: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        result["reasoning"] = reasoning
    if calls:
        result["tool_calls"] = calls
    return result


def usage_details(usage: RequestUsage) -> dict[str, int]:
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_tokens,
        "total": usage.input_tokens + usage.output_tokens,
    }


def cost_details(usage: RequestUsage, model: ModelConfig) -> dict[str, float]:
    """USD cost from the configured list prices (cached input tokens are cheaper)."""
    cached = usage.cache_read_tokens
    fresh = max(usage.input_tokens - cached, 0)
    input_cost = (
        fresh * model.price_input_per_mtok + cached * model.price_cache_read_per_mtok
    ) / 1e6
    output_cost = usage.output_tokens * model.price_output_per_mtok / 1e6
    return {
        "input": input_cost,
        "output": output_cost,
        "total": input_cost + output_cost,
    }


def _as_text(content: Any) -> str:
    return (
        content
        if isinstance(content, str)
        else json.dumps(content, ensure_ascii=False, default=str)
    )


def _clip(text: str) -> str:
    return text if len(text) <= MAX_LOGGED_CHARS else text[:MAX_LOGGED_CHARS] + " [...]"
