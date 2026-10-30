"""PydanticAI capability that manages the context and logs model and tool calls.

Hooks:
- ``wrap_model_request``: measures the prompt, shortens the history when it exceeds
  the limit (a ``context-reduction`` span per event, a child span per step), then logs
  the model call as a Langfuse *generation* (prompt, response, tokens, cost, errors);
- ``wrap_tool_execute``: logs every tool call as a *tool* observation with the MCP
  server name, arguments, result and errors.
"""

from dataclasses import asdict, dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai.tools import RunContext

from source.agent.config import ModelConfig
from source.agent.own.context import ContextManager, ContextResult
from source.agent.own.tools import AgentDeps
from source.agent.tracing import (
    chat_messages,
    cost_details,
    response_message,
    usage_details,
)

MAX_LOGGED_RESULT_CHARS = 20_000


@dataclass
class AgentRuntime(AbstractCapability[AgentDeps]):
    model: ModelConfig
    context: ContextManager | None = None
    label: str = "llm"

    async def wrap_model_request(
        self, ctx: RunContext[AgentDeps], *, request_context, handler
    ):
        deps = ctx.deps
        context_tokens = None
        if self.context is not None:
            context_tokens = await self._manage_context(ctx, request_context)
            deps.state.steps += 1
        with deps.tracing.observe(
            self.label,
            as_type="generation",
            model=self.model.name,
            model_parameters={
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_output_tokens,
            },
            input=chat_messages(request_context.messages),
            metadata={
                "step": deps.state.steps,
                "context_tokens_estimate": context_tokens,
                "context_limit": (
                    self.context.config.limit_tokens if self.context else None
                ),
            },
        ) as generation:
            try:
                response: ModelResponse = await handler(request_context)
            except Exception as error:
                generation.update(level="ERROR", status_message=_describe(error))
                raise
            cost = cost_details(response.usage, self.model)
            generation.update(
                output=response_message(response),
                usage_details=usage_details(response.usage),
                cost_details=cost,
                metadata={"finish_reason": response.finish_reason},
            )
        state = deps.state
        state.llm_calls += 1
        state.input_tokens += response.usage.input_tokens
        state.output_tokens += response.usage.output_tokens
        state.cost_usd += cost["total"]
        return response

    async def wrap_tool_execute(
        self, ctx: RunContext[AgentDeps], *, call, tool_def, args, handler
    ):
        deps = ctx.deps
        deps.state.tool_calls += 1
        deps.state.tools_used.append(call.tool_name)
        server = (tool_def.metadata or {}).get("mcp_server")
        with deps.tracing.observe(
            f"tool:{call.tool_name}",
            as_type="tool",
            input=call.args,
            metadata={
                "tool": call.tool_name,
                "mcp_server": server,
                "transport": "mcp/streamable-http" if server else "local",
                "tool_call_id": call.tool_call_id,
                "retry": ctx.retry,
            },
        ) as span:
            try:
                result = await handler(args)
            except Exception as error:
                span.update(level="ERROR", status_message=_describe(error))
                raise
            span.update(output=_clip(result))
        return result

    async def _manage_context(self, ctx: RunContext[AgentDeps], request_context) -> int:
        deps, manager = ctx.deps, self.context
        state = manager.state(request_context.messages)
        if not state.over_limit:
            deps.state.context_tokens.append(state.tokens)
            return state.tokens
        with deps.tracing.observe(
            "context-reduction",
            input={
                "strategy": manager.config.strategy,
                "tokens": state.tokens,
                "limit": state.limit,
                "messages": state.messages,
                "turns": state.turns,
            },
        ) as span:
            result = await manager.process(request_context.messages)
            _log_steps(deps, result)
            still_over = result.after.over_limit
            span.update(
                output={
                    "tokens": result.after.tokens,
                    "messages": result.after.messages,
                    "turns": result.after.turns,
                    "steps": [r.kind for r in result.reductions],
                },
                level="WARNING" if still_over else "DEFAULT",
                status_message=(
                    "context still exceeds the limit" if still_over else None
                ),
            )
        request_context.messages = list(result.messages)
        ctx.messages[:] = result.messages  # the shortened history persists
        deps.state.reductions.extend(asdict(r) for r in result.reductions)
        deps.state.context_tokens.append(result.after.tokens)
        return result.after.tokens


def _log_steps(deps: AgentDeps, result: ContextResult) -> None:
    for reduction in result.reductions:
        with deps.tracing.observe(
            f"reduce:{reduction.kind}",
            input={"tokens": reduction.tokens_before},
            output={
                "tokens": reduction.tokens_after,
                "removed": list(reduction.removed),
                "summary": reduction.summary,
            },
        ):
            pass


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:2000]


def _clip(value: Any) -> Any:
    text = str(value)
    return (
        value
        if len(text) <= MAX_LOGGED_RESULT_CHARS
        else text[:MAX_LOGGED_RESULT_CHARS] + " [...]"
    )
