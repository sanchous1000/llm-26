"""OpenAI-compatible LLM proxy that logs a ready-made agent (OpenCode) to Langfuse.

OpenCode talks to ``http://127.0.0.1:4100/v1`` as to an OpenAI-compatible provider. The
proxy adds the real API key, forwards the request to the provider (streaming is passed
through chunk by chunk) and logs:

- every model call as a *generation* (prompt, parameters, answer, usage, cost, latency);
- the tool results that arrived since the previous call as *tool* observations (OpenCode
  ran them itself, through MCP), with a nested *retriever* for ``search_papers``.

Headers set by the runner (``eval``/CLI) bind the observations to its trace:
``x-lab3-session`` (session id), ``x-lab3-trace`` and ``x-lab3-parent`` (trace and
parent observation id), ``x-lab3-tags`` (comma-separated) and ``x-lab3-user``.
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic_ai.usage import RequestUsage
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from source.agent.config import AgentConfig, require_env
from source.agent.opencode.proxy_stream import (
    StreamAccumulator,
    code_tool_calls,
    completion_message,
    new_tool_results,
    retrieved_chunks,
    usage_from,
)
from source.agent.tracing import Tracing, cost_details

logger = logging.getLogger(__name__)
TOOL_ERROR_MARKERS = ("Error executing tool", "MCP error", "error:")
MAX_LOGGED_CHARS = 20_000
DEFAULT_TAGS = ("agent:opencode",)
LOGGED_PARAMETERS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stream",
)
COMPACTION_MARKERS = (  # OpenCode 2.0.25 compaction prompt (see its binary)
    "summarize only what the user and the assistant said",
    "update the existing checkpoint",
    "required summary template",
)


@dataclass(frozen=True)
class RequestMeta:
    session: str
    trace_id: str | None
    parent_id: str | None
    tags: tuple[str, ...]
    user: str

    @classmethod
    def from_headers(cls, headers: Any, default_tags: tuple[str, ...]) -> "RequestMeta":
        tags = tuple(t for t in (headers.get("x-lab3-tags") or "").split(",") if t)
        return cls(
            session=headers.get("x-lab3-session") or "opencode-manual",
            trace_id=headers.get("x-lab3-trace") or None,
            parent_id=headers.get("x-lab3-parent") or None,
            tags=tags or default_tags,
            user=headers.get("x-lab3-user") or "anonymous",
        )


@dataclass(frozen=True)
class PendingCall:
    """A model call in flight: its generation and the key its usage is counted under."""

    generation: Any
    key: str
    name: str = "llm"


class ProxyLogger:
    """Turns proxied requests and responses into Langfuse observations.

    It also sums usage per trace (or session): ``GET /usage/<key>`` serves the totals to
    the runner, because OpenCode's own events miss the usage of some steps.
    """

    def __init__(self, config: AgentConfig, tracing: Tracing):
        self.config = config
        self.tracing = tracing
        self.totals: dict[str, dict[str, float]] = {}

    def start_generation(self, meta: RequestMeta, body: dict[str, Any]) -> PendingCall:
        messages = body.get("messages") or []
        with self.tracing.attributes(
            session_id=meta.session,
            user_id=meta.user,
            tags=list(meta.tags),
            version=self.config.langfuse.version,
        ):
            self._log_tools(meta, messages)
            generation = self.tracing.start(
                _generation_name(messages),
                "generation",
                trace_id=meta.trace_id,
                parent_id=meta.parent_id,
                model=body.get("model") or self.config.model.name,
                model_parameters=_parameters(body),
                input=_clip(messages),
                metadata={"proxy": True, "tools_offered": len(body.get("tools") or [])},
            )
        return PendingCall(
            generation, meta.trace_id or meta.session, _generation_name(messages)
        )

    def finish(  # noqa: PLR0913 - everything known about one response
        self,
        pending: PendingCall,
        *,
        message: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        completion_start: datetime | None = None,
        error: str | None = None,
    ) -> None:
        fields: dict[str, Any] = {}
        if message is not None:
            fields["output"] = message
        if usage:
            details = usage_from(usage)
            fields["usage_details"] = details
            fields["cost_details"] = cost_details(
                RequestUsage(
                    input_tokens=details["input"],
                    output_tokens=details["output"],
                    cache_read_tokens=details["cache_read_input_tokens"],
                ),
                self.config.model,
            )
            self._count(pending, details, fields["cost_details"]["total"])
        if completion_start is not None:
            fields["completion_start_time"] = completion_start
        fields["metadata"] = {"finish_reason": finish_reason}
        if error:
            fields.update(level="ERROR", status_message=error[:2000])
        pending.generation.update(**fields)
        pending.generation.end()

    def _count(self, pending: PendingCall, usage: dict[str, int], cost: float) -> None:
        total = self.totals.setdefault(
            pending.key,
            {"llm_calls": 0, "input": 0, "output": 0, "cached": 0, "cost_usd": 0.0,
             "compactions": 0},
        )  # fmt: skip
        total["compactions"] += int(pending.name == "compaction")
        total["llm_calls"] += 1
        total["input"] += usage["input"]
        total["output"] += usage["output"]
        total["cached"] += usage["cache_read_input_tokens"]
        total["cost_usd"] += cost

    def _log_tools(self, meta: RequestMeta, messages: list[dict[str, Any]]) -> None:
        for exchange in new_tool_results(messages):
            calls = _calls_of(exchange)
            server, _, tool = (calls[0] if calls else exchange.name).partition(".")
            failed = any(
                marker in exchange.result[:200] for marker in TOOL_ERROR_MARKERS
            )
            span = self.tracing.start(
                f"tool:{exchange.name}",
                "tool",
                trace_id=meta.trace_id,
                parent_id=meta.parent_id,
                input=exchange.arguments,
                output=_clip(exchange.result),
                metadata={
                    "tool": tool or exchange.name,
                    "mcp_server": server if tool else None,
                    "mcp_calls": calls,
                    "tool_call_id": exchange.call_id,
                    "logged_by": "proxy (result seen in the next model request)",
                },
                level="ERROR" if failed else None,
                status_message=exchange.result[:500] if failed else None,
            )
            searched = any(call.endswith("search_papers") for call in calls)
            chunks = retrieved_chunks(exchange.result) if searched else None
            if chunks is not None:
                span.start_observation(
                    name="retrieval",
                    as_type="retriever",
                    input=exchange.arguments,
                    output=chunks,
                    metadata={"source": f"MCP {server}", "count": len(chunks)},
                ).end()
            span.end()


def create_app(
    config: AgentConfig,
    tracing: Tracing,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    """The proxy app; ``transport`` replaces the network in tests."""
    routes = _ProxyRoutes(config, tracing, transport)

    @asynccontextmanager
    async def lifespan(_: Starlette):
        try:
            yield
        finally:
            await routes.close()

    return Starlette(
        routes=[
            Route("/v1/chat/completions", routes.chat, methods=["POST"]),
            Route("/v1/models", routes.models, methods=["GET"]),
            Route("/usage/{key}", routes.usage, methods=["GET"]),
            Route("/health", routes.health, methods=["GET"]),
        ],
        lifespan=lifespan,
    )


class _ProxyRoutes:
    """Request handlers: forward to the provider, log through ``ProxyLogger``."""

    def __init__(
        self,
        config: AgentConfig,
        tracing: Tracing,
        transport: httpx.AsyncBaseTransport | None,
    ):
        self.upstream = config.model.base_url.rstrip("/")
        self.api_key = require_env(config.model.api_key_env)
        self.tracing = tracing
        self.logger = ProxyLogger(config, tracing)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.model.request_timeout_s * 3),
            transport=transport,
        )

    async def close(self) -> None:
        await self.client.aclose()
        self.tracing.flush()

    async def chat(self, request: Request) -> Response:
        body = await request.json()
        meta = RequestMeta.from_headers(request.headers, DEFAULT_TAGS)
        stream = bool(body.get("stream"))
        if stream:  # usage arrives in the last chunk only when asked for
            body["stream_options"] = {
                **(body.get("stream_options") or {}),
                "include_usage": True,
            }
        pending = self.logger.start_generation(meta, body)
        outgoing = self.client.build_request(
            "POST",
            f"{self.upstream}/chat/completions",
            json=body,
            headers=self._headers(meta.session),
        )
        try:
            response = await self.client.send(outgoing, stream=stream)
        except httpx.HTTPError as error:
            self.logger.finish(pending, error=f"upstream unreachable: {error}")
            return JSONResponse({"error": {"message": str(error)}}, status_code=502)
        if response.status_code >= 400:
            return await self._failed(response, pending)
        if not stream:
            return self._complete(response.json(), pending)
        return StreamingResponse(
            _relay(response, pending, self.logger),
            media_type=response.headers.get("content-type", "text/event-stream"),
        )

    async def models(self, _: Request) -> Response:
        response = await self.client.get(
            f"{self.upstream}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return Response(
            response.content, response.status_code, media_type="application/json"
        )

    async def usage(self, request: Request) -> Response:
        return JSONResponse(self.logger.totals.get(request.path_params["key"], {}))

    async def health(self, _: Request) -> Response:
        return JSONResponse({"status": "ok", "upstream": self.upstream})

    def _headers(self, session: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "x-opencode-session": session,  # required by the opencode provider
        }

    async def _failed(self, response: httpx.Response, pending: PendingCall) -> Response:
        content = await response.aread()
        await response.aclose()
        self.logger.finish(pending, error=content.decode("utf-8", "replace"))
        return Response(content, response.status_code, media_type="application/json")

    def _complete(self, data: dict[str, Any], pending: PendingCall) -> Response:
        choice = (data.get("choices") or [{}])[0]
        self.logger.finish(
            pending,
            message=completion_message(data),
            usage=data.get("usage"),
            finish_reason=choice.get("finish_reason"),
        )
        return JSONResponse(data)


async def _relay(
    response: httpx.Response, pending: PendingCall, proxy_logger: ProxyLogger
):
    accumulator = StreamAccumulator()
    first_chunk: datetime | None = None
    error = None
    try:
        # aiter_bytes decodes any content-encoding: the client receives none
        async for chunk in response.aiter_bytes():
            if first_chunk is None:
                first_chunk = datetime.now(UTC)
            accumulator.feed(chunk)
            yield chunk
    except Exception as failure:  # noqa: BLE001 - logged, the client sees a cut stream
        error = f"stream interrupted: {failure}"
        raise
    finally:
        await response.aclose()
        accumulator.close()
        proxy_logger.finish(
            pending,
            message=accumulator.message(),
            usage=accumulator.usage,
            finish_reason=accumulator.finish_reason,
            completion_start=first_chunk,
            error=error,
        )


def _calls_of(exchange) -> list[str]:
    """MCP calls made by one tool call: OpenCode wraps them in code (``execute``)."""
    if exchange.name == "execute" and isinstance(exchange.arguments, dict):
        return code_tool_calls(str(exchange.arguments.get("code") or ""))
    server, _, tool = exchange.name.partition("_")
    return [f"{server}.{tool}"] if tool else []


def _generation_name(messages: list[dict[str, Any]]) -> str:
    """OpenCode also asks the model for session titles and summaries: name them."""
    system = " ".join(
        str(m.get("content")) for m in messages if m.get("role") == "system"
    ).lower()
    last = str(messages[-1].get("content") if messages else "").lower()
    if any(marker in last for marker in COMPACTION_MARKERS):
        return "compaction"
    if "title" in system[:400]:
        return "title"
    return "llm"


def _parameters(body: dict[str, Any]) -> dict[str, Any]:
    return {k: body[k] for k in LOGGED_PARAMETERS if k in body}


def _clip(value: Any) -> Any:
    text = str(value)
    return (
        value
        if len(text) <= MAX_LOGGED_CHARS * 4
        else text[: MAX_LOGGED_CHARS * 4] + " [...]"
    )
