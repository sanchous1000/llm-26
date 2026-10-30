"""Prometheus metrics of the MCP service (scraped from a separate port)."""

import functools
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from prometheus_client import Counter, Histogram, start_http_server

logger = logging.getLogger(__name__)

TOOL_CALLS = Counter(
    "rag_mcp_tool_calls_total", "MCP tool calls by outcome", ["tool", "status"]
)
TOOL_SECONDS = Histogram(
    "rag_mcp_tool_duration_seconds",
    "Time spent in an MCP tool call",
    ["tool"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

T = TypeVar("T")


def instrumented(tool: str) -> Callable[[Callable[..., Awaitable[T]]], Any]:
    """Count and time every call of an async MCP tool; the signature is kept."""

    def decorate(function: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(function)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            started = time.perf_counter()
            status = "ok"
            try:
                return await function(*args, **kwargs)
            except BaseException:
                status = "error"
                raise
            finally:
                TOOL_SECONDS.labels(tool).observe(time.perf_counter() - started)
                TOOL_CALLS.labels(tool, status).inc()

        return wrapper

    return decorate


def serve_metrics(port: int) -> None:
    """Expose ``/metrics`` on ``port`` (all interfaces; keep it off public networks)."""
    start_http_server(port)
    logger.info("metrics are served on port %d", port)
