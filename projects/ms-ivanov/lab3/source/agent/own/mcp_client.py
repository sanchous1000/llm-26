"""Client of the retrieval MCP server (papers-rag) over streamable HTTP."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)
CONNECT_ATTEMPTS = 4
CONNECT_BACKOFF_S = 2.0
MAX_CONCURRENT_CALLS = 3  # the shared server allows ~10 requests/s per IP
SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]


class McpToolError(RuntimeError):
    """The server rejected the call (bad arguments) or could not be reached."""


class ToolBackend(Protocol):
    """What the agent tools need from the retrieval server (a fake in tests)."""

    server_name: str

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class McpBackend:
    """Client of the MCP server: a short MCP session per call, retried on failure.

    A long-lived session dies with its transport (a dropped TLS connection cancels the
    whole turn); one session per call keeps a network failure inside that call, where
    the agent sees it as a tool error. ``async with`` checks that the server answers.
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None,
        name: str,
        timeout_s: float,
        session_factory: SessionFactory | None = None,
    ):
        self.server_name = name
        self._url = url
        self._token = token
        self._timeout_s = timeout_s
        self._factory = session_factory or self._open_session
        self._slots: asyncio.Semaphore | None = None  # created in the running loop

    async def __aenter__(self) -> "McpBackend":
        await self._with_retries("connect", lambda session: session.send_ping())
        return self

    async def __aexit__(self, *exc_info) -> None:
        return None

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._with_retries(
            tool,
            lambda session: session.call_tool(
                tool, arguments, read_timeout_seconds=self._timeout_s
            ),
        )
        text = "".join(getattr(c, "text", "") for c in result.content or [])
        if result.is_error:
            raise McpToolError(text or f"{tool} failed")
        if result.structured_content:
            return dict(result.structured_content)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}

    async def _with_retries(self, what: str, action) -> Any:
        if self._slots is None:
            self._slots = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            try:
                async with self._slots, self._factory() as session:
                    return await action(session)
            except Exception as error:  # noqa: BLE001 - transport, TLS, timeouts
                if attempt == CONNECT_ATTEMPTS:
                    raise McpToolError(
                        f"{self.server_name} unavailable ({what}): {_describe(error)}"
                    ) from error
                logger.warning("MCP %s attempt %d failed: %s", what, attempt, error)
                await asyncio.sleep(CONNECT_BACKOFF_S * attempt)
        raise AssertionError("unreachable")

    @asynccontextmanager
    async def _open_session(self) -> AsyncIterator[ClientSession]:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        async with (
            httpx2.AsyncClient(headers=headers, timeout=self._timeout_s) as client,
            streamable_http_client(  # no DELETE on close: one request less per call
                self._url, http_client=client, terminate_on_close=False
            ) as streams,
            ClientSession(streams[0], streams[1]) as session,
        ):
            await session.initialize()
            yield session


def _describe(error: BaseException) -> str:
    """The first real error of an exception group (anyio wraps transport errors)."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return f"{type(error).__name__}: {error}"
