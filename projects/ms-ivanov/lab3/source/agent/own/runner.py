"""Wiring: config -> MCP connection -> dialogue session; running a whole dialogue."""

import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from source.agent.config import (
    DEFAULT_AGENT_CONFIG,
    AgentConfig,
    load_agent_config,
    load_env_file,
    load_prompts,
    require_env,
)
from source.agent.own.agent import AgentFactory, DialogSession, TurnResult
from source.agent.own.mcp_client import McpBackend
from source.agent.tracing import Tracing, connect_langfuse


def new_session_id(prefix: str = "chat") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def build_factory(
    config_path: Path = DEFAULT_AGENT_CONFIG,
    overrides: Sequence[str] = (),
    *,
    tracing: Tracing | None = None,
) -> AgentFactory:
    load_env_file()
    config = load_agent_config(config_path, list(overrides))
    prompts = load_prompts(config, config_path)
    return AgentFactory(config, prompts, tracing or connect_langfuse())


def mcp_backend(config: AgentConfig) -> McpBackend:
    token = require_env(config.mcp.token_env) if config.mcp.token_env else None
    return McpBackend(
        config.mcp.url,
        token=token,
        name=config.mcp.name,
        timeout_s=config.mcp.timeout_s,
    )


@asynccontextmanager
async def open_session(
    factory: AgentFactory, session_id: str, user_id: str | None = None
) -> AsyncIterator[DialogSession]:
    async with mcp_backend(factory.config) as backend:
        yield factory.session(backend, session_id, user_id)


async def run_dialog(
    factory: AgentFactory,
    turns: Sequence[str],
    *,
    session_id: str,
    on_turn: Callable[[int, str, TurnResult], None] | None = None,
) -> list[TurnResult]:
    """All user turns of one dialogue in one session (one Langfuse session)."""
    results: list[TurnResult] = []
    try:
        async with open_session(factory, session_id) as session:
            for index, text in enumerate(turns):
                result = await session.ask(text)
                results.append(result)
                if on_turn is not None:
                    on_turn(index, text, result)
    finally:
        factory.tracing.flush()
    return results
