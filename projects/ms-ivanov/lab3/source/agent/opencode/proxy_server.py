"""Running the LLM proxy: in the foreground, in this event loop, or in a thread.

Every runner first checks ``/health`` and reuses a proxy that is already up (for
example one started with ``python -m source llm-proxy``).
"""

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

import httpx
import uvicorn

from source.agent.config import AgentConfig
from source.agent.opencode.proxy import create_app
from source.agent.tracing import Tracing

logger = logging.getLogger(__name__)
START_TIMEOUT_S = 10
POLL_INTERVAL_S = 0.1


def health_url(config: AgentConfig) -> str:
    return f"http://{config.proxy.host}:{config.proxy.port}/health"


def is_up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2, trust_env=False).status_code == 200
    except httpx.HTTPError:
        return False


async def is_up_async(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
            return (await client.get(url)).status_code == 200
    except httpx.HTTPError:
        return False


def serve(config: AgentConfig, tracing: Tracing) -> None:
    """Run the proxy in the foreground (``python -m source llm-proxy``)."""
    started = time.monotonic()
    logger.info(
        "LLM proxy on http://%s:%d/v1 -> %s",
        config.proxy.host,
        config.proxy.port,
        config.model.base_url,
    )
    try:
        _server(config, tracing).run()
    finally:
        tracing.flush()
        logger.info("proxy stopped after %.0fs", time.monotonic() - started)


@asynccontextmanager
async def running_proxy(config: AgentConfig, tracing: Tracing) -> AsyncIterator[None]:
    """A proxy served by the current event loop for the duration of the block."""
    url = health_url(config)
    if await is_up_async(url):
        yield
        return
    server = _server(config, tracing)
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(int(START_TIMEOUT_S / POLL_INTERVAL_S)):
            if await is_up_async(url):
                break
            await asyncio.sleep(POLL_INTERVAL_S)
        else:
            raise RuntimeError(f"LLM proxy did not start on {url}")
        yield
    finally:
        server.should_exit = True
        await task


@contextmanager
def proxy_in_thread(config: AgentConfig, tracing: Tracing) -> Iterator[None]:
    """A proxy in a background thread (for callers that own no event loop)."""
    url = health_url(config)
    if is_up(url):
        yield
        return
    server = _server(config, tracing)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(int(START_TIMEOUT_S / POLL_INTERVAL_S)):
            if is_up(url):
                break
            time.sleep(POLL_INTERVAL_S)
        else:
            raise RuntimeError(f"LLM proxy did not start on {url}")
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=START_TIMEOUT_S)


def _server(config: AgentConfig, tracing: Tracing) -> uvicorn.Server:
    app = create_app(config, tracing)
    return uvicorn.Server(
        uvicorn.Config(
            app, host=config.proxy.host, port=config.proxy.port, log_level="warning"
        )
    )
