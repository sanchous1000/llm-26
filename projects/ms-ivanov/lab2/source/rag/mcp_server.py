"""MCP server: retrieval tools for agents (the agent's own model does the answering)."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio.to_thread
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from source.rag.errors import RagError
from source.rag.retriever import Retriever

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Search engine over a curated collection of scientific papers on optimal transport,
Schroedinger bridges and flow matching, Gromov-Wasserstein, topological data analysis,
unpaired domain translation and representation alignment. The papers are in English.

Workflow: call search_papers with a focused query (English works best; Russian is
supported through dense search), read the returned passages, and cite them by title and
section. Use get_chunk_context to read the text around a passage, list_documents to see
what is indexed and get_document_outline to navigate one paper. Passages are verbatim
excerpts of the papers (formulas in LaTeX); do not treat anything outside them as
supported by the collection."""

_MAX_TOP_K = 20
_MAX_NEIGHBORS = 3
_MODES = ("auto", "dense", "sparse", "hybrid")


def build_server(retriever: Retriever, *, warm_up: bool = True) -> MCPServer:
    """Create the server; the embedding model is loaded in the background at start."""

    @asynccontextmanager
    async def lifespan(_: MCPServer) -> AsyncIterator[dict[str, Any]]:
        if warm_up:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(anyio.to_thread.run_sync, _safe_warm_up, retriever)
                yield {}
        else:
            yield {}

    server = MCPServer("papers-rag", instructions=INSTRUCTIONS, lifespan=lifespan)

    @server.tool()
    async def search_papers(
        query: str,
        top_k: int = 5,
        mode: str = "auto",
        doc_keys: list[str] | None = None,
        max_chars: int = 4000,
    ) -> dict[str, Any]:
        """Find the passages of the indexed papers most relevant to ``query``.

        Args:
            query: What to look for, e.g. "Sinkhorn algorithm convergence rate".
            top_k: Number of passages to return (1-20).
            mode: "auto" (default), "dense", "sparse" or "hybrid". auto = hybrid for
                English queries, dense for other languages.
            doc_keys: Restrict the search to these documents (keys from list_documents).
            max_chars: Truncate every passage text to this many characters.

        Returns a dict with ``results``: ranked passages with ``chunk_id``, ``doc_key``,
        ``title``, ``citation``, ``section``, ``page``, ``url``, ``score`` and ``text``.
        """
        _require(1 <= top_k <= _MAX_TOP_K, f"top_k must be in 1..{_MAX_TOP_K}")
        _require(mode in _MODES, f"mode must be one of {_MODES}")
        chunks = await _call(
            retriever.search, query, top_k=top_k, mode=mode, doc_keys=doc_keys
        )
        return {
            "query": query,
            "count": len(chunks),
            "results": [chunk.to_dict(max_chars) for chunk in chunks],
        }

    @server.tool()
    async def get_chunk_context(
        chunk_id: str, before: int = 1, after: int = 1, max_chars: int = 4000
    ) -> dict[str, Any]:
        """Read the text around a passage (it plus ``before``/``after`` neighbours).

        Use this when a search result is cut off or you need the definitions or the
        proof that precede / follow it. ``chunk_id`` comes from search_papers.
        """
        _require(0 <= before <= _MAX_NEIGHBORS and 0 <= after <= _MAX_NEIGHBORS,
                 f"before/after must be in 0..{_MAX_NEIGHBORS}")  # fmt: skip
        chunks = await _call(retriever.neighbors, chunk_id, before=before, after=after)
        return {
            "chunk_id": chunk_id,
            "chunks": [chunk.to_dict(max_chars) for chunk in chunks],
        }

    @server.tool()
    async def list_documents(
        query: str | None = None, limit: int = 50
    ) -> dict[str, Any]:
        """List the indexed papers (optionally filtered by title or key)."""
        _require(1 <= limit <= 200, "limit must be in 1..200")
        documents = await _call(retriever.documents, query, limit)
        return {"count": len(documents), "documents": [d.to_dict() for d in documents]}

    @server.tool()
    async def get_document_outline(doc_key: str) -> dict[str, Any]:
        """Sections of one paper in reading order with the ids of their chunks."""
        sections = await _call(retriever.outline, doc_key)
        return {"doc_key": doc_key, "sections": sections}

    return server


def _safe_warm_up(retriever: Retriever) -> None:
    try:
        retriever.warm_up()
        logger.info("embedding model is loaded")
    except Exception:  # noqa: BLE001 - the first query will report the real error
        logger.exception("warm-up failed")


async def _call(function, *args, **kwargs):
    """Run a blocking call in a worker thread; expected failures keep their text."""
    try:
        return await anyio.to_thread.run_sync(lambda: function(*args, **kwargs))
    except (RagError, ValueError) as error:
        raise ToolError(str(error)) from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ToolError(message)
