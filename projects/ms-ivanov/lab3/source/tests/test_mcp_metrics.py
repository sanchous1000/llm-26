"""Prometheus metrics of the MCP service."""

import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from prometheus_client import REGISTRY

from source.rag.mcp_server import build_server

# ---- MCP tool metrics ---------------------------------------------------------


def sample(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_successful_tool_call_is_counted_and_timed(retriever):
    retr, _ = retriever
    server = build_server(retr, warm_up=False)
    calls = sample("rag_mcp_tool_calls_total", tool="list_documents", status="ok")
    timed = sample("rag_mcp_tool_duration_seconds_count", tool="list_documents")

    asyncio.run(server.call_tool("list_documents", {"limit": 5}))

    assert sample("rag_mcp_tool_calls_total", tool="list_documents", status="ok") == (
        calls + 1
    )
    assert sample("rag_mcp_tool_duration_seconds_count", tool="list_documents") == (
        timed + 1
    )


def test_failed_tool_call_is_counted_as_error(retriever):
    retr, _ = retriever
    server = build_server(retr, warm_up=False)
    errors = sample("rag_mcp_tool_calls_total", tool="search_papers", status="error")

    with pytest.raises(ToolError):
        asyncio.run(server.call_tool("search_papers", {"query": "q", "top_k": 99}))

    assert sample("rag_mcp_tool_calls_total", tool="search_papers", status="error") == (
        errors + 1
    )


def test_instrumentation_keeps_the_tool_schema(retriever):
    retr, _ = retriever
    server = build_server(retr, warm_up=False)

    tools = {t.name: t for t in asyncio.run(server.list_tools())}

    assert set(tools["search_papers"].input_schema["properties"]) == {
        "query", "top_k", "mode", "doc_keys", "max_chars",
    }  # fmt: skip


def test_mcp_command_has_a_metrics_port_option():
    from source.rag.commands import build_parser_for_tests

    args = build_parser_for_tests().parse_args(["mcp", "--metrics-port", "9101"])

    assert args.metrics_port == 9101
