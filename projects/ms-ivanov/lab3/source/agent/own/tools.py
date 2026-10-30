"""The agent's tools: four proxies of the papers-rag MCP server and a calculator.

Descriptions shown to the model come from ``configs/agent/tools.yaml``; the functions
below only define names, types and behaviour.
"""

import ast
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import ModelRetry, RunContext, Tool
from pydantic_ai.tools import ToolDefinition

from source.agent.own.mcp_client import McpToolError, ToolBackend
from source.agent.tracing import Tracing

MCP_TOOLS = (
    "search_papers",
    "get_chunk_context",
    "list_documents",
    "get_document_outline",
)


@dataclass
class TurnState:
    """Counters of one user turn, filled by the runtime hooks (mutable on purpose)."""

    steps: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    tools_used: list[str] = field(default_factory=list)
    retrieved_docs: list[list[str]] = field(default_factory=list)
    context_tokens: list[int] = field(default_factory=list)
    reductions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentDeps:
    backend: ToolBackend
    tracing: Tracing
    state: TurnState = field(default_factory=TurnState)


async def search_papers(  # noqa: PLR0917 - mirrors the MCP tool signature
    ctx: RunContext[AgentDeps],
    query: str,
    top_k: int = 5,
    mode: str = "auto",
    doc_keys: list[str] | None = None,
    max_chars: int = 800,
) -> dict[str, Any]:
    arguments = {"query": query, "top_k": top_k, "mode": mode, "max_chars": max_chars}
    if doc_keys:
        arguments["doc_keys"] = doc_keys
    deps = ctx.deps
    with deps.tracing.observe(
        "retrieval",
        as_type="retriever",
        input=arguments,
        metadata={"source": f"MCP {deps.backend.server_name}: hybrid BGE-M3 index"},
    ) as span:
        result = await _call(deps, "search_papers", arguments)
        hits = result.get("results", [])
        span.update(
            output=[
                {
                    "rank": hit.get("rank"),
                    "score": hit.get("score"),
                    "chunk_id": hit.get("chunk_id"),
                    "doc_key": hit.get("doc_key"),
                    "section": hit.get("section"),
                    "citation": hit.get("citation"),
                }
                for hit in hits
            ],
            metadata={"count": len(hits), "doc_keys": [h.get("doc_key") for h in hits]},
        )
    deps.state.retrieved_docs.append([h.get("doc_key") for h in hits])
    return result


async def get_chunk_context(
    ctx: RunContext[AgentDeps],
    chunk_id: str,
    before: int = 1,
    after: int = 1,
    max_chars: int = 4000,
) -> dict[str, Any]:
    arguments = {
        "chunk_id": chunk_id,
        "before": before,
        "after": after,
        "max_chars": max_chars,
    }
    return await _call(ctx.deps, "get_chunk_context", arguments)


async def list_documents(
    ctx: RunContext[AgentDeps], query: str | None = None, limit: int = 50
) -> dict[str, Any]:
    arguments: dict[str, Any] = {"limit": limit}
    if query:
        arguments["query"] = query
    return await _call(ctx.deps, "list_documents", arguments)


async def get_document_outline(
    ctx: RunContext[AgentDeps], doc_key: str
) -> dict[str, Any]:
    return await _call(ctx.deps, "get_document_outline", {"doc_key": doc_key})


async def _call(
    deps: AgentDeps, tool: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    try:
        return await deps.backend.call(tool, arguments)
    except McpToolError as error:
        deps.state.tool_errors += 1
        raise ModelRetry(
            f"{tool} failed: {error}. Fix the arguments or try another tool."
        ) from error


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: Mapping[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "abs": abs,
    "min": min,
    "max": max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 10_000


def evaluate(expression: str) -> float:
    """Evaluate arithmetic safely: no names, attributes or calls off the whitelist."""
    tree = ast.parse(expression.replace("^", "**"), mode="eval")
    return _eval(tree.body)


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_eval(arg) for arg in node.args))
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")


async def calculator(expression: str) -> dict[str, Any]:
    try:
        value = evaluate(expression)
    except (
        ValueError,
        SyntaxError,
        ZeroDivisionError,
        OverflowError,
        TypeError,
    ) as error:
        raise ModelRetry(f"cannot evaluate {expression!r}: {error}") from error
    return {"expression": expression, "value": value}


def build_tools(
    descriptions: Mapping[str, Any], *, retries: int
) -> list[Tool[AgentDeps]]:
    """Tools with descriptions (and parameter descriptions) taken from the YAML file."""
    functions = [
        search_papers,
        get_chunk_context,
        list_documents,
        get_document_outline,
        calculator,
    ]
    tools = []
    for function in functions:
        spec = descriptions.get(function.__name__)
        if not spec or not spec.get("description"):
            raise ValueError(f"tools.yaml has no description for {function.__name__}")
        tools.append(
            Tool(
                function,
                name=function.__name__,
                description=" ".join(spec["description"].split()),
                max_retries=retries,
                prepare=_describe_parameters(spec.get("parameters") or {}),
                metadata=(
                    {"mcp_server": "papers-rag"}
                    if function.__name__ in MCP_TOOLS
                    else {}
                ),
            )
        )
    return tools


def _describe_parameters(parameters: Mapping[str, str]):
    async def prepare(
        _: RunContext[AgentDeps], tool_def: ToolDefinition
    ) -> ToolDefinition:
        schema = dict(tool_def.parameters_json_schema)
        properties = {
            name: (
                {**prop, "description": parameters[name]}
                if name in parameters
                else prop
            )
            for name, prop in schema.get("properties", {}).items()
        }
        return replace(
            tool_def, parameters_json_schema={**schema, "properties": properties}
        )

    return prepare
