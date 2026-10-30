"""Stdio MCP server with the agent's calculator, so that OpenCode gets the same tool.

Started by OpenCode itself (``mcp.servers.calc`` in its config):
``python -m source.agent.opencode.calc_mcp``.
"""

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from source.agent.own.tools import evaluate


def build_server() -> MCPServer:
    server = MCPServer(
        "calc", instructions="Exact arithmetic for the research assistant."
    )

    @server.tool()
    async def calculator(expression: str) -> dict[str, Any]:
        """Evaluate an arithmetic expression exactly as written and return the number.

        Supports + - * / // % **, parentheses, sqrt, log (natural), log2, log10, exp,
        ceil, floor, min, max, abs and the constants pi and e.

        Args:
            expression: The expression, e.g. "450 * 2**16" or "3 ** (-1/16)".
        """
        try:
            return {"expression": expression, "value": evaluate(expression)}
        except (
            ValueError,
            SyntaxError,
            ZeroDivisionError,
            OverflowError,
            TypeError,
        ) as e:
            raise ToolError(f"cannot evaluate {expression!r}: {e}") from e

    return server


if __name__ == "__main__":
    build_server().run("stdio")
