"""MCP response parsing, message dumps, costs, dialogue runner and CLI of the agent."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from source.agent import commands
from source.agent.config import ModelConfig, load_env_file
from source.agent.own import runner
from source.agent.own.mcp_client import McpBackend, McpToolError
from source.agent.tracing import chat_messages, connect_langfuse, cost_details
from source.cli import main
from source.tests.test_agent_session import (
    FakeBackend,
    RecordingTracing,
    ScriptedFactory,
    search_then_answer,
)


class FakeSession:
    def __init__(self, result=None, error=None):
        self._result, self._error = result, error

    async def call_tool(self, name, arguments, read_timeout_seconds=None):
        if self._error:
            raise self._error
        return self._result

    async def send_ping(self):
        return None


def backend_with(*sessions) -> McpBackend:
    """Each call opens the next session (a session or an exception to raise)."""
    queue = list(sessions)

    @asynccontextmanager
    async def factory():
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        yield item

    return McpBackend(
        "http://x", token=None, name="papers-rag", timeout_s=1, session_factory=factory
    )


def result(text="", structured=None, is_error=False):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        structured_content=structured,
        is_error=is_error,
    )


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr("source.agent.own.mcp_client.CONNECT_BACKOFF_S", 0)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (result(structured={"count": 1}), {"count": 1}),
        (result(text='{"count": 2}'), {"count": 2}),
        (result(text="plain"), {"text": "plain"}),
    ],
)
def test_mcp_results_are_parsed(response, expected):
    assert asyncio.run(backend_with(FakeSession(response)).call("t", {})) == expected


def test_a_transient_transport_failure_is_retried():
    backend = backend_with(
        ConnectionError("tls"), FakeSession(result(text='{"ok": 1}'))
    )

    assert asyncio.run(backend.call("t", {})) == {"ok": 1}


def test_server_side_tool_errors_are_not_retried():
    backend = backend_with(FakeSession(result(text="bad top_k", is_error=True)))

    with pytest.raises(McpToolError, match="bad top_k"):
        asyncio.run(backend.call("t", {}))


def test_a_dead_server_becomes_a_tool_error_with_the_real_cause():
    group = ExceptionGroup("tg", [ConnectionError("certificate verify failed")])
    backend = backend_with(group)

    with pytest.raises(McpToolError, match="certificate verify failed"):
        asyncio.run(backend.call("t", {}))

    async def opened():
        async with backend_with(FakeSession()) as b:
            return b.server_name

    assert asyncio.run(opened()) == "papers-rag"


def test_prompt_dump_has_roles_and_tool_calls():
    messages = [
        ModelRequest(parts=[UserPromptPart("q")], instructions="system text"),
        ModelResponse(parts=[ToolCallPart("calculator", {"expression": "1+1"}, "c")]),
        ModelRequest(parts=[ToolReturnPart("calculator", {"value": 2}, "c")]),
        ModelResponse(parts=[TextPart("2")]),
    ]

    chat = chat_messages(messages)

    assert [m["role"] for m in chat] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert chat[2]["tool_calls"][0]["name"] == "calculator"
    assert json.loads(chat[3]["content"]) == {"value": 2}


def test_cost_uses_list_prices_and_cheaper_cache():
    model = ModelConfig(
        name="m",
        base_url="u",
        price_input_per_mtok=0.15,
        price_output_per_mtok=0.6,
        price_cache_read_per_mtok=0.003,
    )
    usage = RequestUsage(input_tokens=297, output_tokens=39)

    assert cost_details(usage, model)["total"] == pytest.approx(6.795e-05)
    cached = RequestUsage(input_tokens=1000, output_tokens=0, cache_read_tokens=1000)
    assert cost_details(cached, model)["total"] == pytest.approx(3e-6)


def test_tracing_can_be_switched_off():
    assert not connect_langfuse(enabled=False).enabled


def test_env_file_does_not_override_existing_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nA_TEST_KEY=from-file\nB_TEST_KEY=x\n", encoding="utf-8")
    monkeypatch.setenv("A_TEST_KEY", "from-env")
    monkeypatch.delenv("B_TEST_KEY", raising=False)

    load_env_file(env)

    import os

    assert os.environ["A_TEST_KEY"] == "from-env"
    assert os.environ["B_TEST_KEY"] == "x"


def test_run_dialog_keeps_one_session_for_all_turns(monkeypatch):
    @asynccontextmanager
    async def fake_backend(_config):
        yield FakeBackend()

    monkeypatch.setattr(runner, "mcp_backend", fake_backend)
    tracing = RecordingTracing()
    factory = ScriptedFactory(search_then_answer, tracing=tracing)
    seen = []

    results = asyncio.run(
        runner.run_dialog(
            factory, ["a", "b"], session_id="s1", on_turn=lambda i, t, r: seen.append(i)
        )
    )

    assert [r.status for r in results] == ["ok", "ok"]
    assert seen == [0, 1]
    assert len(tracing.named("turn")) == 2
    assert runner.new_session_id("x").startswith("x-")


def test_turn_printers_show_status_and_cost(capsys):
    factory = ScriptedFactory(search_then_answer)

    async def one():
        return await factory.session(FakeBackend(), "s").ask("q")

    turn = asyncio.run(one())
    commands._print_turn(0, "q", turn)
    commands._json_turn(0, "q", turn)

    out = capsys.readouterr().out
    assert "-- ok | 2 steps, 1 tools (search_papers)" in out
    record = json.loads(out.strip().splitlines()[-1])
    assert record["tools_used"] == ["search_papers"] and record["status"] == "ok"


def test_agent_tasks_command_validates_and_exports(tmp_path, capsys):
    target = tmp_path / "tasks.jsonl"

    assert main(["agent-tasks", "--export", str(target)]) == 0

    assert '"tasks": 24' in capsys.readouterr().out
    assert len(target.read_text(encoding="utf-8").splitlines()) == 24


def test_agent_dialog_rejects_an_unknown_task():
    assert main(["agent-dialog", "--task", "ZZZ"]) == 1


def test_parallel_calls_are_limited_to_a_few_at_a_time():
    from source.agent.own import mcp_client

    active, peak = 0, 0

    class SlowSession(FakeSession):
        async def call_tool(self, name, arguments, read_timeout_seconds=None):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return result(text="{}")

    backend = backend_with(SlowSession())

    async def many():
        await asyncio.gather(*(backend.call("t", {}) for _ in range(10)))

    asyncio.run(many())
    assert peak == mcp_client.MAX_CONCURRENT_CALLS
