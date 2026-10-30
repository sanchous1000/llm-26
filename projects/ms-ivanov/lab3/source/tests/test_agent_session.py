"""Offline tests of the dialogue agent: scripted model, fake MCP backend, recorded tracing."""

import asyncio
from contextlib import contextmanager

import pytest
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from source.agent.config import load_agent_config, load_prompts
from source.agent.own.agent import AgentFactory
from source.agent.own.mcp_client import McpToolError
from source.agent.own.tools import evaluate
from source.agent.tracing import Tracing


class RecordingTracing(Tracing):
    """Keeps (name, type, parent, fields) of every observation."""

    def __init__(self):
        super().__init__(client=None)
        self.records: list[dict] = []
        self._stack: list[dict] = []

    @contextmanager
    def observe(self, name, as_type="span", **fields):
        record = {
            "name": name,
            "type": as_type,
            "parent": self._stack[-1]["name"] if self._stack else None,
            **fields,
        }
        self.records.append(record)
        self._stack.append(record)
        observation = _Recorder(record)
        try:
            yield observation
        finally:
            self._stack.pop()

    def named(self, prefix):
        return [r for r in self.records if r["name"].startswith(prefix)]


class _Recorder:
    def __init__(self, record):
        self.record = record
        self.id = f"obs-{record['name']}"
        self.trace_id = "trace-1"

    def update(self, **fields):
        self.record.update(fields)
        return self


class FakeBackend:
    server_name = "papers-rag"

    def __init__(self, fail_first: int = 0):
        self.calls: list[tuple[str, dict]] = []
        self._fail = fail_first

    async def call(self, tool, arguments):
        self.calls.append((tool, arguments))
        if self._fail:
            self._fail -= 1
            raise McpToolError("top_k must be in 1..20")
        return {
            "query": arguments.get("query"),
            "count": 1,
            "results": [{"rank": 1, "score": 0.9, "chunk_id": "c1", "doc_key": "doc1"}],
        }


class ScriptedFactory(AgentFactory):
    def __init__(self, function, overrides=(), tracing=None):
        config = load_agent_config(overrides=list(overrides))
        super().__init__(config, load_prompts(config), tracing or RecordingTracing())
        self._function = function

    def model(self, session_id):
        return FunctionModel(self._function, model_name="scripted")


def search_then_answer(messages, info: AgentInfo) -> ModelResponse:
    if not info.function_tools:  # the helper agent (summaries, forced answers)
        return ModelResponse(parts=[TextPart("helper text")])
    last = messages[-1].parts
    if any(isinstance(p, ToolReturnPart) for p in last):
        return ModelResponse(parts=[TextPart("final answer [Chazal et al., 2011]")])
    if any(isinstance(p, RetryPromptPart) for p in last):
        return ModelResponse(
            parts=[ToolCallPart("search_papers", {"query": "dtm", "top_k": 5}, "c2")]
        )
    return ModelResponse(
        parts=[ToolCallPart("search_papers", {"query": "dtm", "top_k": 50}, "c1")]
    )


def always_tools(messages, info: AgentInfo) -> ModelResponse:
    if not info.function_tools:
        return ModelResponse(parts=[TextPart("forced answer from gathered material")])
    n = len(messages)
    return ModelResponse(
        parts=[ToolCallPart("calculator", {"expression": "1+1"}, f"k{n}")]
    )


async def slow(messages, info):
    await asyncio.sleep(5)
    return ModelResponse(parts=[TextPart("too late")])


def ask_all(factory, texts, backend=None):
    async def go():
        session = factory.session(backend or FakeBackend(), "test-session")
        return session, [await session.ask(t) for t in texts]

    return asyncio.run(go())


def test_a_turn_logs_the_tree_and_counts_tools():
    tracing = RecordingTracing()
    factory = ScriptedFactory(search_then_answer, tracing=tracing)
    backend = FakeBackend()

    _, (result,) = ask_all(factory, ["what is dtm"], backend)

    assert result.status == "ok"
    assert result.answer.startswith("final answer")
    assert result.state.tools_used == ["search_papers"]
    assert result.state.retrieved_docs == [["doc1"]]
    turn = tracing.named("turn")[0]
    assert turn["type"] == "agent" and turn["level"] == "DEFAULT"
    llm = tracing.named("llm")
    assert [r["parent"] for r in llm] == ["iteration 1", "iteration 2"]
    assert all(r["type"] == "generation" and "usage_details" in r for r in llm)
    tool = tracing.named("tool:search_papers")[0]
    assert tool["metadata"]["mcp_server"] == "papers-rag"
    assert tracing.named("retrieval")[0]["parent"] == "tool:search_papers"


def test_a_failing_tool_call_is_logged_and_the_model_recovers():
    tracing = RecordingTracing()
    factory = ScriptedFactory(search_then_answer, tracing=tracing)

    session, (result,) = ask_all(factory, ["what is dtm"], FakeBackend(fail_first=1))

    assert result.status == "ok"
    assert result.state.tool_errors == 1
    levels = [r.get("level") for r in tracing.named("tool:search_papers")]
    assert levels[0] == "ERROR"
    message = tracing.named("tool:search_papers")[0]["status_message"]
    assert "top_k must be in 1..20" in message
    returned = [  # the model sees the error text as the tool result and can react
        str(p.content)
        for m in session.history
        for p in m.parts
        if isinstance(p, ToolReturnPart | RetryPromptPart)
    ]
    assert any("top_k must be in 1..20" in text for text in returned)


def test_step_limit_forces_an_answer_and_the_dialogue_goes_on():
    tracing = RecordingTracing()
    factory = ScriptedFactory(
        always_tools, ["limits.max_steps=3", "limits.max_tool_calls=10"], tracing
    )

    session, (first, second) = ask_all(factory, ["compute", "and again"])

    assert first.status == "step_limit"
    assert first.answer == "forced answer from gathered material"
    assert tracing.named("turn")[0]["level"] == "WARNING"
    assert second.status == "step_limit"
    assert isinstance(session.history[-1], ModelResponse)


def test_turn_timeout_is_reported_as_a_warning():
    tracing = RecordingTracing()
    factory = ScriptedFactory(slow, ["limits.turn_timeout_s=0.2"], tracing)

    _, (result,) = ask_all(factory, ["hello"])

    assert result.status == "timeout"
    assert tracing.named("turn")[0]["level"] == "WARNING"
    assert "time limit" in result.answer


def test_long_dialogue_is_summarised_under_a_small_limit():
    tracing = RecordingTracing()
    factory = ScriptedFactory(
        search_then_answer,
        ["context.limit_tokens=2500", "context.keep_recent_turns=1"],
        tracing,
    )
    texts = [f"question {i} " + "padding " * 400 for i in range(4)]

    session, results = ask_all(factory, texts)

    assert all(r.status == "ok" for r in results)
    assert any(r.reduced for r in results)
    assert tracing.named("context-reduction")
    assert tracing.named("reduce:summarise_turns")
    assert tracing.named("helper")[0]["parent"] == "context-reduction"
    assert session.context.state(session.history).tokens <= 2500 + 1200


def test_calculator_is_safe_and_exact():
    assert evaluate("450 * 2**16") == 29491200
    assert evaluate("sqrt(0.04)") == pytest.approx(0.2)
    assert evaluate("3 ^ 2") == 9
    for bad in ("__import__('os')", "open('x')", "(1).real", "2**100000"):
        with pytest.raises(ValueError):
            evaluate(bad)


def test_step_limit_after_a_summary_keeps_the_material_of_the_turn():
    """Regression: a summary shrinks the turn count; the gathered material must survive."""
    prompts = []

    def model(messages, info: AgentInfo) -> ModelResponse:
        if not info.function_tools:  # helper: summary or forced answer
            text = str(messages[-1].parts[0].content)
            prompts.append(text)
            return ModelResponse(parts=[TextPart("helper text")])
        n = len(messages)
        return ModelResponse(
            parts=[
                ToolCallPart("search_papers", {"query": f"q{n}", "top_k": 5}, f"k{n}")
            ]
        )

    factory = ScriptedFactory(
        model,
        [
            "limits.max_steps=3",
            "context.limit_tokens=2500",
            "context.keep_recent_turns=0",
        ],
    )
    texts = [
        "first " + "padding " * 500,
        "second " + "padding " * 500,
        "third question",
    ]

    _, results = ask_all(factory, texts)

    last = results[-1]
    assert last.status == "step_limit"
    forced = prompts[-1]
    assert "third question" in forced  # the current question reached the forced answer
    assert (
        "TOOL CALL: search_papers" in forced
    )  # and the material gathered in this turn
