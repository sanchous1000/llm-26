"""OpenCode runner: generated config, event parsing, status; proxy tool extraction."""

import json

import pytest

from source.agent.config import load_agent_config, load_prompts
from source.agent.opencode.proxy import (
    ProxyLogger,
    RequestMeta,
    _calls_of,
    _generation_name,
)
from source.agent.opencode.proxy_stream import (
    ToolExchange,
    code_tool_calls,
    retrieved_chunks,
)
from source.agent.opencode.runner import (
    RunOutcome,
    _cost,
    classify,
    opencode_config,
    parse_events,
)
from source.tests.test_agent_session import RecordingTracing


def opencode_settings():
    from source.agent.commands import DEFAULT_AGENT_CONFIG

    path = DEFAULT_AGENT_CONFIG.parent / "opencode.yaml"
    config = load_agent_config(path)
    return config, load_prompts(config, path)


SEARCH_OUTPUT = json.dumps(
    {
        "calcRes": {"expression": "450*2**16", "value": 29491200},
        "searchRes": {
            "query": "Weed Bach",
            "results": [
                {"rank": 1, "score": 1.0, "chunk_id": "c1", "doc_key": "weed2017sharp"},
                {
                    "rank": 2,
                    "score": 0.5,
                    "chunk_id": "c2",
                    "doc_key": "chewi2024statistical",
                },
            ],
        },
    }
)


def events(*items) -> str:
    return "\n".join(json.dumps(item) for item in items)


def tool_event(code, output, calls, status="completed"):
    return {
        "type": "tool_use",
        "part": {
            "tool": "execute",
            "state": {
                "status": status,
                "input": {"code": code},
                "output": output,
                "metadata": {"metadata": {"toolCalls": calls}},
            },
        },
    }


def step_finish(fresh, cached, output, reasoning):
    return {
        "type": "step_finish",
        "part": {
            "reason": "tool-calls",
            "tokens": {
                "input": fresh, "output": output, "reasoning": reasoning,
                "cache": {"read": cached, "write": 0},
            },
        },
    }  # fmt: skip


def test_events_of_a_run_are_folded():
    stdout = events(
        {"type": "step_start", "part": {}},
        tool_event(
            'await Promise.all([tools["calc"].calculator({}), tools["papers-rag"].search_papers({})])',
            SEARCH_OUTPUT,
            [
                {"tool": "calc.calculator", "status": "completed"},
                {"tool": "papers-rag.search_papers", "status": "completed"},
            ],
        ),
        step_finish(fresh=600, cached=2000, output=80, reasoning=20),
        {"type": "step_start", "part": {}},
        {"type": "text", "part": {"messageID": "m", "id": "t", "text": "The answer."}},
        "not json",
    )

    outcome = parse_events(stdout)

    assert outcome.steps == 2
    assert outcome.answer == "The answer."
    assert outcome.tools_used == ["calculator", "search_papers"]
    assert outcome.retrieved_docs == [["weed2017sharp", "chewi2024statistical"]]
    assert (outcome.input_tokens, outcome.cached_tokens, outcome.output_tokens) == (
        2600,
        2000,
        100,
    )
    assert outcome.tool_errors == 0


def test_failed_tool_calls_and_errors_are_counted():
    stdout = events(
        tool_event(
            "tools.calc.calculator({})",
            "boom",
            [{"tool": "calc.calculator", "status": "error"}],
        ),
        {"type": "error", "error": {"message": "provider down"}},
    )

    outcome = parse_events(stdout)

    assert outcome.tool_errors == 1
    assert outcome.errors == ["provider down"]


@pytest.mark.parametrize(
    ("outcome", "returncode", "timed_out", "expected"),
    [
        (RunOutcome(answer="a", steps=3), 0, False, "ok"),
        (RunOutcome(answer="a", steps=12), 0, False, "step_limit"),
        (RunOutcome(answer="", steps=4), 0, False, "step_limit"),
        (RunOutcome(answer="", errors=["x"]), 1, False, "error"),
        (RunOutcome(), None, True, "timeout"),
    ],
)
def test_status_classification(outcome, returncode, timed_out, expected):
    assert classify(outcome, returncode, timed_out, max_steps=12) == expected


def test_cost_uses_the_same_prices_as_our_agent():
    config, _ = opencode_settings()
    outcome = RunOutcome(input_tokens=2297, cached_tokens=2000, output_tokens=39)

    assert _cost(outcome, config) == pytest.approx(
        (297 * 0.15 + 2000 * 0.003 + 39 * 0.6) / 1e6
    )


def test_generated_opencode_config_routes_through_the_proxy_with_our_tools():
    config, prompts = opencode_settings()

    generated = opencode_config(config, prompts, {"x-lab3-session": "s"})

    provider = generated["provider"]["lab3"]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:4100/v1"
    assert "OPENCODE_API_KEY" not in json.dumps(generated)  # the key stays in the proxy
    assert provider["options"]["headers"] == {"x-lab3-session": "s"}
    assert (
        provider["models"][config.model.name]["limit"]["context"]
        == config.context.limit_tokens + config.model.max_output_tokens
    )
    assert generated["compaction"] == {"auto": True, "buffer": 0}
    servers = generated["mcp"]["servers"]
    assert (
        servers["papers-rag"]["headers"]["Authorization"]
        == "Bearer {env:PAPERS_RAG_TOKEN_OPENCODE}"
    )
    assert servers["calc"]["command"][-1] == "source.agent.opencode.calc_mcp"
    agent = generated["agents"]["papers"]
    assert agent["system"] == prompts.system
    assert agent["steps"] == config.limits.max_steps
    denied = {p["action"] for p in agent["permissions"] if p["effect"] == "deny"}
    assert {"shell", "edit", "webfetch", "websearch"} <= denied


def test_code_mode_calls_and_nested_results_are_found():
    code = 'const [a, b] = await Promise.all([tools["papers-rag"].search_papers({}), tools.calc.calculator({})]);'

    assert code_tool_calls(code) == ["papers-rag.search_papers", "calc.calculator"]
    assert [c["doc_key"] for c in retrieved_chunks(SEARCH_OUTPUT)] == [
        "weed2017sharp",
        "chewi2024statistical",
    ]
    logged = 'null  Logs: {"results": [{"rank": 1, "doc_key": "d"}]}'
    assert retrieved_chunks(logged)[0]["doc_key"] == "d"
    assert retrieved_chunks('{"value": 2}') is None


def test_proxy_logs_code_mode_tool_calls_with_retrieval():
    config, _ = opencode_settings()
    tracing = RecordingTracing()
    started = []
    tracing.start = lambda name, as_type="span", **fields: _record(
        started, name, as_type, fields
    )
    meta = RequestMeta.from_headers(
        {"x-lab3-session": "s", "x-lab3-trace": "t"}, ("agent:opencode",)
    )
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "execute", "arguments": json.dumps({"code": 'tools["papers-rag"].search_papers({})'})}}]},
        {"role": "tool", "tool_call_id": "1", "content": SEARCH_OUTPUT},
    ]  # fmt: skip

    ProxyLogger(config, tracing).start_generation(
        meta, {"messages": messages, "model": "m"}
    )

    names = [s["name"] for s in started]
    assert names == ["tool:execute", "retrieval", "llm"]
    tool = started[0]
    assert tool["fields"]["metadata"]["mcp_server"] == "papers-rag"
    assert tool["fields"]["trace_id"] == "t"
    assert meta.tags == ("agent:opencode",)


def _record(store, name, as_type, fields):
    entry = {"name": name, "type": as_type, "fields": fields}
    store.append(entry)

    class Observation:
        def start_observation(self, **child):
            return _record(store, child["name"], child.get("as_type"), child)

        def end(self):
            return None

        def update(self, **_):
            return self

    return Observation()


def test_generation_names_and_plain_tool_names():
    assert (
        _generation_name([{"role": "system", "content": "Generate a title for..."}])
        == "title"
    )
    assert _generation_name([{"role": "user", "content": "q"}]) == "llm"
    compaction = "Summarize only what the user and the assistant said and did."
    assert _generation_name([{"role": "user", "content": compaction}]) == "compaction"
    plain = ToolExchange("1", "papers-rag_list_documents", {}, "{}")
    assert _calls_of(plain) == ["papers-rag.list_documents"]


def test_proxy_sums_usage_per_trace_and_serves_it(monkeypatch):
    from starlette.testclient import TestClient

    from source.agent.opencode.proxy import PendingCall, create_app
    from source.agent.tracing import Tracing

    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    config, _ = opencode_settings()
    tracing = Tracing()
    proxy_logger = ProxyLogger(config, tracing)
    pending = PendingCall(tracing.start("llm", "generation"), "trace-1")
    for _ in range(2):
        proxy_logger.finish(
            pending,
            usage={"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 40}},
        )  # fmt: skip

    totals = proxy_logger.totals["trace-1"]
    assert (totals["llm_calls"], totals["input"], totals["cached"]) == (2, 200, 80)
    assert totals["cost_usd"] == pytest.approx(
        2 * (60 * 0.15 + 40 * 0.003 + 10 * 0.6) / 1e6
    )

    with TestClient(create_app(config, tracing)) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/usage/unknown").json() == {}


def proxy_client(handler, monkeypatch):
    import httpx
    from starlette.testclient import TestClient

    from source.agent.opencode.proxy import create_app

    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    config, _ = opencode_settings()
    tracing = RecordingTracing()
    started = []
    tracing.start = lambda name, as_type="span", **fields: _record(
        started, name, as_type, fields
    )
    app = create_app(config, tracing, transport=httpx.MockTransport(handler))
    return TestClient(app), started


def test_proxy_forwards_a_stream_and_logs_it(monkeypatch):
    import httpx

    seen = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["session"] = request.headers["x-opencode-session"]
        seen["body"] = json.loads(request.content)
        chunks = [
            {"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
        ]
        body = (
            "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    client, started = proxy_client(upstream, monkeypatch)
    with client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "stream": True,
                "messages": [{"role": "user", "content": "q"}],
            },
            headers={"x-lab3-session": "dialog-1", "x-lab3-trace": "trace-9"},
        )
        usage = client.get("/usage/trace-9").json()

    assert "Hi" in response.text
    assert seen["auth"] == "Bearer test-key"
    assert seen["session"] == "dialog-1"
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert [s["name"] for s in started] == ["llm"]
    assert usage["llm_calls"] == 1 and usage["input"] == 5


def test_proxy_passes_upstream_errors_and_plain_responses(monkeypatch):
    import httpx

    replies = iter(
        [
            httpx.Response(429, json={"error": {"message": "rate limited"}}),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}),
        ]
    )  # fmt: skip
    client, _ = proxy_client(lambda request: next(replies), monkeypatch)
    payload = {"model": "m", "messages": [{"role": "user", "content": "q"}]}

    with client:
        limited = client.post("/v1/chat/completions", json=payload)
        plain = client.post("/v1/chat/completions", json=payload)

    assert limited.status_code == 429
    assert plain.json()["choices"][0]["message"]["content"] == "ok"


def test_opencode_session_runs_the_cli_and_parses_its_events(tmp_path, monkeypatch):
    import asyncio
    import os
    import stat

    from source.agent.opencode.runner import OpencodeSession

    script = tmp_path / "bin" / "opencode"
    script.parent.mkdir()
    lines = [
        {"type": "step_start", "part": {}},
        tool_event('tools["papers-rag"].search_papers({})', SEARCH_OUTPUT, [{"tool": "papers-rag.search_papers", "status": "completed"}]),
        step_finish(fresh=100, cached=0, output=10, reasoning=0),
        {"type": "text", "part": {"messageID": "m", "id": "t", "text": "Answer [Weed & Bach, 2019]"}},
    ]  # fmt: skip
    events_file = tmp_path / "events.jsonl"
    events_file.write_text(events(*lines), encoding="utf-8")
    script.write_text(f"#!/bin/sh\ncat {events_file}\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    config, prompts = opencode_settings()
    config = config.model_copy(
        update={"proxy": config.proxy.model_copy(update={"port": 1})}
    )
    tracing = RecordingTracing()
    session = OpencodeSession(
        config, prompts, tracing, session_id="s", workdir=tmp_path / "work"
    )

    record = asyncio.run(session.ask("question"))

    assert record["status"] == "ok"
    assert record["answer"] == "Answer [Weed & Bach, 2019]"
    assert record["tools_used"] == ["search_papers"]
    assert record["retrieved_docs"] == [["weed2017sharp", "chewi2024statistical"]]
    assert record["cost_usd"] > 0  # proxy unreachable -> priced from the events
    written = json.loads(
        (tmp_path / "work" / "opencode.json").read_text(encoding="utf-8")
    )
    assert written["agents"]["papers"]["steps"] == config.limits.max_steps
    assert (tmp_path / "work" / "turn-01.events.jsonl").exists()
    turn = tracing.named("turn")[0]
    assert turn["level"] == "DEFAULT" and "context:opencode-compaction" in session.tags
