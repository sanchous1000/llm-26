"""Проверяет живые трейсы графа, общий таймаут и resume на синтетических моделях без LLM."""

import argparse
import json
import tempfile
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .. import run_agent
from ..tracing import make_recorder
from .check_tracing import test_agent


def worker(args):
    project = Path(args.project)
    settings = json.loads((project / "config.json").read_text())
    trace = make_recorder(project, settings)
    try:
        with trace.scope("agent-attempt", trace_context=args.trace_context):
            with trace.attributes(args.session, settings):
                agent = test_agent(trace, sleep=30 if args.message == "timeout" and not args.resume else None)
                config = {"configurable": {"thread_id": args.session}}
                with SqliteSaver.from_conn_string(str(project / "state.sqlite")) as saver:
                    data = None if args.resume else {
                        "history": [HumanMessage(content=args.message)], "iterations": 0,
                        "trace_id": args.trace_id,
                    }
                    agent.build_graph(saver).invoke(data, config, durability="sync")
    finally:
        trace.close()


def read_observations(client, trace_id, predicate):
    deadline = monotonic() + 30
    while monotonic() < deadline:
        rows = client.api.observations.get_many(
            trace_id=trace_id, fields="basic,io,model,usage,trace_context", limit=100,
            request_options={"timeout_in_seconds": 5, "max_retries": 0},
        ).data
        if predicate(rows):
            return rows
        sleep(1)
    raise AssertionError("Ожидаемые события трейса не появились за 30 секунд")


def main():
    actual = Path(__file__).resolve().parents[2]
    load_dotenv(actual / ".env")
    settings = json.loads((actual / "config.json").read_text())
    settings["tracing"] = {"enabled": True, "user_id": "da-tereshin",
                           "version": "synthetic-tracing-test"}
    settings["paths"]["checkpoints"] = "state.sqlite"
    session = "lab3-tracing-test-" + uuid4().hex[:12]
    reader = make_recorder(actual, settings)
    try:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "config.json").write_text(json.dumps(settings))
            normal = argparse.Namespace(session=session, message="synthetic", resume=False, project=directory)
            with patch.object(run_agent, "run_turn", worker):
                assert run_agent.traced_run(normal, project, settings) == 0
                rows = read_observations(reader.client, normal.trace_id,
                                         lambda items: len([x for x in items if x.type == "GENERATION"]) == 2)
                names = {row.name for row in rows}
                assert {"user-turn", "agent-attempt", "iteration-1", "iteration-2", "retrieval"} <= names
                assert all(row.session_id == session for row in rows)
                generations = [row for row in rows if row.type == "GENERATION"]
                assert all(row.model == "synthetic-model" for row in generations)
                print("Normal trace:", reader.client.get_trace_url(trace_id=normal.trace_id))

                settings["agent"]["turn_timeout_seconds"] = 4
                (project / "config.json").write_text(json.dumps(settings))
                stalled = argparse.Namespace(session=session, message="timeout", resume=False, project=directory)
                assert run_agent.traced_run(stalled, project, settings) == 2
                rows = read_observations(reader.client, stalled.trace_id,
                                         lambda items: any(x.status_message == "turn_timeout" for x in items))
                assert any(row.level == "WARNING" and row.status_message == "turn_timeout" for row in rows)
                config = {"configurable": {"thread_id": session}}
                with SqliteSaver.from_conn_string(str(project / "state.sqlite")) as saver:
                    snapshot = test_agent(reader).build_graph(saver).get_state(config)
                    assert snapshot.next == ("model",)

                settings["agent"]["turn_timeout_seconds"] = 15
                (project / "config.json").write_text(json.dumps(settings))
                resumed = argparse.Namespace(session=session, message=None, resume=True, project=directory)
                assert run_agent.traced_run(resumed, project, settings) == 0
                assert resumed.trace_id == stalled.trace_id
                rows = read_observations(reader.client, resumed.trace_id,
                                         lambda items: len([x for x in items if x.type == "GENERATION"]) == 2)
                roots = [row for row in rows if row.name == "user-turn"]
                assert len(roots) == 2, "Ожидались две попытки одной реплики в одном trace_id"
                assert any(json.loads(row.output or "{}").get("status") == "ok" for row in roots)
                print("Timeout + resume trace:", reader.client.get_trace_url(trace_id=resumed.trace_id))
        print("OK: дерево графа, generation, retrieval, сессия, общий таймаут и resume проверены без LLM.")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
