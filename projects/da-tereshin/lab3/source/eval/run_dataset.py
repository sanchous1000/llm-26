"""Прогоняет датасет через своего агента или smolagents и сохраняет результаты реплик."""

import argparse
import json
import os
import shutil
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid4, uuid5

from dotenv import load_dotenv


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def custom_turns(project, settings, session, turns):
    from ..run_agent import saved_state, traced_run

    database = project / settings["paths"]["checkpoints"]
    previous_length = 0
    for turn in turns:
        args = SimpleNamespace(session=session, message=turn["input"], resume=False)
        started = perf_counter()
        code = traced_run(args, project, settings)
        elapsed = perf_counter() - started
        state = saved_state(database, session)
        history = state.get("history", [])
        current = state.get("trace_id") == args.trace_id
        messages = history[previous_length:] if current else []
        tool_results = [
            {
                "name": message.name,
                "tool_call_id": message.tool_call_id,
                "status": getattr(message, "status", None),
                "content": message.content,
            }
            for message in messages
            if message.type == "tool"
        ]
        completed = (
            code == 0
            and current
            and bool(messages)
            and messages[-1].type == "ai"
            and not getattr(messages[-1], "tool_calls", [])
            and not state.get("context_error")
        )
        yield {
            "status": "ok" if completed else "timeout" if code == 2 else "error",
            "exit_code": code,
            "trace_id": args.trace_id,
            "answer": (
                messages[-1].content if messages and messages[-1].type == "ai" else None
            ),
            "elapsed_seconds": elapsed,
            "iterations": state.get("iterations") if current else None,
            "summary_calls": state.get("summary_calls") if current else None,
            "context_stats": state.get("context_stats") if current else None,
            "tool_results": tool_results,
        }
        if not completed:
            return
        previous_length = len(history)


@contextmanager
def smol_runtime(settings):
    from langfuse import Langfuse
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from opentelemetry.sdk.trace import TracerProvider

    from ..tracing import TraceRecorder

    provider = TracerProvider()
    enabled = (
        settings["tracing"]["enabled"]
        and os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() != "false"
    )
    client = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
        tracer_provider=provider,
        tracing_enabled=enabled,
        timeout=5,
    )
    trace = TraceRecorder(client)
    instrumentor = SmolagentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    try:
        yield trace
    finally:
        instrumentor.uninstrument()
        trace.close()


def smol_turns(project, settings, session, turns, trace):
    from langfuse import propagate_attributes

    from ..baseline.run_smolagents import build_agent
    from ..tools.study_tools import StudyTools

    client = trace.client
    paths = settings["paths"]
    output = project / paths["plans"] / "smolagents" / uuid5(NAMESPACE_URL, session).hex
    tools = StudyTools(
        project / paths["docs"], project / paths["vector_store"], output, trace=trace
    )
    agent = build_agent(project, settings, tools)
    try:
        for index, turn in enumerate(turns):
            trace_id = uuid4().hex
            started = perf_counter()
            answer = None
            error = None
            limit_reached = False
            try:
                with trace.scope(
                    "smolagents-turn",
                    trace_context={"trace_id": trace_id},
                    input={"message": turn["input"]},
                    metadata={
                        "config": settings,
                        "agent": "ToolCallingAgent",
                        "task_id": turn["id"].split("-")[0],
                        "turn_id": turn["id"],
                        "tracing_version": "shared-runtime-v2",
                    },
                ) as event:
                    with propagate_attributes(
                        session_id=session,
                        user_id=settings["tracing"]["user_id"],
                        tags=[
                            "lab3",
                            "ready-agent",
                            "dataset-eval",
                            "shared-runtime-v2",
                            settings["model"]["name"],
                        ],
                    ):
                        memory_start = len(agent.memory.steps)
                        answer = str(agent.run(turn["input"], reset=index == 0))
                        limit_reached = any(
                            type(getattr(step, "error", None)).__name__
                            == "AgentMaxStepsError"
                            for step in agent.memory.steps[memory_start:]
                        )
                        if limit_reached:
                            event.update(
                                level="WARNING", status_message="iteration_limit"
                            )
                        event["output"] = {"answer": answer}
            except Exception as exception:
                error = {"type": type(exception).__name__, "message": str(exception)}
            client.flush()
            yield {
                "status": "error" if error else "ok",
                "trace_id": trace_id,
                "answer": answer,
                "error": error,
                "iteration_limit": limit_reached,
                "elapsed_seconds": perf_counter() - started,
            }
            if error:
                return
    finally:
        client.flush()


def run_dataset(args, project, settings, trace=None):
    dataset = json.loads(
        (project / settings["paths"]["docs"] / "evaluation/tasks.json").read_text(
            encoding="utf-8"
        )
    )
    selected = set(args.task) if args.task else {task["id"] for task in dataset}
    unknown = selected - {task["id"] for task in dataset}
    if unknown:
        raise SystemExit("Неизвестные задачи: " + ", ".join(sorted(unknown)))
    tasks = [task for task in dataset if task["id"] in selected]
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + args.agent
        + "-"
        + uuid4().hex[:8]
    )
    directory = project / "artifacts/eval" / run_id
    (directory / "logs").mkdir(parents=True)
    write_json(directory / "config.json", settings)
    write_json(
        directory / "inputs.json",
        [
            {
                "id": task["id"],
                "turns": [
                    {"id": turn["id"], "input": turn["input"]} for turn in task["turns"]
                ],
            }
            for task in tasks
        ],
    )
    run = custom_turns if args.agent == "custom" else partial(smol_turns, trace=trace)
    failures = 0
    for task in tasks:
        session = run_id + "-" + task["id"]
        plan_directory = project / settings["paths"]["plans"]
        if args.agent == "smolagents":
            plan_directory /= "smolagents"
        plan_directory /= uuid5(NAMESPACE_URL, session).hex
        stream = run(project, settings, session, task["turns"])
        try:
            for turn in task["turns"]:
                log_path = directory / "logs" / (turn["id"] + ".log")
                started = perf_counter()
                with log_path.open("w", encoding="utf-8") as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        try:
                            result = next(stream)
                        except Exception as error:
                            import traceback

                            traceback.print_exc()
                            result = {
                                "status": "error",
                                "trace_id": None,
                                "answer": None,
                                "elapsed_seconds": perf_counter() - started,
                                "error": {
                                    "type": type(error).__name__,
                                    "message": str(error),
                                },
                            }
                copies = directory / "plans" / turn["id"]
                plans = []
                for path in sorted(plan_directory.glob("*.json")):
                    copies.mkdir(parents=True, exist_ok=True)
                    target = copies / path.name
                    shutil.copy2(path, target)
                    plans.append(str(target.relative_to(directory)))
                record = {
                    "run_id": run_id,
                    "agent": args.agent,
                    "task_id": task["id"],
                    "turn_id": turn["id"],
                    "session_id": session,
                    "input": turn["input"],
                    "log": str(log_path.relative_to(directory)),
                    "plans_so_far": plans,
                    **result,
                }
                with (directory / "results.jsonl").open(
                    "a", encoding="utf-8"
                ) as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(turn["id"], result["status"], flush=True)
                if result["status"] != "ok":
                    failures += 1
                    break
        finally:
            stream.close()
    print("Результаты:", directory)
    raise SystemExit(1 if failures else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["custom", "smolagents"], required=True)
    parser.add_argument(
        "--task",
        action="append",
        help="ID задачи; можно повторить. Без флага — все задачи.",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    load_dotenv(project / ".env")
    settings = json.loads((project / "config.json").read_text(encoding="utf-8"))
    if args.agent == "smolagents":
        with smol_runtime(settings) as trace:
            run_dataset(args, project, settings, trace)
    else:
        run_dataset(args, project, settings)


if __name__ == "__main__":
    main()
