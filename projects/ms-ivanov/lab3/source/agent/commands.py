"""CLI of the agent layer: task set, own agent, OpenCode and the evaluation."""

import argparse
import asyncio
import json
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from source import config as project
from source.agent.config import DEFAULT_AGENT_CONFIG
from source.agent.evaluation.dataset import (
    DEFAULT_TASKS_PATH,
    DatasetError,
    load_tasks,
    summarise,
    to_langfuse_item,
    unknown_doc_keys,
)

if TYPE_CHECKING:
    from source.agent.own.agent import TurnResult

logger = logging.getLogger(__name__)
MANIFEST_PATH = project.DEFAULT_OUTPUT_DIR / "manifest.json"


OPENCODE_CONFIG = DEFAULT_AGENT_CONFIG.parent / "opencode.yaml"


def register(commands, add_common: Callable[[argparse.ArgumentParser], None]) -> None:
    """Add the agent commands to the top-level parser."""
    del add_common  # the agent commands take no corpus-location options
    tasks = commands.add_parser(
        "agent-tasks", help="validate and summarise the agent task set"
    )
    tasks.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH)
    tasks.add_argument(
        "--export", type=Path, help="write Langfuse dataset items as JSON lines"
    )
    _register_own_agent(commands)
    _register_opencode(commands)
    _register_evaluation(commands)


def _register_own_agent(commands) -> None:
    ask = commands.add_parser("agent-ask", help="run one dialogue given as arguments")
    ask.add_argument("turns", nargs="+", help="user messages, one per turn")
    chat = commands.add_parser("agent-chat", help="interactive dialogue with the agent")
    dialog = commands.add_parser("agent-dialog", help="run a dialogue of the task set")
    _add_task_arguments(dialog)
    for parser in (ask, chat, dialog):
        _add_config_arguments(parser, DEFAULT_AGENT_CONFIG)
        _add_session_arguments(parser)


def _register_opencode(commands) -> None:
    proxy = commands.add_parser(
        "llm-proxy", help="LLM proxy for OpenCode (logs to Langfuse)"
    )
    ask = commands.add_parser("opencode-ask", help="run OpenCode on a dialogue")
    ask.add_argument("turns", nargs="+", help="user messages, one per turn")
    dialog = commands.add_parser("opencode-dialog", help="run OpenCode on a task")
    _add_task_arguments(dialog)
    for parser in (proxy, ask, dialog):
        _add_config_arguments(parser, OPENCODE_CONFIG)
    for parser in (ask, dialog):
        _add_session_arguments(parser)


def _register_evaluation(commands) -> None:
    upload = commands.add_parser(
        "agent-eval-upload", help="task set -> Langfuse dataset"
    )
    upload.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH)
    run = commands.add_parser("agent-eval-run", help="run an agent on the dataset")
    run.add_argument("--agent", choices=("own", "opencode"), required=True)
    run.add_argument("--run-name", required=True)
    _add_config_arguments(run, None)
    run.add_argument("--only", nargs="*", default=[], help="task id prefixes")
    run.add_argument("--concurrency", type=int, default=2)
    judge = commands.add_parser(
        "agent-eval-judge", help="grade a run, scores -> Langfuse"
    )
    judge.add_argument("--run-name", required=True)
    judge.add_argument("--no-push", action="store_true", help="do not send scores")
    report = commands.add_parser("agent-eval-report", help="tables of judged runs")
    report.add_argument("--runs", nargs="+", required=True)
    report.add_argument(
        "--out", type=Path, default=DEFAULT_TASKS_PATH.parent / "RESULTS.md"
    )


def _add_task_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, help="task id or its prefix, e.g. D07")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH)


def _add_config_arguments(
    parser: argparse.ArgumentParser, default: Path | None
) -> None:
    help_text = None if default else "default: agent.yaml / opencode.yaml"
    parser.add_argument("--config", type=Path, default=default, help=help_text)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")


def _add_session_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", help="Langfuse session id (default: random)")
    parser.add_argument("--json", action="store_true", help="print turn records")


def run_agent_tasks(args: argparse.Namespace) -> int:
    try:
        tasks = load_tasks(args.tasks)
    except DatasetError as error:
        logger.error("invalid task set: %s", error)
        return 1
    print(json.dumps(summarise(tasks), ensure_ascii=False, indent=2))
    if MANIFEST_PATH.exists():
        entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["entries"]
        indexed = {
            e["key"] for e in entries if e.get("conversion", {}).get("status") == "ok"
        }
        if missing := unknown_doc_keys(tasks, indexed):
            logger.error("reference documents missing from the corpus: %s", missing)
            return 1
    if args.export:
        lines = (json.dumps(to_langfuse_item(t), ensure_ascii=False) for t in tasks)
        args.export.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("wrote %d items to %s", len(tasks), args.export)
    return 0


def run_agent_ask(args: argparse.Namespace) -> int:
    return _run_turns(args, args.turns, args.session or _new_id("ask"))


def run_agent_dialog(args: argparse.Namespace) -> int:
    task = _find_task(args)
    if task is None:
        return 1
    turns = [turn.user for turn in task.turns]
    return _run_turns(args, turns, args.session or _new_id(task.id.split("-")[0]))


def _find_task(args: argparse.Namespace):
    matches = [t for t in load_tasks(args.tasks) if t.id.startswith(args.task)]
    if len(matches) != 1:
        logger.error("task %r matches %d tasks", args.task, len(matches))
        return None
    return matches[0]


def run_llm_proxy(args: argparse.Namespace) -> int:
    from source.agent.config import load_agent_config, load_env_file
    from source.agent.opencode.proxy_server import serve
    from source.agent.tracing import connect_langfuse

    load_env_file()
    serve(load_agent_config(args.config, args.set), connect_langfuse())
    return 0


def run_opencode_ask(args: argparse.Namespace) -> int:
    return _run_opencode(args, args.turns, args.session or _new_id("oc-ask"))


def run_opencode_dialog(args: argparse.Namespace) -> int:
    task = _find_task(args)
    if task is None:
        return 1
    turns = [turn.user for turn in task.turns]
    session = args.session or _new_id("oc-" + task.id.split("-")[0])
    return _run_opencode(args, turns, session)


def _run_opencode(args: argparse.Namespace, turns: list[str], session_id: str) -> int:
    from source.agent.config import load_agent_config, load_env_file, load_prompts
    from source.agent.opencode.runner import run_opencode_dialog
    from source.agent.tracing import connect_langfuse

    load_env_file()
    config = load_agent_config(args.config, args.set)
    prompts = load_prompts(config, args.config)
    workdir = Path(tempfile.mkdtemp(prefix=f"opencode-{session_id}-"))
    printer = _json_record if args.json else _print_record
    results = asyncio.run(
        run_opencode_dialog(
            config, prompts, connect_langfuse(), turns,
            session_id=session_id, workdir=workdir, on_turn=printer,
        )
    )  # fmt: skip
    print(f"session: {session_id} (workdir {workdir})")
    return 0 if all(r["status"] == "ok" for r in results) else 2


def _print_record(index: int, text: str, record: dict) -> None:
    print(f"\n[{index + 1}] {text}\n\n{record['answer']}\n")
    r = record
    print(
        f"-- {r['status']} | {r['steps']} steps, {r['tool_calls']} tools "
        f"({', '.join(r['tools_used'])}) | compacted={r['reduced']} | "
        f"{r['latency_s']:.1f}s | ${r['cost_usd']:.4f} | trace {r['trace_id']}"
    )


def _json_record(index: int, text: str, record: dict) -> None:
    print(json.dumps({"turn": index, "user": text, **record}, ensure_ascii=False))


def _run_turns(args: argparse.Namespace, turns: list[str], session_id: str) -> int:
    from source.agent.own.runner import build_factory, run_dialog

    factory = build_factory(args.config, args.set)
    printer = _json_turn if args.json else _print_turn
    results = asyncio.run(
        run_dialog(factory, turns, session_id=session_id, on_turn=printer)
    )
    print(f"session: {session_id}")
    return 0 if all(r.status == "ok" for r in results) else 2


def run_agent_chat(args: argparse.Namespace) -> int:
    from source.agent.own.runner import build_factory

    factory = build_factory(args.config, args.set)
    session_id = args.session or _new_id("chat")
    print(f"session {session_id}; /ctx context size, /reset new session, /exit quit")
    try:
        asyncio.run(_chat_loop(factory, session_id, args.json))
    finally:
        factory.tracing.flush()
    return 0


async def _chat_loop(factory, session_id: str, as_json: bool) -> None:
    from source.agent.own.runner import open_session

    async with open_session(factory, session_id) as session:
        while True:
            try:
                text = (await asyncio.to_thread(input, "you> ")).strip()
            except EOFError:
                return
            if text in {"/exit", "/quit"}:
                return
            if text == "/ctx":
                state = session.context.state(session.history)
                print(
                    f"context: {state.tokens}/{state.limit} tokens, {state.turns} turns"
                )
                continue
            if text == "/reset":
                session.history, session.turns = [], 0
                print("history cleared")
                continue
            if text:
                result = await session.ask(text)
                (_json_turn if as_json else _print_turn)(
                    session.turns - 1, text, result
                )


def _print_turn(index: int, text: str, result: "TurnResult") -> None:
    state = result.state
    reduced = " | context reduced" if result.reduced else ""
    print(f"\n[{index + 1}] {text}\n\n{result.answer}\n")
    print(
        f"-- {result.status} | {state.steps} steps, {state.tool_calls} tools "
        f"({', '.join(state.tools_used)}) | context {result.context_tokens_end}/"
        f"{result.context_limit}{reduced} | {result.latency_s:.1f}s | "
        f"${state.cost_usd:.4f} | trace {result.trace_id}"
    )


def _json_turn(index: int, text: str, result: "TurnResult") -> None:
    print(
        json.dumps(
            {"turn": index, "user": text, **result.to_dict()}, ensure_ascii=False
        )
    )


def _new_id(prefix: str) -> str:
    from source.agent.own.runner import new_session_id

    return new_session_id(prefix)


def run_agent_eval_upload(args: argparse.Namespace) -> int:
    from source.agent.config import load_env_file
    from source.agent.evaluation.run import DATASET_NAME, upload_dataset
    from source.agent.tracing import connect_langfuse

    load_env_file()
    count = upload_dataset(connect_langfuse(), load_tasks(args.tasks))
    print(f"dataset {DATASET_NAME}: {count} items")
    return 0


def run_agent_eval_run(args: argparse.Namespace) -> int:
    from source.agent.evaluation.run import run_experiment

    default = DEFAULT_AGENT_CONFIG if args.agent == "own" else OPENCODE_CONFIG
    path = run_experiment(
        agent=args.agent,
        config_path=args.config or default,
        overrides=args.set,
        run_name=args.run_name,
        only=args.only,
        concurrency=args.concurrency,
    )
    print(f"records: {path}")
    return 0


def run_agent_eval_judge(args: argparse.Namespace) -> int:
    from source.agent.config import load_env_file
    from source.agent.evaluation import run as eval_run
    from source.agent.evaluation.judge import (
        judge_records,
        load_judge,
        openai_judge,
        push_scores,
        verdicts_of,
    )
    from source.agent.tracing import connect_langfuse

    load_env_file()
    tasks = eval_run.tasks_by_id()
    config, template = load_judge()
    ask, usage = openai_judge(config)
    name = args.run_name
    earlier = eval_run.run_path(name, "judged")
    previous = verdicts_of(eval_run.load_judged(name)) if earlier.exists() else {}
    judged = asyncio.run(
        judge_records(
            eval_run.load_records(name), tasks, ask, template,
            concurrency=config.concurrency, previous=previous,
        )
    )  # fmt: skip
    out = eval_run.save_judged(name, judged)
    print(f"judged {len(judged)} tasks -> {out}; judge usage {usage}")
    if not args.no_push:
        experiment_id = eval_run.load_meta(name).get("experiment_id")
        summary = push_scores(connect_langfuse().client, judged, tasks, experiment_id)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def run_agent_eval_report(args: argparse.Namespace) -> int:
    from source.agent.evaluation.report import write_report
    from source.agent.evaluation.run import load_judged, tasks_by_id

    judged = {run: load_judged(run) for run in args.runs}
    write_report(judged, tasks_by_id(), args.out)
    print(f"report: {args.out}")
    return 0


HANDLERS = {
    "agent-tasks": run_agent_tasks,
    "agent-ask": run_agent_ask,
    "agent-chat": run_agent_chat,
    "agent-dialog": run_agent_dialog,
    "llm-proxy": run_llm_proxy,
    "opencode-ask": run_opencode_ask,
    "opencode-dialog": run_opencode_dialog,
    "agent-eval-upload": run_agent_eval_upload,
    "agent-eval-run": run_agent_eval_run,
    "agent-eval-judge": run_agent_eval_judge,
    "agent-eval-report": run_agent_eval_report,
}
