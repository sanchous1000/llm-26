"""Evaluation runs: the task set as a Langfuse dataset, one experiment per agent/config.

Each dataset item is one task (a dialogue of one or more turns). An experiment runs
every item as one Langfuse trace (``experiment-item-run``); inside it every user turn
is a ``turn`` observation of the agent, with the session, tags and the agent's own
tree. The per-turn records (answers, status, tokens, cost, tools, documents,
context) are written to ``eval/agent/runs/<run>.jsonl`` for judging and metrics.

Files of a run in ``RUNS_DIR``: ``<run>.jsonl`` (raw records, not kept in git),
``<run>.judged.jsonl`` (records + verdicts) and ``<run>.meta.json`` (experiment id).
"""

import json
import logging
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

from source import config as project
from source.agent.config import (
    AgentConfig,
    Prompts,
    load_agent_config,
    load_env_file,
    load_prompts,
)
from source.agent.evaluation.dataset import Task, load_tasks, to_langfuse_item
from source.agent.opencode.proxy_server import proxy_in_thread
from source.agent.opencode.runner import OpencodeSession
from source.agent.own.agent import AgentFactory
from source.agent.own.runner import open_session
from source.agent.tracing import Tracing, connect_langfuse

logger = logging.getLogger(__name__)
DATASET_NAME = "papers-agent-tasks"
RUNS_DIR = project.DEFAULT_EVAL_DIR / "agent" / "runs"


def upload_dataset(tracing: Tracing, tasks: Sequence[Task]) -> int:
    """Create (or update) the dataset; item ids are the task ids."""
    client = _client(tracing)
    client.create_dataset(
        name=DATASET_NAME,
        description="Research-assistant tasks over the papers-rag corpus (lab 3)",
        metadata={"tasks": len(tasks), "source": "eval/agent/tasks.yaml"},
    )
    for task in tasks:
        item = to_langfuse_item(task)
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=item["id"],
            input=item["input"],
            expected_output=item["expected_output"],
            metadata={"task_id": item["id"], **item["metadata"]},
        )
    client.flush()
    return len(tasks)


def run_path(run_name: str, kind: str = "") -> Path:
    """``RUNS_DIR/<run>.jsonl``, or ``<run>.<kind>.json(l)`` for judged/meta."""
    suffix = {"": ".jsonl", "judged": ".judged.jsonl", "meta": ".meta.json"}[kind]
    return RUNS_DIR / f"{run_name}{suffix}"


def run_experiment(
    *,
    agent: str,
    config_path: Path,
    overrides: Sequence[str],
    run_name: str,
    only: Sequence[str] = (),
    concurrency: int = 2,
) -> Path:
    """Run every (or the selected) dataset item; returns the records file."""
    load_env_file()
    config = load_agent_config(config_path, list(overrides))
    prompts = load_prompts(config, config_path)
    tracing = connect_langfuse()
    client = _client(tracing)
    items = [
        item
        for item in client.get_dataset(DATASET_NAME).items
        if not only or any(item.id.startswith(prefix) for prefix in only)
    ]
    out = run_path(run_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    if agent == "own":
        dialog = partial(_own_dialog, AgentFactory(config, prompts, tracing))
    else:
        dialog = partial(_opencode_dialog, config, prompts, tracing)
    task = _item_task(
        agent=agent, config_name=config.name, run_name=run_name, tracing=tracing,
        dialog=dialog, writer=_RecordWriter(out),
    )  # fmt: skip
    proxy = proxy_in_thread(config, tracing) if agent == "opencode" else nullcontext()
    with proxy:
        result = client.run_experiment(
            name=DATASET_NAME,
            run_name=run_name,
            description=f"{agent} / {config.name} / {config.model.name}",
            data=items,
            task=task,
            max_concurrency=concurrency,
            metadata={
                "agent": agent,
                "config": config.name,
                "model": config.model.name,
            },
        )
    experiment_id = getattr(result, "experiment_id", None) or getattr(
        result, "dataset_run_id", None
    )
    meta = {"run": run_name, "experiment_id": experiment_id}
    run_path(run_name, "meta").write_text(json.dumps(meta, indent=2), "utf-8")
    tracing.flush()
    return out


Dialog = Callable[[list[str], str], Awaitable[list[dict]]]


def _item_task(
    *,
    agent: str,
    config_name: str,
    run_name: str,
    tracing: Tracing,
    dialog: Dialog,
    writer: "_RecordWriter",
):
    """The Langfuse experiment task: one dataset item = one dialogue."""

    async def task(*, item, **_: Any) -> dict[str, Any]:
        session_id = f"{run_name}--{item.id}"
        started = time.monotonic()
        item_trace = tracing.current_trace_id()
        records = await dialog(list(item.input["turns"]), session_id)
        writer.write(
            {
                "run": run_name,
                "agent": agent,
                "config": config_name,
                "task_id": item.id,
                "session_id": session_id,
                "item_trace_id": item_trace,
                "latency_s": round(time.monotonic() - started, 2),
                "turns": records,
            }
        )
        return {
            "answers": [r["answer"] for r in records],
            "statuses": [r["status"] for r in records],
        }

    return task


async def _own_dialog(
    factory: AgentFactory, turns: list[str], session_id: str
) -> list[dict]:
    records = []
    async with open_session(factory, session_id) as session:
        for index, text in enumerate(turns):
            result = await session.ask(text)
            records.append({"turn": index, "user": text, **result.to_dict()})
    return records


async def _opencode_dialog(
    config: AgentConfig,
    prompts: Prompts,
    tracing: Tracing,
    turns: list[str],
    session_id: str,
) -> list[dict]:
    workdir = Path(tempfile.mkdtemp(prefix=f"opencode-{session_id[:60]}-"))
    session = OpencodeSession(
        config, prompts, tracing, session_id=session_id, workdir=workdir
    )
    records = []
    for index, text in enumerate(turns):
        record = await session.ask(text)
        records.append({"turn": index, "user": text, **record, "workdir": str(workdir)})
    return records


class _RecordWriter:
    """Appends one JSON line per finished task (safe for concurrent tasks)."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _client(tracing: Tracing):
    if tracing.client is None:
        raise RuntimeError("Langfuse is not configured (LANGFUSE_* in .env)")
    return tracing.client


def load_records(run_name: str) -> list[dict[str, Any]]:
    """Records of a run; a task run twice keeps its last record.

    Without the raw file the judged one is read: it holds the same records.
    """
    path = run_path(run_name)
    if not path.exists():
        path = run_path(run_name, "judged")
    records = {record["task_id"]: record for record in _read_jsonl(path)}
    return list(records.values())


def load_judged(run_name: str) -> list[dict[str, Any]]:
    return _read_jsonl(run_path(run_name, "judged"))


def save_judged(run_name: str, records: Sequence[dict[str, Any]]) -> Path:
    path = run_path(run_name, "judged")
    lines = (json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text("".join(lines), encoding="utf-8")
    return path


def load_meta(run_name: str) -> dict[str, Any]:
    path = run_path(run_name, "meta")
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def tasks_by_id() -> dict[str, Task]:
    return {task.id: task for task in load_tasks()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
