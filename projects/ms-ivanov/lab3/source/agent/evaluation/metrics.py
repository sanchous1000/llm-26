"""Metrics of an evaluation run: per turn, per task and aggregated per run.

Inputs are the judged run records: one record per task with the agent's turn records
(answer, status, tokens, cost, tools, retrieved documents, context) and a ``judge``
verdict per turn. Everything here is pure, so the numbers are reproducible offline.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from source.agent.evaluation.dataset import Task

KS = (5, 10)


@dataclass(frozen=True)
class TurnRow:
    """One evaluated turn, flat (a row of the result tables)."""

    run: str
    agent: str
    config: str
    task_id: str
    task_type: str
    difficulty: str
    category: str
    turn: int
    depends_on_history: bool
    memory_probe: bool
    after_reduction: bool
    correct: bool
    score: float
    status: str
    cost_usd: float
    latency_s: float
    input_tokens: int
    output_tokens: int
    steps: int
    tool_calls: int
    context_tokens_max: int | None
    tool_recall: float | None
    retrieval: dict[str, float] | None


def ranked_docs(retrieved: Iterable[Sequence[str | None]]) -> list[str]:
    """Documents in the order the agent first saw them (duplicates dropped)."""
    seen: dict[str, None] = {}
    for batch in retrieved:
        for doc in batch:
            if doc and doc not in seen:
                seen[doc] = None
    return list(seen)


def retrieval_scores(
    ranked: Sequence[str], reference: Iterable[str]
) -> dict[str, float]:
    refs = set(reference)
    scores = {}
    for k in KS:
        hits = len(refs & set(ranked[:k]))
        scores[f"recall@{k}"] = hits / len(refs)
        scores[f"precision@{k}"] = hits / k
    first = next((i for i, doc in enumerate(ranked, 1) if doc in refs), None)
    scores["mrr"] = 1 / first if first else 0.0
    return scores


def tool_recall(expected: Iterable[str], used: Iterable[str]) -> float | None:
    """Share of the expected tools the agent called (None if nothing is expected)."""
    wanted = set(expected)
    if not wanted:
        return None
    return len(wanted & set(used)) / len(wanted)


def turn_rows(task: Task, record: dict[str, Any]) -> list[TurnRow]:
    rows, reduced = [], False
    for spec, turn in zip(task.turns, record["turns"], strict=False):
        reduced = reduced or bool(turn.get("reduced"))
        judge = turn.get("judge") or {}
        refs = spec.reference_docs
        rows.append(
            TurnRow(
                run=record["run"],
                agent=record["agent"],
                config=record["config"],
                task_id=task.id,
                task_type=task.type,
                difficulty=task.difficulty,
                category=task.category,
                turn=int(turn["turn"]),
                depends_on_history=spec.depends_on_history,
                memory_probe=spec.memory_probe,
                after_reduction=reduced,
                correct=bool(judge.get("correct")),
                score=float(judge.get("score", 0.0)),
                status=turn.get("status", "error"),
                cost_usd=float(turn.get("cost_usd") or 0.0),
                latency_s=float(turn.get("latency_s") or 0.0),
                input_tokens=int(turn.get("input_tokens") or 0),
                output_tokens=int(turn.get("output_tokens") or 0),
                steps=int(turn.get("steps") or 0),
                tool_calls=int(turn.get("tool_calls") or 0),
                context_tokens_max=turn.get("context_tokens_max"),
                tool_recall=tool_recall(spec.expected_tools, _tool_names(turn)),
                retrieval=(
                    retrieval_scores(
                        ranked_docs(turn.get("retrieved_docs") or []), refs
                    )
                    if refs
                    else None
                ),
            )
        )
    return rows


def _tool_names(turn: dict[str, Any]) -> list[str]:
    """Plain tool names (OpenCode records ``server.tool`` for MCP calls)."""
    return [name.rpartition(".")[2] for name in turn.get("tools_used") or []]


def task_success(rows: Sequence[TurnRow]) -> bool:
    return bool(rows) and all(r.correct for r in rows)


def summarise(rows: Sequence[TurnRow]) -> dict[str, Any]:
    """Aggregates of a set of turns (a run, or a slice of it)."""
    tasks = _by_task(rows)
    single = [t for t in tasks.values() if t[0].task_type == "single"]
    dialogs = [t for t in tasks.values() if t[0].task_type == "dialog"]
    history = [r for r in rows if r.depends_on_history]
    retrieval = [r.retrieval for r in rows if r.retrieval]
    tools = [r.tool_recall for r in rows if r.tool_recall is not None]
    contexts = [r.context_tokens_max for r in rows if r.context_tokens_max]
    return {
        "tasks": len(tasks),
        "turns": len(rows),
        "success_rate": _mean(task_success(t) for t in tasks.values()),
        "success_single": _mean(task_success(t) for t in single),
        "success_dialog": _mean(task_success(t) for t in dialogs),
        "turn_accuracy": _mean(r.correct for r in rows),
        "turn_score": _mean(r.score for r in rows),
        "history_accuracy": _mean(r.correct for r in history),
        "history_accuracy_before_reduction": _mean(
            r.correct for r in history if not r.after_reduction
        ),
        "history_accuracy_after_reduction": _mean(
            r.correct for r in history if r.after_reduction
        ),
        "history_turns_after_reduction": sum(r.after_reduction for r in history),
        "memory_probe_accuracy": _mean(r.correct for r in rows if r.memory_probe),
        "cost_per_task_usd": _mean(sum(r.cost_usd for r in t) for t in tasks.values()),
        "cost_total_usd": sum(r.cost_usd for r in rows),
        "latency_per_turn_s": _mean(r.latency_s for r in rows),
        "latency_per_task_s": _mean(
            sum(r.latency_s for r in t) for t in tasks.values()
        ),
        "tokens_per_task": _mean(
            sum(r.input_tokens + r.output_tokens for r in t) for t in tasks.values()
        ),
        "error_rate": _mean(r.status == "error" for r in rows),
        "step_limit_rate": _mean(r.status == "step_limit" for r in rows),
        "timeout_rate": _mean(r.status == "timeout" for r in rows),
        "failure_rate": _mean(r.status != "ok" for r in rows),
        "steps_per_turn": _mean(r.steps for r in rows),
        "tool_calls_per_turn": _mean(r.tool_calls for r in rows),
        "context_tokens_max": max(contexts) if contexts else None,
        "tool_recall": _mean(tools),
        "tool_correctness": _mean(t == 1.0 for t in tools),
        **{
            key: _mean(r[key] for r in retrieval)
            for key in ("recall@5", "precision@5", "recall@10", "precision@10", "mrr")
        },
    }


def summarise_by(
    rows: Sequence[TurnRow], key: Callable[[TurnRow], str]
) -> dict[str, dict]:
    groups: dict[str, list[TurnRow]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name: summarise(group) for name, group in sorted(groups.items())}


def _by_task(rows: Iterable[TurnRow]) -> dict[str, list[TurnRow]]:
    tasks: dict[str, list[TurnRow]] = defaultdict(list)
    for row in rows:
        tasks[row.task_id].append(row)
    return tasks


def _mean(values: Iterable[float | bool]) -> float | None:
    items = [float(v) for v in values]
    return sum(items) / len(items) if items else None
