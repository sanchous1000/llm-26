"""Tables of the evaluation: runs side by side, by task type, difficulty, category."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from source.agent.evaluation.dataset import Task
from source.agent.evaluation.metrics import TurnRow, summarise, summarise_by, turn_rows

MAIN = (
    ("success_rate", "Task success", "pct"),
    ("success_single", "— одиночные", "pct"),
    ("success_dialog", "— диалоги", "pct"),
    ("turn_accuracy", "Верных реплик", "pct"),
    ("history_accuracy", "Учёт контекста (реплики с историей)", "pct"),
    ("history_accuracy_before_reduction", "— до сокращения истории", "pct"),
    ("history_accuracy_after_reduction", "— после сокращения истории", "pct"),
    ("history_turns_after_reduction", "— реплик после сокращения", "int"),
    ("memory_probe_accuracy", "Проверки памяти (memory_probe)", "pct"),
    ("cost_per_task_usd", "Стоимость задачи, $", "usd"),
    ("cost_total_usd", "Стоимость прогона, $", "usd"),
    ("tokens_per_task", "Токенов на задачу", "int"),
    ("latency_per_turn_s", "Латентность реплики, с", "num"),
    ("latency_per_task_s", "Латентность задачи, с", "num"),
    ("failure_rate", "Реплик с ошибкой/лимитом/таймаутом", "pct"),
    ("step_limit_rate", "— лимит шагов", "pct"),
    ("error_rate", "— ошибка", "pct"),
    ("steps_per_turn", "Шагов на реплику", "num"),
    ("tool_calls_per_turn", "Вызовов инструментов на реплику", "num"),
    ("context_tokens_max", "Максимальный контекст, токенов", "int"),
    ("tool_correctness", "Tool correctness (все ожидаемые)", "pct"),
    ("tool_recall", "Tool recall", "pct"),
    ("recall@5", "Recall@5 (документы)", "pct"),
    ("precision@5", "Precision@5", "pct"),
    ("recall@10", "Recall@10", "pct"),
    ("mrr", "MRR", "num"),
)
SLICE = (
    ("success_rate", "success", "pct"),
    ("turn_accuracy", "реплики", "pct"),
    ("cost_per_task_usd", "$/задача", "usd"),
    ("latency_per_task_s", "с/задача", "num"),
)


def all_rows(
    judged_runs: dict[str, list[dict]], tasks: dict[str, Task]
) -> dict[str, list[TurnRow]]:
    return {
        run: [
            row
            for record in records
            for row in turn_rows(tasks[record["task_id"]], record)
        ]
        for run, records in judged_runs.items()
    }


def fmt(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{100 * value:.0f} %"
    if kind == "usd":
        return f"{value:.4f}"
    if kind == "int":
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:.2f}"


def main_table(rows_by_run: dict[str, list[TurnRow]]) -> str:
    runs = list(rows_by_run)
    stats = {run: summarise(rows) for run, rows in rows_by_run.items()}
    lines = ["| Метрика | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    for key, label, kind in MAIN:
        lines.append(
            f"| {label} | "
            + " | ".join(fmt(stats[r].get(key), kind) for r in runs)
            + " |"
        )
    return "\n".join(lines)


def slice_table(rows_by_run: dict[str, list[TurnRow]], key, title: str) -> str:
    runs = list(rows_by_run)
    groups = sorted({key(row) for rows in rows_by_run.values() for row in rows})
    header = (
        f"| {title} | "
        + " | ".join(f"{r}: {label}" for r in runs for _, label, _ in SLICE)
        + " |"
    )
    lines = [header, "|---|" + "---|" * (len(runs) * len(SLICE))]
    per_run = {run: summarise_by(rows, key) for run, rows in rows_by_run.items()}
    for group in groups:
        cells = [
            fmt(per_run[run].get(group, {}).get(metric), kind)
            for run in runs
            for metric, _, kind in SLICE
        ]
        lines.append(f"| {group} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def task_table(rows_by_run: dict[str, list[TurnRow]]) -> str:
    """Per task: share of correct turns per run (✓ = whole task solved)."""
    runs = list(rows_by_run)
    by_run = {
        run: summarise_by(rows, lambda r: r.task_id)
        for run, rows in rows_by_run.items()
    }
    task_ids = sorted({row.task_id for rows in rows_by_run.values() for row in rows})
    lines = ["| Задача | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    for task_id in task_ids:
        cells = []
        for run in runs:
            stats = by_run[run].get(task_id)
            if not stats:
                cells.append("—")
                continue
            mark = "✓" if stats["success_rate"] == 1.0 else "✗"
            cells.append(f"{mark} {fmt(stats['turn_accuracy'], 'pct')}")
        lines.append(f"| {task_id} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    judged_runs: dict[str, list[dict]], tasks: dict[str, Task], path: Path,
    extra: Sequence[str] = (),
) -> dict[str, dict]:  # fmt: skip
    rows_by_run = all_rows(judged_runs, tasks)
    parts = [
        "# Результаты оценки агентов",
        "Файл генерируется командой `python -m source agent-eval-report`.",
        "## Сводка по прогонам",
        main_table(rows_by_run),
        "## По типу задачи",
        slice_table(rows_by_run, lambda r: r.task_type, "тип"),
        "## По сложности",
        slice_table(rows_by_run, lambda r: r.difficulty, "сложность"),
        "## По категории",
        slice_table(rows_by_run, lambda r: r.category, "категория"),
        "## По задачам (✓ — решена целиком, % — доля верных реплик)",
        task_table(rows_by_run),
        *extra,
    ]
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    summaries = {run: summarise(rows) for run, rows in rows_by_run.items()}
    path.with_suffix(".json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summaries
