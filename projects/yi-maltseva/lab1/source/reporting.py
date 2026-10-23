from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from config import MODELS


def mean_or_none(values: list[float]) -> float | None:
    cleaned = [value for value in values if isinstance(value, (int, float))]
    return statistics.mean(cleaned) if cleaned else None


def fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    repeat_groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        groups[(row["model"], row["mode"])].append(row)
        repeat_groups[(row["model"], row["prompt_id"], row["mode"])].append(row.get("answer", ""))

    summary_rows = []
    for model in MODELS:
        for mode in ("A_baseline", "B_tuned"):
            bucket = groups.get((model["model"], mode), [])
            summary_rows.append({
                "model": model["model"],
                "family": model["family"],
                "mode": mode,
                "runs": len(bucket),
                "mean_response_time_s": mean_or_none([float(x["response_time_s"]) for x in bucket if x.get("response_time_s") is not None]),
                "mean_input_tokens": mean_or_none([float(x["prompt_tokens"]) for x in bucket if x.get("prompt_tokens") is not None]),
                "mean_output_tokens": mean_or_none([float(x["completion_tokens"]) for x in bucket if x.get("completion_tokens") is not None]),
                "mean_end_to_end_output_tokens_per_sec": mean_or_none([float(x["end_to_end_output_tokens_per_sec"]) for x in bucket if x.get("end_to_end_output_tokens_per_sec") is not None]),
                "mean_prompt_eval_tokens_per_sec": mean_or_none([float(x["prompt_eval_tokens_per_sec"]) for x in bucket if x.get("prompt_eval_tokens_per_sec") is not None]),
                "mean_probe_generation_tokens_per_sec": mean_or_none([float(x["probe_generation_tokens_per_sec"]) for x in bucket if x.get("probe_generation_tokens_per_sec") is not None]),
                "mean_task_score": mean_or_none([float(x["task_score"]) for x in bucket if x.get("task_score") is not None]),
            })

    stable_groups = 0
    eligible_groups = 0
    repeat_details = []
    for (model, prompt_id, mode), answers in sorted(repeat_groups.items()):
        if len(answers) > 1:
            eligible_groups += 1
            exact = len(set(answers)) == 1
            stable_groups += int(exact)
            status = "полное совпадение" if exact else "ответы различаются"
            repeat_details.append(f"- {model} / {prompt_id} / {mode}: {status} ({len(set(answers))} уникальных варианта из {len(answers)} прогонов).")

    lines = [
        "# Автоматическая сводка лабораторной № 1",
        "",
        f"Дата запуска: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Средние показатели по модели и режиму",
        "",
        "| Модель | Режим | Прогонов | Время ответа, с | Входных токенов | Выходных токенов | Выходные токены/с (всё время запроса) | Prompt eval, токенов/с | Decode probe, токенов/с | Средняя оценка задачи |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['mode']} | {row['runs']} | {fmt(row['mean_response_time_s'])} | "
            f"{fmt(row['mean_input_tokens'], 1)} | {fmt(row['mean_output_tokens'], 1)} | "
            f"{fmt(row['mean_end_to_end_output_tokens_per_sec'])} | {fmt(row['mean_prompt_eval_tokens_per_sec'])} | "
            f"{fmt(row['mean_probe_generation_tokens_per_sec'])} | {fmt(row['mean_task_score'], 3)} |"
        )
    lines += [
        "",
        "Примечание: скорость вывода по основному запросу рассчитана как число выходных токенов, делённое на полное время HTTP-запроса; она включает не только декодирование. Скорость prompt eval и decode probe измеряется отдельным коротким диагностическим запросом Ollama с тем же промптом и режимом.",
        "",
        "## Повторяемость ответов",
        "",
        f"Полностью совпали ответы в {stable_groups} из {eligible_groups} групп модель × промпт × режим, где было больше одного повтора. Доля точного совпадения: {fmt(stable_groups / eligible_groups if eligible_groups else None, 3)}.",
        "",
    ]
    lines.extend(repeat_details or ["Повторов для расчёта повторяемости недостаточно."])
    lines += [
        "",
        "## Как читать автоматическую оценку качества",
        "",
        "- P1: доля выполненных формальных проверок письма. Это не полная оценка качества текста.",
        "- P2: доля трёх меток классификации, совпавших с заранее заданной последовательностью.",
        "- P3: доля полей товара, совпавших с ожидаемыми значениями; при невалидном JSON применяется упрощённая проверка текста.",
        "- Фактические ошибки и галлюцинации нельзя надёжно определить только эвристиками. Заполни `manual_review.csv` после чтения ответов.",
        "",
        "## Ручная проверка",
        "",
        "Для каждого ответа оцени адекватность от 1 до 5, укажи фактические ошибки/галлюцинации, повторяющиеся фразы и примечания в `manual_review.csv`.",
    ]
    return summary_rows, "\n".join(lines) + "\n"
