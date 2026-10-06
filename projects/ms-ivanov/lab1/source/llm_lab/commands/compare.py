"""Команда compare: сравнение всех моделей → results/comparison.md."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_lab.reporting.comparison import write_comparison
from llm_lab.storage import default_results_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сводный отчёт по всем прогнанным моделям")
    parser.add_argument("--results", type=Path, default=default_results_root(), help="директория с результатами")
    args = parser.parse_args(argv)

    try:
        output, models = write_comparison(args.results)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1
    print(f"Сводный отчёт: {output}")
    print(f"Моделей: {len(models)} — {', '.join(models)}")
    return 0
