"""Команда analyze: сводные таблицы анализа → results/analysis.md, results/analysis.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_lab.reporting.analysis import write_analysis
from llm_lab.storage import default_results_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сводный эмпирический анализ результатов")
    parser.add_argument("--results", type=Path, default=default_results_root())
    args = parser.parse_args(argv)

    try:
        md_path, json_path = write_analysis(args.results)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}")
        return 1
    print(f"Анализ: {md_path}, {json_path}")
    return 0
