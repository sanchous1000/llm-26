"""Команда recompute-metrics: пересчёт метрик скорости в сохранённых прогонах."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_lab.maintenance.recompute_metrics import recompute_model
from llm_lab.storage import default_results_root, resolve_model_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Пересчёт метрик скорости в сохранённых прогонах")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", help="ID модели (как в main.py run --model)")
    target.add_argument("--all", action="store_true", help="все модели в директории результатов")
    parser.add_argument("--results", type=Path, default=default_results_root())
    parser.add_argument("--dry-run", action="store_true", help="только показать, что изменится")
    args = parser.parse_args(argv)

    targets = resolve_model_paths(args.results, None if args.all else args.model)
    if not targets:
        print(f"[error] В {args.results} нет результатов")
        return 1
    return max(recompute_model(paths, args.dry_run) for paths in targets)
