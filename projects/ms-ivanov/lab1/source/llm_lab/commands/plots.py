"""Команда plots: графики для отчёта → results/figures/ (нужен matplotlib)."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_lab.reporting.plots import render_all
from llm_lab.storage import default_results_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Графики для отчёта лабораторной")
    parser.add_argument("--results", type=Path, default=default_results_root())
    args = parser.parse_args(argv)

    try:
        written = render_all(args.results)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}")
        return 1
    for path in written:
        print(f"  {path}")
    return 0
