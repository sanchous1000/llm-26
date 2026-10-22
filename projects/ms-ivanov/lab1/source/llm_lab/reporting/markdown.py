"""Markdown-примитивы для отчётов: форматирование чисел и таблиц."""

from __future__ import annotations

from typing import Any, Sequence


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def esc(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(esc(cell) for cell in row) + " |" for row in rows]
    lines.append("")
    return lines
