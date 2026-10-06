"""Единая точка входа лабораторной №1.

    python source/main.py <команда> [аргументы]
    python source/main.py <команда> --help

Каждая команда — модуль llm_lab/commands/ со своей функцией main(argv);
логика лежит в остальных пакетах по назначению.
"""

from __future__ import annotations

import importlib
import sys
import unittest

from llm_lab.paths import SOURCE_DIR, TESTS_DIR

# Команда → (модуль, описание). Порядок — порядок работы с лабораторной.
COMMANDS: dict[str, tuple[str, str]] = {
    "run": ("llm_lab.commands.run", "прогон тестов одной модели (также --list-models, --report-only, --self-test)"),
    "compare": ("llm_lab.commands.compare", "сравнение моделей → results/comparison.md"),
    "analyze": ("llm_lab.commands.analyze", "сводные таблицы анализа → results/analysis.md, analysis.json"),
    "plots": ("llm_lab.commands.plots", "графики для отчёта → results/figures/ (нужен matplotlib)"),
    "merge": ("llm_lab.commands.merge", "склеить прогоны модели из нескольких сессий"),
    "recompute-metrics": ("llm_lab.commands.recompute_metrics", "пересчитать метрики скорости в сохранённых прогонах"),
    "rescore": ("llm_lab.commands.rescore", "пересчитать автопроверки по сохранённым ответам"),
}
TEST_COMMAND = ("test", "запустить тесты (-v — подробно); сеть и модель не нужны")


def usage() -> str:
    width = max(len(name) for name in (*COMMANDS, TEST_COMMAND[0]))
    rows = [f"  {name:<{width}}  {description}" for name, (_, description) in COMMANDS.items()]
    rows.append(f"  {TEST_COMMAND[0]:<{width}}  {TEST_COMMAND[1]}")
    return "\n".join([
        "Использование: python source/main.py <команда> [аргументы]",
        "",
        "Команды:",
        *rows,
        "",
        "Справка по команде: python source/main.py <команда> --help",
    ])


def run_tests(argv: list[str]) -> int:
    suite = unittest.defaultTestLoader.discover(start_dir=str(TESTS_DIR), top_level_dir=str(SOURCE_DIR))
    verbosity = 2 if {"-v", "--verbose"} & set(argv) else 1
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(usage())
        return 2
    command, rest = args[0], args[1:]
    if command in {"-h", "--help", "help"}:
        print(usage())
        return 0
    if command == TEST_COMMAND[0]:
        return run_tests(rest)
    if command not in COMMANDS:
        print(f"[error] Неизвестная команда {command!r}\n")
        print(usage())
        return 2

    module = importlib.import_module(COMMANDS[command][0])
    # argparse берёт имя программы из sys.argv[0] — пусть справка показывает реальную команду.
    sys.argv[0] = f"main.py {command}"
    return module.main(rest)


if __name__ == "__main__":
    sys.exit(main())
