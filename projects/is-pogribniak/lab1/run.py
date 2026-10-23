"""Точка входа лабы 1. Весь код — в source/, здесь только разбор команд.

  uv run python run.py experiment --modes A     основной прогон (+ замер prefill)
  uv run python run.py analyze                  метрики, таблицы, графики по results/
  uv run python run.py trial --model mistral:7b --prompt P3 --param temperature=0.3 -n 3
                                                разовый эксперимент, в results/ не пишется
"""

import argparse
import sys

import yaml
from source.analysis import analyze
from source.config import CONFIG, MODES, load_config, missing_params_b
from source.runner import run_experiments, run_trial


def parse_params(pairs: list | None) -> dict:
    """['temperature=0.3', 'max_tokens=200'] → {'temperature': 0.3, 'max_tokens': 200}"""
    params = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        if not value:
            sys.exit(f"--param ждёт формат ключ=значение, получено «{pair}»")
        params[key] = yaml.safe_load(value)   # числа и true/false приводятся к нужному типу
    return params


def main() -> None:
    ap = argparse.ArgumentParser(description="Лаба 1: прогоны LLM и анализ")
    ap.add_argument("--config", default=CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("experiment", help="основной прогон")
    p.add_argument("--models", nargs="*", help="подмножество моделей из config.yaml")
    p.add_argument("--modes", nargs="*", choices=MODES, default=list(MODES))
    p.add_argument("--no-prefill", action="store_true", help="пропустить замер prefill")

    sub.add_parser("analyze", help="метрики, таблицы и графики")

    p = sub.add_parser("trial", help="разовый эксперимент с одной моделью")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True, help="P1 / P2 / P3")
    p.add_argument("--mode", choices=MODES, default="A", help="база параметров: A — дефолты, B — params_b")
    p.add_argument("--param", action="append", metavar="КЛЮЧ=ЗНАЧЕНИЕ", help="параметр запроса, можно несколько раз")
    p.add_argument("-n", "--repeats", type=int, default=1)

    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "experiment":
        if "B" in args.modes and (missing := missing_params_b(cfg)):
            sys.exit(f"Для режима B не заданы params_b у {', '.join(missing)} — "
                     f"заполните config.yaml или запустите с --modes A")
        models = [m for m in cfg["models"] if not args.models or m["name"] in args.models]
        run_experiments(cfg, models, args.modes, prefill=not args.no_prefill)
    elif args.cmd == "analyze":
        analyze(cfg)
    elif args.cmd == "trial":
        run_trial(cfg, args.model, args.prompt, args.mode, parse_params(args.param), args.repeats)


if __name__ == "__main__":
    main()
