"""CLI: python -m server.cli "вопрос" [--show-prompt] (запуск из lab2/src)."""

from __future__ import annotations

import argparse
import json

from config import load_config

from .engine import build_engine
from .llm import LlmUnavailable, PromptTooLong


def show_prompt(engine, question: str, top_n: int | None) -> None:
    messages, blocks = engine.prepare(question, top_n)
    print(f"Найдено фрагментов: {len(blocks)}")
    for b in blocks:
        print(f"  [{b['n']}] score={b['score']} {b['source']} стр. {b['page']} :: {b['section']}")
    print("\n--- SYSTEM ---\n" + messages[0]["content"])
    print("\n--- USER ---\n" + messages[1]["content"])


def print_result(result: dict) -> None:
    print(result["answer"])
    print("\nИсточники:")
    for c in result["citations"]:
        mark = "" if c["cited"] else " (не процитирован моделью)"
        print(f"  [{c['n']}] {c['source']}, стр. {c['page']}, раздел: {c['section']}{mark}")
        print(f"      {c['snippet']}")
    print(f"\nмодель: {result['model']}  usage: {result['usage']}  latency: {result['latency_s']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-запрос к документам")
    parser.add_argument("question", nargs="?", help="вопрос; без него запускается интерактивный режим")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--show-prompt", action="store_true", help="только поиск и промпт, без вызова LLM")
    parser.add_argument("--json", action="store_true", help="вывести ответ как JSON")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    engine = build_engine(config, with_llm=not args.show_prompt)

    def handle(question: str) -> None:
        if args.show_prompt:
            show_prompt(engine, question, args.top_n)
            return
        try:
            result = engine.ask(question, args.top_n)
        except (LlmUnavailable, PromptTooLong) as error:
            raise SystemExit(f"Ошибка: {error}")
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result(result)

    if args.question:
        handle(args.question)
        return
    while True:
        try:
            question = input("\nВопрос (пустая строка для выхода): ").strip()
        except EOFError:
            break
        if not question:
            break
        handle(question)


if __name__ == "__main__":
    main()
