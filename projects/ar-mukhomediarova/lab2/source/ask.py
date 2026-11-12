"""CLI к RAG-агенту.

    python source/ask.py "Какие критерии приёмки заданы в ТЗ?"
    python source/ask.py --mode dense --top-k 8 --show-context "..."
    python source/ask.py            # интерактивный режим
"""
from __future__ import annotations

import argparse
import json

import settings
from rag import RAGEngine, format_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    settings.add_common_args(parser)
    settings.add_index_args(parser)
    parser.add_argument("question", nargs="*")
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"])
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--show-context", action="store_true", help="показать все найденные чанки")
    parser.add_argument("--json", action="store_true", help="вывести ответ в JSON")
    args = parser.parse_args()
    engine = RAGEngine(settings.config_from_args(args))

    def run(q: str):
        res = engine.ask(q, top_k=args.top_k, mode=args.mode)
        print(json.dumps(res, ensure_ascii=False, indent=2) if args.json else format_answer(res, args.show_context))

    if args.question:
        run(" ".join(args.question))
        return
    print("RAG по материалам НИР. Пустая строка — выход.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        run(q)


if __name__ == "__main__":
    main()
