"""CLI of the RAG layer: ask, chat, search and the MCP server."""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from source import config as project
from source.indexing.config import PipelineConfig, load_config
from source.rag.engine import RagEngine
from source.rag.errors import RagError
from source.rag.evaluation import run_evaluation, sample_questions, summarise
from source.rag.llm import OpencodeLlm
from source.rag.render import format_answer, format_chunks
from source.rag.service import build_engine, build_retriever

logger = logging.getLogger(__name__)
DEFAULT_CONFIG = project.PROJECT_ROOT / "configs" / "default.yaml"
MODES = ("auto", "dense", "sparse", "hybrid")
CHAT_HELP = """\
/help            this help
/sources         sources of the last answer
/ctx             passages retrieved for the last question
/top N           number of passages to retrieve
/mode M          auto | dense | sparse | hybrid
/condense on|off rewrite follow-up questions into standalone search queries
/reset           forget the conversation
/exit            quit"""


def register(commands, add_common: Callable[[argparse.ArgumentParser], None]) -> None:
    """Add the RAG commands (ask, chat, search, mcp, ...) to the top-level parser."""
    ask = commands.add_parser("ask", help="answer a question with citations (DeepSeek)")
    ask.add_argument("question", nargs="+")
    chat = commands.add_parser("chat", help="interactive chat over the papers")
    search = commands.add_parser(
        "search", help="retrieval only: print the best passages"
    )
    search.add_argument("question", nargs="+")
    mcp = commands.add_parser("mcp", help="serve retrieval tools to agents over MCP")
    rag_eval = commands.add_parser(
        "rag-eval", help="end-to-end check with an LLM judge"
    )
    for parser in (ask, chat, search, mcp, rag_eval):
        add_common(parser)
        parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        parser.add_argument(
            "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
            help="override a config value, e.g. rag.top_k=8 (repeatable)",
        )  # fmt: skip
    for parser in (ask, chat, search):
        parser.add_argument(
            "--top-k", type=int, help="passages to retrieve (rag.top_k)"
        )
        parser.add_argument("--mode", choices=MODES, help="search mode (rag.mode)")
        parser.add_argument(
            "--docs", nargs="+", metavar="KEY", help="only these documents"
        )
    for parser in (ask, search):
        parser.add_argument(
            "--json", action="store_true", help="machine-readable output"
        )
    for parser in (ask, chat):
        parser.add_argument("--model", help="opencode model (rag.model)")
        parser.add_argument(
            "--condense", action="store_true", help="rewrite follow-ups"
        )
    mcp.add_argument(
        "--transport", choices=("stdio", "streamable-http"), default="stdio"
    )
    mcp.add_argument("--host", default="127.0.0.1")
    mcp.add_argument("--port", type=int, default=8765)
    mcp.add_argument("--no-warmup", action="store_true", help="load the model lazily")
    rag_eval.add_argument("--n", type=int, default=30, help="questions to sample")
    rag_eval.add_argument("--seed", type=int, default=1)
    rag_eval.add_argument("--top-k", type=int, help="passages per question")
    rag_eval.add_argument("--eval-dir", type=Path, default=project.DEFAULT_EVAL_DIR)
    rag_eval.add_argument("--name", default="rag-eval", help="report file name")
    commands.add_parser(
        "mcp-config", help="print how to connect an agent to the MCP server"
    )


def _config(args: argparse.Namespace) -> PipelineConfig:
    overrides = list(args.overrides)
    if getattr(args, "model", None):
        overrides.append(f"rag.model={args.model}")
    try:
        return load_config(args.config, overrides)
    except (ValidationError, ValueError) as error:
        raise SystemExit(f"invalid configuration: {error}") from error


def _guard(action: Callable[[], int]) -> int:
    try:
        return action()
    except (RagError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def run_search(args: argparse.Namespace) -> int:
    def action() -> int:
        retriever = build_retriever(_config(args))
        chunks = retriever.search(
            " ".join(args.question),
            top_k=args.top_k,
            mode=args.mode,
            doc_keys=args.docs,
        )
        if args.json:
            print(
                json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2)
            )
        else:
            print(format_chunks(chunks))
        return 0

    return _guard(action)


def run_ask(args: argparse.Namespace) -> int:
    def action() -> int:
        engine = build_engine(_config(args))
        answer = engine.answer(
            " ".join(args.question), top_k=args.top_k, mode=args.mode,
            doc_keys=args.docs, condense=args.condense,
        )  # fmt: skip
        if args.json:
            print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(format_answer(answer))
        return 0

    return _guard(action)


def run_chat(args: argparse.Namespace) -> int:
    def action() -> int:
        engine = build_engine(_config(args))
        print("Chat over the indexed papers. /help for commands, /exit to quit.")
        return chat_loop(
            engine, input, print, top_k=args.top_k, mode=args.mode,
            doc_keys=args.docs, condense=args.condense,
        )  # fmt: skip

    return _guard(action)


def chat_loop(
    engine: RagEngine,
    read: Callable[[str], str],
    write: Callable[[str], None],
    *,
    top_k: int | None = None,
    mode: str | None = None,
    doc_keys: list[str] | None = None,
    condense: bool = False,
) -> int:
    """REPL: questions go through RAG, ``/commands`` control the session."""
    history: list[tuple[str, str]] = []
    last = None
    state = {"top_k": top_k, "mode": mode, "condense": condense}
    while True:
        try:
            line = read("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            write("")
            return 0
        if not line:
            continue
        if line.startswith("/"):
            if line in ("/exit", "/quit"):
                return 0
            last = _chat_command(line, state, history, last, write)
            continue
        try:
            last = engine.answer(
                line, history=history, top_k=state["top_k"], mode=state["mode"],
                doc_keys=doc_keys, condense=state["condense"],
            )  # fmt: skip
        except (RagError, ValueError) as error:
            write(f"error: {error}")
            continue
        history.append((line, last.text))
        write("\n" + format_answer(last))


def _chat_command(line, state, history, last, write):
    """Handle a slash command; returns the (possibly unchanged) last answer."""
    command, _, argument = line.partition(" ")
    argument = argument.strip()
    if command == "/help":
        write(CHAT_HELP)
    elif command == "/reset":
        history.clear()
        write("conversation cleared")
        return None
    elif command == "/sources":
        write(format_answer(last, show_timings=False) if last else "no answer yet")
    elif command == "/ctx":
        write(format_chunks(last.contexts) if last else "no answer yet")
    elif command == "/top" and argument.isdigit() and int(argument) > 0:
        state["top_k"] = int(argument)
        write(f"top_k = {argument}")
    elif command == "/mode" and argument in MODES:
        state["mode"] = argument
        write(f"mode = {argument}")
    elif command == "/condense" and argument in ("on", "off"):
        state["condense"] = argument == "on"
        write(f"condense = {argument}")
    else:
        write("unknown command, try /help")
    return last


def run_rag_eval(args: argparse.Namespace) -> int:
    def action() -> int:
        cfg = _config(args)
        engine = build_engine(cfg)
        judge_llm = OpencodeLlm(cfg.rag.model, cfg.rag.llm_timeout_s)
        questions = json.loads(
            (args.eval_dir / "questions.json").read_text(encoding="utf-8")
        )
        sample = sample_questions(questions["questions"], args.n, args.seed)

        def log_row(row) -> None:
            logger.info(
                "%s -> %s (evidence cited: %s)",
                row.question_id,
                row.verdict,
                row.evidence_cited,
            )

        rows = run_evaluation(
            engine, judge_llm, sample, top_k=args.top_k, on_row=log_row
        )
        summary = summarise(rows)
        out = args.eval_dir / "rag"
        out.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "rows": [r.to_dict() for r in rows]}
        (out / f"{args.name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    return _guard(action)


def mcp_config_text(
    python: str = sys.executable, root: Path = project.PROJECT_ROOT
) -> str:
    """Ready-to-paste client configuration with absolute paths of this checkout."""
    entry = {
        "command": python,
        "args": ["-W", "ignore", "-m", "source", "mcp"],
        "env": {"PYTHONPATH": str(root)},
    }
    snippet = json.dumps({"mcpServers": {"papers-rag": entry}}, indent=2)
    return (
        "# Claude Code (one command):\n"
        f"claude mcp add papers-rag -e PYTHONPATH={root} "
        f"-- {python} -W ignore -m source mcp\n\n"
        "# Any MCP client (.mcp.json / mcp settings):\n"
        f"{snippet}\n\n"
        "# Requires `docker compose up -d` (Qdrant + PostgreSQL) to be running."
    )


def run_mcp_config(_: argparse.Namespace) -> int:
    print(mcp_config_text())
    return 0


def run_mcp(args: argparse.Namespace) -> int:
    from source.rag.mcp_server import build_server

    def action() -> int:
        retriever = build_retriever(_config(args))
        retriever.check_ready()
        server = build_server(retriever, warm_up=not args.no_warmup)
        if args.transport == "stdio":
            server.run("stdio")
        else:
            server.run("streamable-http", host=args.host, port=args.port)
        return 0

    return _guard(action)


HANDLERS = {
    "ask": run_ask,
    "chat": run_chat,
    "search": run_search,
    "mcp": run_mcp,
    "rag-eval": run_rag_eval,
    "mcp-config": run_mcp_config,
}


def build_parser_for_tests() -> argparse.ArgumentParser:
    """A parser holding only the RAG commands (for unit tests)."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--bib", type=Path)
        sub.add_argument("--catalog", type=Path)
        sub.add_argument("--output", type=Path)

    register(commands, add_common)
    return parser
