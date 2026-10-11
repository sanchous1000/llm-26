"""Command line interface: ``python -m source download|report``."""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from source import config
from source.bibliography import SourceKind, load_sources
from source.conversion_pipeline import run_conversion
from source.downloaders import DownloadContext, PdfPolicy, build_downloaders
from source.evaluation import opencode as opencode_client
from source.evaluation.export import SELECTION_HELP, export
from source.evaluation.generation import GenerationSettings, generate_all
from source.evaluation.review import render
from source.indexing import commands as indexing_commands
from source.net import HttpClient
from source.pipeline import run_pipeline, select_sources
from source.rag import commands as rag_commands
from source.report import format_report
from source.storage import EntryRecord, EntryStore

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m source", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download the corpus (resumable)")
    _add_common_arguments(download)
    _add_selection_arguments(download)
    download.add_argument(
        "--pdf",
        type=PdfPolicy,
        default=PdfPolicy.FALLBACK,
        choices=list(PdfPolicy),
        help="arXiv PDFs: never / only if no HTML (default) / always",
    )
    download.add_argument(
        "--force", action="store_true", help="re-download existing files"
    )
    download.add_argument(
        "--retry-missing",
        action="store_true",
        help="retry files recorded as permanently missing",
    )
    convert = commands.add_parser(
        "convert", help="convert downloaded files to Markdown"
    )
    _add_common_arguments(convert)
    _add_selection_arguments(convert)
    convert.add_argument(
        "--force", action="store_true", help="convert again even if up to date"
    )
    convert.add_argument(
        "--workers", type=int, help="parallel processes (default: CPUs - 1)"
    )
    convert.add_argument(
        "--no-references",
        action="store_true",
        help="omit the bibliography of arXiv papers",
    )
    questions = commands.add_parser(
        "gen-questions", help="generate candidate evaluation questions with an LLM"
    )
    _add_common_arguments(questions)
    _add_selection_arguments(questions)
    questions.add_argument("--min-priority", type=int, default=3)
    questions.add_argument("--model", default=opencode_client.DEFAULT_MODEL)
    questions.add_argument(
        "--per-part", type=int, default=5, help="questions per request"
    )
    questions.add_argument("--max-chars", type=int, default=150_000)
    questions.add_argument(
        "--timeout", type=int, default=900, help="seconds per request"
    )
    questions.add_argument("--force", action="store_true")
    questions.add_argument("--eval-dir", type=Path, default=config.DEFAULT_EVAL_DIR)
    review = commands.add_parser("review-questions", help="print verified candidates")
    review.add_argument("keys", nargs="+")
    review.add_argument("--eval-dir", type=Path, default=config.DEFAULT_EVAL_DIR)
    review.add_argument(
        "--snippet", type=int, default=260, help="evidence chars to show"
    )
    export_cmd = commands.add_parser("export-questions", help="build questions.json")
    export_cmd.add_argument(
        "--selection", type=Path, required=True, help=SELECTION_HELP
    )
    export_cmd.add_argument("--eval-dir", type=Path, default=config.DEFAULT_EVAL_DIR)
    indexing_commands.register(commands, _add_common_arguments)
    rag_commands.register(commands, _add_common_arguments)
    report = commands.add_parser("report", help="summarise what has been downloaded")
    _add_common_arguments(report)
    return parser


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--kinds",
        nargs="+",
        type=SourceKind,
        default=[],
        choices=list(SourceKind),
        help="only these source kinds",
    )
    parser.add_argument(
        "--keys", nargs="+", default=[], help="only these bibliography keys"
    )
    parser.add_argument("--limit", type=int, help="process at most N sources")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bib", type=Path, default=config.DEFAULT_BIB_PATH)
    parser.add_argument("--catalog", type=Path, default=config.DEFAULT_CATALOG_PATH)
    parser.add_argument("--output", type=Path, default=config.DEFAULT_OUTPUT_DIR)


def run_download(args: argparse.Namespace) -> int:
    sources = load_sources(args.bib, args.catalog)
    selected = select_sources(
        sources, kinds=args.kinds, keys=args.keys, limit=args.limit
    )
    logger.info("Selected %d of %d sources", len(selected), len(sources))
    store = EntryStore(args.output)
    context = DownloadContext(
        client=HttpClient(),
        force=args.force,
        retry_missing=args.retry_missing,
        pdf_policy=args.pdf,
    )
    try:
        run_pipeline(selected, store, build_downloaders(context))
    finally:
        _refresh_manifest(sources, store)
    return 0


def run_convert(args: argparse.Namespace) -> int:
    sources = load_sources(args.bib, args.catalog)
    selected = select_sources(
        sources, kinds=args.kinds, keys=args.keys, limit=args.limit
    )
    store = EntryStore(args.output)
    records = _load_records(selected, store)
    try:
        run_conversion(
            records,
            store,
            force=args.force,
            include_references=not args.no_references,
            workers=args.workers,
        )
    finally:
        _refresh_manifest(sources, store)
    return 0


def run_gen_questions(args: argparse.Namespace) -> int:
    sources = load_sources(args.bib, args.catalog)
    chosen = [s for s in sources if (s.priority or 0) >= args.min_priority]
    selected = select_sources(
        chosen, kinds=args.kinds, keys=args.keys, limit=args.limit
    )
    store = EntryStore(args.output)
    records = [
        r for r in _load_records(selected, store) if r.conversion.get("status") == "ok"
    ]
    logger.info(
        "Generating questions for %d documents with %s", len(records), args.model
    )
    settings = GenerationSettings(
        model=args.model,
        per_part=args.per_part,
        max_chars=args.max_chars,
        timeout_s=args.timeout,
    )
    generate_all(
        records, store, args.eval_dir / "candidates", settings, force=args.force
    )
    return 0


def run_review(args: argparse.Namespace) -> int:
    for key in args.keys:
        print(
            render(args.eval_dir / "candidates" / f"{key}.json", args.snippet),
            end="\n\n",
        )
    return 0


def run_export(args: argparse.Namespace) -> int:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    questions = export(
        selection, args.eval_dir / "candidates", args.eval_dir / "questions.json"
    )
    logger.info(
        "Exported %d questions for %d documents", len(questions), len(selection)
    )
    return 0


def run_report(args: argparse.Namespace) -> int:
    sources = load_sources(args.bib, args.catalog)
    records = _load_records(sources, EntryStore(args.output))
    print(format_report(records, expected_total=len(sources)))
    return 0


def _refresh_manifest(sources: Sequence, store: EntryStore) -> None:
    records = _load_records(sources, store)
    path = store.write_manifest(records)
    logger.info("Manifest with %d entries written to %s", len(records), path)


def _load_records(sources: Sequence, store: EntryStore) -> list[EntryRecord]:
    loaded = (store.load(source) for source in sources)
    return [record for record in loaded if record is not None]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    handlers = {
        "download": run_download,
        "convert": run_convert,
        "gen-questions": run_gen_questions,
        "review-questions": run_review,
        "export-questions": run_export,
        "report": run_report,
        **indexing_commands.HANDLERS,
        **rag_commands.HANDLERS,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
