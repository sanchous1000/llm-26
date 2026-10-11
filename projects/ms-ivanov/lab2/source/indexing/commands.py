"""CLI commands of the indexing stage (registered in :mod:`source.cli`)."""

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy.orm import sessionmaker

from source import config as project
from source.evaluation.documents import read_body
from source.indexing import vectorstore
from source.indexing.chunking import chunk_document
from source.indexing.config import PipelineConfig, load_config
from source.indexing.db import repository as repo
from source.indexing.diagnostics import evaluate_chunking, group_by_doc
from source.indexing.embedding import build_embedder
from source.indexing.hnsw_report import format_row, format_table, recommend
from source.indexing.hnsw_study import BuildSpec, run_grid
from source.indexing.pipeline import run_chunking, run_embedding, select_documents
from source.indexing.retrieval_check import (
    CHECK_HEADER,
    CHECK_RULE,
    check_retrieval,
    format_check_row,
)
from source.indexing.settings import InfraSettings
from source.indexing.tokens import HfTokenCounter

logger = logging.getLogger(__name__)
DEFAULT_CONFIG = project.PROJECT_ROOT / "configs" / "default.yaml"


def register(commands, add_common: Callable[[argparse.ArgumentParser], None]) -> None:
    """Add the indexing sub-commands to the top-level parser."""
    specs = {
        "db-init": "create the PostgreSQL tables",
        "chunk": "split documents into chunks and store them in PostgreSQL",
        "embed": "embed stored chunks into the Qdrant collection",
        "index": "chunk + embed",
        "index-status": "show what is stored in PostgreSQL and Qdrant",
        "index-drop": "delete the chunk set and collection of a configuration",
        "chunk-stats": "compare chunkings against the evaluation set (no DB, no model)",
        "index-check": "Recall@k / MRR of the collection on the evaluation questions",
        "hnsw-bench": "benchmark HNSW parameters on the collection's vectors",
        "index-apply": "apply the config's HNSW settings to the existing collection",
    }
    for name, help_text in specs.items():
        parser = commands.add_parser(name, help=help_text)
        add_common(parser)
        parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        parser.add_argument(
            "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
            help="override a config value, e.g. chunking.chunk_size=2048 (repeatable)",
        )  # fmt: skip
        parser.add_argument(
            "--keys", nargs="+", default=[], help="only these documents"
        )
        parser.add_argument("--limit", type=int, help="at most N documents")
        if name in ("chunk", "embed", "index"):
            parser.add_argument(
                "--rebuild", action="store_true", help="drop and recreate"
            )
        if name in ("chunk-stats", "index-check", "hnsw-bench"):
            parser.add_argument(
                "--eval-dir", type=Path, default=project.DEFAULT_EVAL_DIR
            )
        if name == "index-check":
            parser.add_argument("--lang", choices=("en", "ru"), default="en")
            parser.add_argument(
                "--vary", action="append", default=[], metavar="KEY=V1,V2,..",
                help="check several collections, e.g. chunking.chunk_size=512,1024",
            )  # fmt: skip
            parser.add_argument(
                "--markdown", type=Path, help="also write the table here"
            )
        if name == "hnsw-bench":
            parser.add_argument("--m", type=int, nargs="+", default=[4, 8, 16, 32, 64])
            parser.add_argument(
                "--ef-construct", type=int, nargs="+", default=[32, 64, 128, 256]
            )
            parser.add_argument(
                "--ef", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256]
            )
            parser.add_argument(
                "--quantization", nargs="+", default=["none"],
                choices=("none", "scalar", "binary"),
            )  # fmt: skip
            parser.add_argument("--passes", type=int, default=3)
            parser.add_argument(
                "--keep", action="store_true", help="keep scratch collections"
            )
            parser.add_argument("--name", default="grid", help="report file name")
        if name == "chunk-stats":
            parser.add_argument(
                "--vary", action="append", default=[], metavar="KEY=V1,V2,..",
                help="sweep one config key, e.g. chunking.chunk_size=512,1024,2048",
            )  # fmt: skip


def _config(args: argparse.Namespace) -> PipelineConfig:
    try:
        return load_config(args.config, args.overrides)
    except (ValidationError, ValueError) as error:
        raise SystemExit(f"invalid configuration: {error}") from error


def _infra() -> InfraSettings:
    try:
        return InfraSettings()
    except ValidationError as error:
        raise SystemExit(
            "database settings are missing: copy .env.example to .env "
            f"and run `docker compose up -d`\n{error}"
        ) from error


def _docs(args: argparse.Namespace, cfg: PipelineConfig):
    docs = select_documents(
        args.bib, args.catalog, args.output,
        min_priority=cfg.min_priority, keys=args.keys, limit=args.limit,
    )  # fmt: skip
    if not docs:
        raise SystemExit("no converted documents match the selection")
    return docs


def run_db_init(args: argparse.Namespace) -> int:
    repo.init_schema(repo.make_engine(_infra()))
    logger.info("PostgreSQL schema is ready")
    return 0


def run_chunk(args: argparse.Namespace) -> int:
    cfg = _config(args)
    docs = _docs(args, cfg)
    engine = repo.make_engine(_infra())
    repo.init_schema(engine)
    summary = run_chunking(
        sessionmaker(engine), docs, cfg, HfTokenCounter(cfg.chunking.tokenizer),
        rebuild=args.rebuild,
    )  # fmt: skip
    logger.info(
        "chunk set %s: %d documents chunked, %d unchanged, %d chunks, %d tokens",
        summary.chunk_set_id,
        summary.documents,
        summary.skipped,
        summary.chunks,
        summary.tokens,
    )
    return 0


def run_embed(args: argparse.Namespace) -> int:
    cfg = _config(args)
    docs = _docs(args, cfg)
    settings = _infra()
    summary = run_embedding(
        sessionmaker(repo.make_engine(settings)),
        vectorstore.make_client(settings),
        cfg,
        lambda: build_embedder(cfg.embedding),
        [doc.key for doc in docs],
        rebuild=args.rebuild,
    )
    logger.info(
        "collection %s: %d chunks (%d embedded now, %d already there) in %.0fs",
        summary.collection,
        summary.chunks,
        summary.embedded,
        summary.skipped,
        summary.seconds,
    )
    return 0


def run_index(args: argparse.Namespace) -> int:
    run_chunk(args)
    args.rebuild = False  # the collection must match the chunks just written
    return run_embed(args)


def run_status(args: argparse.Namespace) -> int:
    settings = _infra()
    engine = repo.make_engine(settings)
    repo.init_schema(engine)
    with sessionmaker(engine)() as session:
        print("PostgreSQL:", json.dumps(repo.stats(session)))
    client = vectorstore.make_client(settings)
    for collection in client.get_collections().collections:
        points = vectorstore.count(client, collection.name)
        print(f"Qdrant {collection.name}: {points} points")
    return 0


def run_drop(args: argparse.Namespace) -> int:
    cfg = _config(args)
    settings = _infra()
    client = vectorstore.make_client(settings)
    if client.collection_exists(cfg.collection_name):
        client.delete_collection(cfg.collection_name)
    with sessionmaker(repo.make_engine(settings)).begin() as session:
        repo.drop_chunk_set(session, cfg.chunking.digest())
    logger.info(
        "dropped collection %s and chunk set %s",
        cfg.collection_name,
        cfg.chunking.digest(),
    )
    return 0


def run_check(args: argparse.Namespace) -> int:
    base = _config(args)
    client = vectorstore.make_client(_infra())
    questions = json.loads(
        (args.eval_dir / "questions.json").read_text(encoding="utf-8")
    )["questions"]
    field = "question" if args.lang == "en" else "question_ru"
    embedder = build_embedder(base.embedding)
    lines = [CHECK_HEADER, CHECK_RULE]
    for label, cfg in _variants(args, base):
        if not client.collection_exists(cfg.collection_name):
            raise SystemExit(
                f"collection {cfg.collection_name} ({label}) is missing; run `index`"
            )
        modes = {
            "hybrid": ("dense", "sparse", "hybrid"),
            "dense": ("dense",),
            "sparse": ("sparse",),
        }
        scores = check_retrieval(
            client, cfg.collection_name, embedder, questions, modes[cfg.embedding.mode],
            language_field=field, params=vectorstore.search_params(cfg),
        )  # fmt: skip
        lines += [format_check_row(label, s) for s in scores]
    text = "\n".join(lines)
    print(text)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text + "\n", encoding="utf-8")
    return 0


def run_apply(args: argparse.Namespace) -> int:
    cfg = _config(args)
    client = vectorstore.make_client(_infra())
    if not client.collection_exists(cfg.collection_name):
        raise SystemExit(
            f"collection {cfg.collection_name} does not exist; run `index` first"
        )
    vectorstore.apply_index_config(client, cfg)
    vectorstore.wait_until_indexed(client, cfg.collection_name)
    info = client.get_collection(cfg.collection_name)
    logger.info(
        "%s: %s points, hnsw %s",
        cfg.collection_name,
        info.points_count,
        info.config.params.vectors["dense"].hnsw_config,
    )
    return 0


def run_hnsw_bench(args: argparse.Namespace) -> int:
    cfg = _config(args)
    client = vectorstore.make_client(_infra())
    if not client.collection_exists(cfg.collection_name):
        raise SystemExit(
            f"collection {cfg.collection_name} does not exist; run `index` first"
        )
    questions = json.loads(
        (args.eval_dir / "questions.json").read_text(encoding="utf-8")
    )["questions"]
    queries = build_embedder(cfg.embedding).embed([q["question"] for q in questions])
    specs = [
        BuildSpec(m, ef_construct, quantization)
        for quantization in args.quantization
        for m in args.m
        for ef_construct in args.ef_construct
    ]
    rows = run_grid(
        client, cfg.collection_name, queries, questions, specs,
        efs=args.ef, passes=args.passes, keep=args.keep,
        on_row=lambda row: logger.info(format_row(row)),
    )  # fmt: skip
    out = args.eval_dir / "hnsw"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(
        json.dumps([row.to_dict() for row in rows], indent=1), encoding="utf-8"
    )
    (out / f"{args.name}.md").write_text(format_table(rows) + "\n", encoding="utf-8")
    best = recommend(rows)
    print(f"{len(rows)} measurements -> {out / (args.name + '.md')}")
    print("recommended:", format_row(best) if best else "none met the quality targets")
    return 0


def run_chunk_stats(args: argparse.Namespace) -> int:
    base = _config(args)
    docs = _docs(args, base)
    questions = json.loads(
        (args.eval_dir / "questions.json").read_text(encoding="utf-8")
    )["questions"]
    keys = {doc.key for doc in docs}
    questions = [q for q in questions if q["doc_key"] in keys]
    bodies = {doc.key: read_body(doc.path) for doc in docs}
    titles = {doc.key: doc.title for doc in docs}
    counter = HfTokenCounter(base.chunking.tokenizer)
    columns = ("variant", "chunks", "median", "p90", "contained", "dilution")
    print(f"{columns[0]:44}" + "".join(f"{name:>10}" for name in columns[1:]))
    for label, cfg in _variants(args, base):
        chunks = [
            c for key, body in bodies.items()
            for c in chunk_document(key, titles[key], body, cfg.chunking, counter)
        ]  # fmt: skip
        report = evaluate_chunking(group_by_doc(chunks), questions, bodies, counter)
        print(
            f"{label:44}{report.chunks:10d}{report.tokens_median:10.0f}"
            f"{report.tokens_p90:10d}{report.contained:10.1%}{report.dilution_median:10.1%}"
        )
    return 0


def _variants(args: argparse.Namespace, base: PipelineConfig):
    if not args.vary:
        yield "config", base
        return
    key, _, raw = args.vary[0].partition("=")
    for value in raw.split(","):
        yield f"{key}={value}", load_config(
            args.config, [*args.overrides, f"{key}={value}"]
        )


HANDLERS = {
    "db-init": run_db_init,
    "chunk": run_chunk,
    "embed": run_embed,
    "index": run_index,
    "index-status": run_status,
    "index-drop": run_drop,
    "chunk-stats": run_chunk_stats,
    "index-check": run_check,
    "hnsw-bench": run_hnsw_bench,
    "index-apply": run_apply,
}
