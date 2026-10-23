"""Индексация PDF: парсинг -> Markdown -> нарезка -> dense-эмбеддинги -> загрузка в Qdrant.

Коллекция определяется параметрами splitter/embeddings (см. Config.collection_name):
смена параметров создаёт новую коллекцию, старая остаётся для сравнения.

Примеры:
    uv run python scripts/build_index.py                      # все PDF из documents/, существующие пропускаются
    uv run python scripts/build_index.py --replace            # существующие документы переиндексировать
    uv run python scripts/build_index.py --file a.pdf         # один документ
    uv run python scripts/build_index.py --chunk-size 128 --chunk-overlap 16 --no-headers-in-text
    uv run python scripts/build_index.py --reset              # очистить коллекцию текущей конфигурации и загрузить заново
"""
import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger

from doc_rag.cli import add_splitter_args, apply_splitter_overrides
from doc_rag.config import SOURCE_DIR, load_config
from doc_rag.services.embeddings import EmbeddingsService
from doc_rag.services.parser import ParserService
from doc_rag.services.qdrant import QdrantService
from doc_rag.services.splitter import SplitterService

DEFAULT_DOCS_DIR = SOURCE_DIR.parent / "documents"
RESULTS_DIR = SOURCE_DIR.parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="путь к config.yml (по умолчанию DOC_RAG_CONFIG или source/config.yml)")
    p.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR, help="папка с PDF")
    p.add_argument("--file", help="индексировать только этот файл (имя в docs-dir или путь)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--replace", action="store_true", help="заменить документы, которые уже есть в коллекции")
    mode.add_argument("--reset", action="store_true", help="удалить коллекцию текущей конфигурации и собрать заново")

    add_splitter_args(p)
    return p.parse_args()


def find_documents(args: argparse.Namespace) -> list[Path]:
    if args.file:
        f = Path(args.file)
        f = f if f.exists() else args.docs_dir / args.file
        if not f.is_file():
            sys.exit(f"Файл не найден: {args.file}")
        return [f]
    return sorted(p for p in args.docs_dir.glob("*.pdf") if p.is_file())


def record_index_stats(collection: str, seconds: float, stats: dict, device: str) -> None:
    """Затраты на индексацию для сравнения конфигураций (читает evaluate.py)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "index_stats.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[collection] = {
        "index_s": round(seconds, 1),
        "chunks_loaded": stats["chunks"],
        "docs": stats["loaded"] + stats["replaced"],
        "device": device,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # кириллица в логах на Windows
    args = parse_args()
    config = apply_splitter_overrides(load_config(args.config), args)
    collection = config.collection_name()

    docs = find_documents(args)
    if not docs:
        sys.exit(f"PDF не найдены в {args.docs_dir}")

    s = config.splitter
    logger.info(
        f"Коллекция: {collection} | модель: {config.embeddings.model_name} | "
        f"chunk_size={s.chunk_size} overlap={s.chunk_overlap} headers_in_text={s.include_headers_in_text}"
    )

    embeddings = EmbeddingsService(config.embeddings)
    qdrant = QdrantService(config.qdrant, collection)
    parser = ParserService()
    splitter = SplitterService(s, embeddings.count_tokens, embeddings.max_seq_length)

    if args.reset and qdrant.collection_exists():
        qdrant.drop_collection()
        logger.warning(f"Коллекция {collection} удалена (--reset)")
    if qdrant.ensure_collection(embeddings.dim):
        logger.info(f"Коллекция {collection} создана")

    stats = {"loaded": 0, "replaced": 0, "skipped": 0, "failed": 0, "chunks": 0}
    started = time.perf_counter()

    for path in docs:
        source = path.name
        exists = qdrant.document_exists(source)
        if exists and not args.replace:
            logger.info(f"[skip] {source}: уже в коллекции (используйте --replace для замены)")
            stats["skipped"] += 1
            continue

        try:
            t0 = time.perf_counter()
            pages = parser.parse_pdf(path)
            chunks = splitter.split(pages)
            if not chunks:
                logger.warning(f"[fail] {source}: не удалось извлечь текст (скан без текстового слоя?)")
                stats["failed"] += 1
                continue
            vectors = embeddings.embed_batch([c.text for c in chunks])

            # удаляем старые чанки только когда новые уже готовы - при ошибке документ не пропадёт
            if exists:
                qdrant.delete_document(source)
            qdrant.upsert_chunks(source, str(path), chunks, vectors, config.index_params())
        except Exception as e:
            logger.error(f"[fail] {source}: {e}")
            stats["failed"] += 1
            continue

        stats["replaced" if exists else "loaded"] += 1
        stats["chunks"] += len(chunks)
        logger.success(
            f"[{'replace' if exists else 'load'}] {source}: {len(pages)} стр., {len(chunks)} чанков, "
            f"{time.perf_counter() - t0:.1f} с"
        )

    logger.info(
        f"Готово за {time.perf_counter() - started:.1f} с: загружено {stats['loaded']}, заменено {stats['replaced']}, "
        f"пропущено {stats['skipped']}, ошибок {stats['failed']}, чанков {stats['chunks']}; "
        f"в коллекции {collection}: {qdrant.count()} точек, документы: {qdrant.list_documents()}"
    )
    if stats["loaded"] + stats["replaced"]:
        record_index_stats(collection, time.perf_counter() - started, stats, embeddings.device)
    qdrant.close()
    if stats["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
