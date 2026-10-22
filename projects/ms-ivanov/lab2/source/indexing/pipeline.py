"""Chunk -> PostgreSQL, embed -> Qdrant. Both stages are idempotent and resumable."""

import hashlib
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from sqlalchemy.orm import Session, sessionmaker

from source.bibliography import load_sources
from source.evaluation.documents import read_body
from source.indexing import vectorstore
from source.indexing.chunking import chunk_document
from source.indexing.config import PipelineConfig
from source.indexing.db import models as m
from source.indexing.db import repository as repo
from source.indexing.embedding import Embedder
from source.indexing.tokens import TokenCounter
from source.indexing.vectorstore import VectorPoint
from source.storage import EntryStore

logger = logging.getLogger(__name__)
SessionFactory = sessionmaker[Session]


@dataclass(frozen=True)
class DocInfo:
    """A converted document that can be indexed."""

    key: str
    title: str
    kind: str
    year: int | None
    venue: str | None
    authors: tuple[str, ...]
    priority: int | None
    tags: tuple[str, ...]
    source_url: str
    markdown_path: str
    path: Path


@dataclass(frozen=True)
class ChunkingSummary:
    chunk_set_id: str
    documents: int
    skipped: int
    chunks: int
    tokens: int


@dataclass(frozen=True)
class EmbeddingSummary:
    collection: str
    chunks: int
    embedded: int
    skipped: int
    tokens: int
    seconds: float


def select_documents(
    bib: Path,
    catalog: Path,
    papers: Path,
    *,
    min_priority: int,
    keys: Sequence[str] = (),
    limit: int | None = None,
) -> list[DocInfo]:
    """Converted documents with at least ``min_priority`` stars, sorted by key."""
    store = EntryStore(papers)
    docs: list[DocInfo] = []
    for source in load_sources(bib, catalog):
        if (source.priority or 0) < min_priority or (keys and source.key not in keys):
            continue
        record = store.load(source)
        if record is None or record.conversion.get("status") != "ok":
            continue
        path = store.entry_dir(source) / record.conversion["path"]
        docs.append(
            DocInfo(
                source.key, source.title, source.kind.value, source.year, source.venue,
                source.authors, source.priority, source.tags, source.url,
                path.relative_to(papers).as_posix(), path,
            )
        )  # fmt: skip
    docs.sort(key=lambda d: d.key)
    return docs[:limit] if limit else docs


def run_chunking(
    sessions: SessionFactory,
    docs: Sequence[DocInfo],
    config: PipelineConfig,
    counter: TokenCounter,
    *,
    rebuild: bool = False,
) -> ChunkingSummary:
    """Chunk documents into PostgreSQL; unchanged documents are skipped."""
    with sessions.begin() as session:
        chunk_set_id = repo.ensure_chunk_set(session, config.chunking)
        if rebuild:
            repo.drop_chunk_set(session, chunk_set_id)
            session.flush()
            chunk_set_id = repo.ensure_chunk_set(session, config.chunking)
        known = repo.document_state(session, chunk_set_id)
    done = skipped = n_chunks = n_tokens = 0
    for number, doc in enumerate(docs, start=1):
        body = read_body(doc.path)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if known.get(doc.key) == digest:
            skipped += 1
            continue
        chunks = chunk_document(doc.key, doc.title, body, config.chunking, counter)
        with sessions.begin() as session:
            repo.upsert_document(session, _document_row(doc, digest, len(body)))
            session.flush()
            repo.replace_document_chunks(session, chunk_set_id, doc.key, digest, chunks)
        done, n_chunks = done + 1, n_chunks + len(chunks)
        n_tokens += sum(c.token_count for c in chunks)
        logger.info("[%d/%d] %s: %d chunks", number, len(docs), doc.key, len(chunks))
    return ChunkingSummary(chunk_set_id, done, skipped, n_chunks, n_tokens)


def _document_row(doc: DocInfo, digest: str, chars: int) -> m.Document:
    return m.Document(
        key=doc.key, title=doc.title, kind=doc.kind, year=doc.year, venue=doc.venue,
        authors=list(doc.authors), priority=doc.priority, tags=list(doc.tags),
        source_url=doc.source_url, markdown_path=doc.markdown_path,
        body_sha256=digest, body_chars=chars,
    )  # fmt: skip


def run_embedding(
    sessions: SessionFactory,
    client: QdrantClient,
    config: PipelineConfig,
    make_embedder: Callable[[], Embedder],
    doc_keys: Sequence[str],
    *,
    rebuild: bool = False,
    group_size: int = 32,
) -> EmbeddingSummary:
    """Embed the chunks of ``doc_keys`` into the configuration's collection."""
    chunk_set_id = config.chunking.digest()
    embedder: Embedder | None = None
    name = config.collection_name
    if rebuild or not client.collection_exists(name):
        embedder = make_embedder()  # the model defines the vector size
        vectorstore.ensure_collection(
            client, config, embedder.dense_dim, recreate=rebuild
        )
    with sessions() as session:
        rows = list(repo.iter_chunks(session, chunk_set_id, doc_keys))
        payloads = [_payload(row, chunk_set_id) for row in rows]
    started = time.monotonic()
    embedded = skipped = tokens = 0
    for group in _groups(list(zip(rows, payloads, strict=True)), group_size):
        present = vectorstore.existing_ids(
            client, name, [row.chunk_id for row, _ in group]
        )
        todo = [(row, payload) for row, payload in group if row.chunk_id not in present]
        skipped += len(group) - len(todo)
        if not todo:
            continue
        embedder = embedder or make_embedder()
        vectors = embedder.embed([row.embed_text for row, _ in todo])
        points = [
            VectorPoint(row.chunk_id, v.dense, v.sparse, payload)
            for (row, payload), v in zip(todo, vectors, strict=True)
        ]
        vectorstore.upsert(client, name, points, config.embedding.mode)
        embedded += len(todo)
        tokens += sum(v.n_tokens for v in vectors)
        rate = tokens / max(time.monotonic() - started, 1e-9)
        logger.info(
            "embedded %d/%d chunks (%.0f tok/s)", embedded + skipped, len(rows), rate
        )
    total = vectorstore.count(client, name)
    with sessions.begin() as session:
        repo.register_collection(session, name, config, total)
    return EmbeddingSummary(
        name, len(rows), embedded, skipped, tokens, time.monotonic() - started
    )


def _payload(row: m.Chunk, chunk_set_id: str) -> dict:
    return {
        "chunk_id": row.chunk_id,
        "chunk_set_id": chunk_set_id,
        "doc_key": row.doc_key,
        "ord": row.ord,
        "start_char": row.start_char,
        "end_char": row.end_char,
        "token_count": row.token_count,
        "section_path": row.section_path,
        "page_start": row.page_start,
    }


def _groups(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
