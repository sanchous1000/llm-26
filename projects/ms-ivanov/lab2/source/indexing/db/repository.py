"""Persistence operations used by the pipeline."""

from collections.abc import Iterable, Sequence

from sqlalchemy import Engine, create_engine, delete, func, select
from sqlalchemy.orm import Session

from source.indexing.chunking import Chunk as ChunkData
from source.indexing.config import ChunkingConfig, PipelineConfig
from source.indexing.db import models as m
from source.indexing.settings import InfraSettings


def make_engine(settings: InfraSettings) -> Engine:
    return create_engine(settings.postgres_url, pool_pre_ping=True)


def init_schema(engine: Engine) -> None:
    """Create missing tables (idempotent)."""
    m.Base.metadata.create_all(engine)


def upsert_document(session: Session, document: m.Document) -> None:
    session.merge(document)


def ensure_chunk_set(session: Session, config: ChunkingConfig) -> str:
    chunk_set_id = config.digest()
    if session.get(m.ChunkSet, chunk_set_id) is None:
        session.add(m.ChunkSet(id=chunk_set_id, config=config.model_dump(mode="json")))
    return chunk_set_id


def document_state(session: Session, chunk_set_id: str) -> dict[str, str]:
    """``doc_key -> body_sha256`` for documents already chunked in this set."""
    rows = session.execute(
        select(m.ChunkSetDocument.doc_key, m.ChunkSetDocument.body_sha256).where(
            m.ChunkSetDocument.chunk_set_id == chunk_set_id
        )
    )
    return dict(rows.all())


def replace_document_chunks(
    session: Session,
    chunk_set_id: str,
    doc_key: str,
    body_sha256: str,
    chunks: Sequence[ChunkData],
) -> None:
    """Atomically swap all chunks of one document inside a chunk set."""
    session.execute(
        delete(m.Chunk).where(
            m.Chunk.chunk_set_id == chunk_set_id, m.Chunk.doc_key == doc_key
        )
    )
    session.execute(
        delete(m.ChunkSetDocument).where(
            m.ChunkSetDocument.chunk_set_id == chunk_set_id,
            m.ChunkSetDocument.doc_key == doc_key,
        )
    )
    session.add_all(_row(chunk_set_id, chunk) for chunk in chunks)
    session.add(
        m.ChunkSetDocument(
            chunk_set_id=chunk_set_id,
            doc_key=doc_key,
            body_sha256=body_sha256,
            n_chunks=len(chunks),
        )
    )


def _row(chunk_set_id: str, chunk: ChunkData) -> m.Chunk:
    return m.Chunk(
        chunk_id=chunk.chunk_id,
        chunk_set_id=chunk_set_id,
        doc_key=chunk.doc_key,
        ord=chunk.index,
        start_char=chunk.start,
        end_char=chunk.end,
        token_count=chunk.token_count,
        section_path=list(chunk.section_path),
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text=chunk.text,
        embed_text=chunk.embed_text,
    )


def iter_chunks(
    session: Session, chunk_set_id: str, doc_keys: Iterable[str] | None = None
):
    """Chunks of a set ordered by (document, position)."""
    query = select(m.Chunk).where(m.Chunk.chunk_set_id == chunk_set_id)
    if doc_keys is not None:
        query = query.where(m.Chunk.doc_key.in_(list(doc_keys)))
    return session.scalars(query.order_by(m.Chunk.doc_key, m.Chunk.ord))


def register_collection(
    session: Session, name: str, config: PipelineConfig, n_points: int
) -> None:
    session.merge(
        m.VectorCollection(
            name=name,
            chunk_set_id=config.chunking.digest(),
            embedding=config.embedding.model_dump(mode="json"),
            index=config.index.model_dump(mode="json"),
            n_points=n_points,
        )
    )


def drop_chunk_set(session: Session, chunk_set_id: str) -> None:
    """Delete a chunk set and all rows that depend on it (no reliance on cascades)."""
    for table in (m.Chunk, m.ChunkSetDocument, m.VectorCollection):
        session.execute(delete(table).where(table.chunk_set_id == chunk_set_id))
    session.execute(delete(m.ChunkSet).where(m.ChunkSet.id == chunk_set_id))


def stats(session: Session) -> dict[str, int]:
    return {
        "documents": session.scalar(select(func.count()).select_from(m.Document)) or 0,
        "chunk_sets": session.scalar(select(func.count()).select_from(m.ChunkSet)) or 0,
        "chunks": session.scalar(select(func.count()).select_from(m.Chunk)) or 0,
        "collections": session.scalar(
            select(func.count()).select_from(m.VectorCollection)
        )
        or 0,
    }


ChunkWithDocument = tuple[m.Chunk, m.Document]


def load_chunks(
    session: Session, chunk_set_id: str, chunk_ids: Iterable[str]
) -> dict[str, ChunkWithDocument]:
    """Chunk rows with their documents, keyed by ``chunk_id``."""
    rows = session.execute(
        select(m.Chunk, m.Document)
        .join(m.Document, m.Document.key == m.Chunk.doc_key)
        .where(
            m.Chunk.chunk_set_id == chunk_set_id, m.Chunk.chunk_id.in_(list(chunk_ids))
        )
    )
    return {chunk.chunk_id: (chunk, document) for chunk, document in rows}


def chunk_window(
    session: Session, chunk_set_id: str, doc_key: str, first_ord: int, last_ord: int
) -> list[ChunkWithDocument]:
    """Consecutive chunks ``first_ord..last_ord`` of one document."""
    rows = session.execute(
        select(m.Chunk, m.Document)
        .join(m.Document, m.Document.key == m.Chunk.doc_key)
        .where(
            m.Chunk.chunk_set_id == chunk_set_id,
            m.Chunk.doc_key == doc_key,
            m.Chunk.ord.between(first_ord, last_ord),
        )
        .order_by(m.Chunk.ord)
    )
    return [(chunk, document) for chunk, document in rows]


def list_documents(
    session: Session, chunk_set_id: str, contains: str | None = None, limit: int = 100
) -> list[tuple[m.Document, int]]:
    """Documents of a chunk set with their chunk counts (optional title/key filter)."""
    query = (
        select(m.Document, m.ChunkSetDocument.n_chunks)
        .join(m.ChunkSetDocument, m.ChunkSetDocument.doc_key == m.Document.key)
        .where(m.ChunkSetDocument.chunk_set_id == chunk_set_id)
        .order_by(m.Document.key)
        .limit(limit)
    )
    if contains:
        pattern = f"%{contains}%"
        query = query.where(
            m.Document.title.ilike(pattern) | m.Document.key.ilike(pattern)
        )
    return [(document, n_chunks) for document, n_chunks in session.execute(query)]


def outline(session: Session, chunk_set_id: str, doc_key: str) -> list[m.Chunk]:
    """Chunks of a document in reading order (their section paths form the outline)."""
    return list(
        session.scalars(
            select(m.Chunk)
            .where(m.Chunk.chunk_set_id == chunk_set_id, m.Chunk.doc_key == doc_key)
            .order_by(m.Chunk.ord)
        )
    )
