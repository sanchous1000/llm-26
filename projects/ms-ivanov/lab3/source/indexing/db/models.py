"""SQLAlchemy schema: texts and metadata live here, vectors in Qdrant."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

Json = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class Document(Base):
    """One converted source document (metadata from the bibliography)."""

    __tablename__ = "documents"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(16))
    year: Mapped[int | None] = mapped_column(Integer)
    venue: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list] = mapped_column(Json, default=list)
    priority: Mapped[int | None] = mapped_column(Integer)
    tags: Mapped[list] = mapped_column(Json, default=list)
    source_url: Mapped[str | None] = mapped_column(Text)
    markdown_path: Mapped[str] = mapped_column(Text)
    body_sha256: Mapped[str] = mapped_column(String(64))
    body_chars: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ChunkSet(Base):
    """All chunks produced with one chunking configuration (id = config digest)."""

    __tablename__ = "chunk_sets"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    config: Mapped[dict] = mapped_column(Json)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ChunkSetDocument(Base):
    """Which version of which document a chunk set was built from (for reruns)."""

    __tablename__ = "chunk_set_documents"

    chunk_set_id: Mapped[str] = mapped_column(
        ForeignKey("chunk_sets.id", ondelete="CASCADE"), primary_key=True
    )
    doc_key: Mapped[str] = mapped_column(
        ForeignKey("documents.key", ondelete="CASCADE"), primary_key=True
    )
    body_sha256: Mapped[str] = mapped_column(String(64))
    n_chunks: Mapped[int] = mapped_column(Integer)


class Chunk(Base):
    """A contiguous slice ``[start_char, end_char)`` of a document body."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    chunk_set_id: Mapped[str] = mapped_column(
        ForeignKey("chunk_sets.id", ondelete="CASCADE")
    )
    doc_key: Mapped[str] = mapped_column(
        ForeignKey("documents.key", ondelete="CASCADE")
    )
    ord: Mapped[int] = mapped_column(Integer)
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    section_path: Mapped[list] = mapped_column(Json, default=list)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embed_text: Mapped[str] = mapped_column(Text)

    __table_args__ = (Index("ix_chunks_set_doc_ord", "chunk_set_id", "doc_key", "ord"),)


class VectorCollection(Base):
    """A Qdrant collection and the configuration that produced its vectors."""

    __tablename__ = "vector_collections"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    chunk_set_id: Mapped[str] = mapped_column(
        ForeignKey("chunk_sets.id", ondelete="CASCADE")
    )
    embedding: Mapped[dict] = mapped_column(Json)
    index: Mapped[dict] = mapped_column(Json)
    n_points: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
