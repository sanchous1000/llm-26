import asyncio
from datetime import datetime

from fastapi import UploadFile
from loguru import logger

from book_rag_backend.services.embeddings import EmbeddingsService, SparseEmbeddingsService
from book_rag_backend.services.parser import Document, ParserService
from book_rag_backend.services.qdrant import QdrantService
from book_rag_backend.services.splitter import SplitterService, chunk_documents


class DocumentService:

    def __init__(self, qdrant: QdrantService, vectorizer: str,
                 embeddings: EmbeddingsService | None, sparse: SparseEmbeddingsService | None,
                 default_chunk_size: int, default_chunk_overlap: int, default_include_headings: bool):
        self.qdrant = qdrant
        self.vectorizer = vectorizer
        self.embeddings = embeddings
        self.sparse = sparse
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.default_include_headings = default_include_headings

    async def add_document(self, file: UploadFile, chunk_size: int | None = None,
                            chunk_overlap: int | None = None, include_headings: bool | None = None) -> dict:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise TypeError("NotSupportedFormat")

        content = await file.read()
        if not content:
            raise ValueError("BadFile")

        try:
            pages = await asyncio.to_thread(ParserService().parse_pdf_pages, content)
        except Exception:
            raise ValueError("BadFile")
        if not pages:
            raise ValueError("BadFile")

        doc = Document(source=file.filename, date=datetime.now().strftime("%Y-%m-%d"), pages=pages)
        cs = chunk_size or self.default_chunk_size
        co = chunk_overlap if chunk_overlap is not None else self.default_chunk_overlap
        ih = self.default_include_headings if include_headings is None else include_headings
        chunks = chunk_documents([doc], SplitterService(cs, co, self.embeddings), ih)
        if not chunks:
            raise ValueError("BadFile")

        texts = [c["embed_text"] for c in chunks]
        dense = (await asyncio.to_thread(self.embeddings.embed_batch, texts)
                 if self.vectorizer in ("dense", "hybrid") else None)
        sparse = (await asyncio.to_thread(self.sparse.embed_batch, texts)
                  if self.vectorizer in ("sparse", "hybrid") else None)

        await self.qdrant.upsert_chunks(chunks, dense, sparse)
        logger.info(f"Добавлен документ {file.filename}: {len(chunks)} чанков (chunk_size={cs})")
        return {"source": doc.source, "n_pages": len(doc.pages), "n_chunks": len(chunks)}

    async def list_documents(self) -> list[dict]:
        return await self.qdrant.list_sources()

    async def delete_document(self, source: str) -> None:
        if not await self.qdrant.delete_source(source):
            raise ValueError("NotFound")