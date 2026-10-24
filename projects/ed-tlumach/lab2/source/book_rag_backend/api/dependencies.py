from fastapi import Request

from book_rag_backend.services.documents import DocumentService
from book_rag_backend.services.rag import RAGService


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag


def get_document_service(request: Request) -> DocumentService:
    rag = request.app.state.rag
    sp = request.app.state.config.splitter
    return DocumentService(
        qdrant=rag.qdrant, vectorizer=rag.vectorizer, embeddings=rag.embeddings, sparse=rag.sparse,
        default_chunk_size=sp.chunk_size, default_chunk_overlap=sp.chunk_overlap,
        default_include_headings=sp.include_headings,
    )