from fastapi import Request

from doc_rag.services.rag import RAGService


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag
