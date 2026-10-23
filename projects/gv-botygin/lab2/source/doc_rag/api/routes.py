from dataclasses import asdict

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from loguru import logger

from doc_rag.api.dependencies import get_rag_service
from doc_rag.api.schemas import AskResponse, ErrorResponse, QueryRequest, SearchHit, SearchResponse
from doc_rag.services.rag import RAGService

router = APIRouter()


@router.get("/health", tags=["service"])
def health(rag: RAGService = Depends(get_rag_service)):
    return {
        "status": "healthy",
        "collection": rag.qdrant.collection,
        "points": rag.qdrant.count(),
        "defaults": rag.defaults.model_dump(),
    }


@router.post("/search", response_model=SearchResponse, tags=["search"])
def search(request: QueryRequest, rag: RAGService = Depends(get_rag_service)):
    """Только поиск: релевантные фрагменты без вызова LLM."""
    hits = rag.search(request.query, request.top_k, request.score_threshold)
    return SearchResponse(results=[SearchHit(**asdict(h)) for h in hits])


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={502: {"model": ErrorResponse}},
    tags=["search"],
)
def ask(request: QueryRequest, rag: RAGService = Depends(get_rag_service)):
    """Поиск + ответ LLM со ссылками [n] на найденные фрагменты."""
    try:
        result = rag.ask(request.query, request.top_k, request.score_threshold)
    except (httpx.HTTPError, RuntimeError) as e:
        logger.error(f"LLM error: {e!r}")
        return JSONResponse(
            status_code=502,
            content={"errorCode": "LLMError", "userMessage": f"Не удалось получить ответ от LLM: {e}"},
        )
    return AskResponse(
        answer=result.answer,
        citations=[asdict(c) for c in result.citations],
        retrieved=[asdict(c) for c in result.retrieved],
    )
