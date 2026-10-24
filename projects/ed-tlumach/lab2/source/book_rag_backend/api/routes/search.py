from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from loguru import logger

from book_rag_backend.api.dependencies import get_rag_service
from book_rag_backend.api.schemas.search import SearchRequest, SearchResponse
from book_rag_backend.services.rag import RAGService

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, rag: RAGService = Depends(get_rag_service)):
    if not req.messages:
        return JSONResponse(status_code=400, content={"errorCode": "EmptyMessages", "userMessage": "Нет сообщений"})
    try:
        result = await rag.search([m.model_dump() for m in req.messages])
        return SearchResponse(**result)
    except Exception:
        logger.exception("Ошибка в /search")
        return JSONResponse(status_code=500, content={"errorCode": "Exception", "userMessage": "Internal server error"})