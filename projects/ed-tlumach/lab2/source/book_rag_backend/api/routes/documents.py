from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from book_rag_backend.api.dependencies import get_document_service
from book_rag_backend.api.schemas.documents import DocumentShort, DocumentsListResponse, DocumentUploadResponse
from book_rag_backend.services.documents import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
        file: UploadFile = File(...),
        chunk_size: int | None = Form(None),
        chunk_overlap: int | None = Form(None),
        include_headings: bool | None = Form(None),
        service: DocumentService = Depends(get_document_service),
):
    try:
        result = await service.add_document(file, chunk_size, chunk_overlap, include_headings)
        return DocumentUploadResponse(**result)
    except TypeError:
        return JSONResponse(status_code=415, content={"errorCode": "NotSupportedFormat", "userMessage": "Поддерживается только PDF"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"errorCode": "ValidationError", "userMessage": str(e)})
    except Exception:
        logger.exception("Ошибка в POST /documents")
        return JSONResponse(status_code=500, content={"errorCode": "Exception", "userMessage": "Internal server error"})


@router.get("", response_model=DocumentsListResponse)
async def list_documents(service: DocumentService = Depends(get_document_service)):
    docs = await service.list_documents()
    return DocumentsListResponse(documents=[DocumentShort(**d) for d in docs])


@router.delete("/{source:path}", status_code=204)
async def delete_document(source: str, service: DocumentService = Depends(get_document_service)):
    try:
        await service.delete_document(source)
    except ValueError:
        return JSONResponse(status_code=404, content={"errorCode": "NotFound", "userMessage": "Document not found"})