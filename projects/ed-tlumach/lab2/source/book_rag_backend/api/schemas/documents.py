from pydantic import BaseModel


class ErrorResponse(BaseModel):
    errorCode: str
    userMessage: str


class DocumentShort(BaseModel):
    source: str
    date: str
    n_pages: int
    n_chunks: int


class DocumentsListResponse(BaseModel):
    documents: list[DocumentShort]


class DocumentUploadResponse(BaseModel):
    source: str
    n_pages: int
    n_chunks: int
