from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Вопрос пользователя")
    top_k: int | None = Field(None, ge=1, le=50, description="Сколько фрагментов вернуть (по умолчанию из конфига)")
    score_threshold: float | None = Field(
        None, ge=-1.0, le=1.0, description="Минимальный cosine score (по умолчанию из конфига)"
    )

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query не может быть пустым")
        return v.strip()


class SearchHit(BaseModel):
    text: str
    score: float
    source: str
    page_start: int
    page_end: int
    section: str


class SearchResponse(BaseModel):
    results: list[SearchHit]


class Citation(BaseModel):
    n: int = Field(..., description="Номер фрагмента, на который ссылается ответ как [n]")
    source: str
    page_start: int
    page_end: int
    section: str
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(..., description="Фрагменты, на которые сослался ответ")
    retrieved: list[Citation] = Field(..., description="Все найденные фрагменты")


class ErrorResponse(BaseModel):
    errorCode: str
    userMessage: str
