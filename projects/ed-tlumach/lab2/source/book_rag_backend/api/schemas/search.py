from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: str
    content: str

    @field_validator("content")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("content не может быть пустым")
        return v


class SearchRequest(BaseModel):
    messages: list[Message]


class Citation(BaseModel):
    n: int
    source: str
    page: int
    heading: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    messages: list[Message]
    citations: list[Citation]


class ErrorResponse(BaseModel):
    errorCode: str
    userMessage: str