"""FastAPI-сервис RAG-агента.

Запуск из каталога lab2:
    python source/api.py                       # http://127.0.0.1:8000/docs
    RAG_SET="search.mode=dense" python source/api.py
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import settings
from rag import RAGEngine

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    overrides = [s for s in os.getenv("RAG_SET", "").split(";") if s.strip()]
    cfg = settings.load_config(os.getenv("RAG_CONFIG") or None, overrides)
    state["cfg"] = cfg
    state["engine"] = RAGEngine(cfg)
    yield
    state.clear()


app = FastAPI(title="RAG по материалам НИР", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Какие критерии приёмки заданы в ТЗ?"])
    top_k: int | None = Field(None, ge=1, le=20)
    mode: Literal["dense", "sparse", "hybrid"] | None = None


class Citation(BaseModel):
    n: int
    cited: bool
    source: str
    path: str | None
    section: str
    page: int | None
    page_mode: str | None
    locator: str
    snippet: str
    score: float
    chunk_id: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[Citation]
    mode: str
    collection: str
    model: str
    timings: dict
    tokens: dict | None


@app.get("/health")
def health():
    engine: RAGEngine = state["engine"]
    try:
        points = engine.retriever.client.count(engine.retriever.collection).count
    except Exception as e:
        raise HTTPException(503, f"Qdrant недоступен: {e}")
    return {"status": "ok", "collection": engine.retriever.collection, "points": points,
            "llm": state["cfg"]["llm"]["model"]}


@app.get("/config")
def config():
    cfg = state["cfg"]
    return {"index": settings.index_params(cfg), "collection": settings.collection_name(cfg),
            "search": cfg["search"], "llm": cfg["llm"]}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return state["engine"].ask(req.question, top_k=req.top_k, mode=req.mode)
    except ValueError as e:
        raise HTTPException(400, str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
