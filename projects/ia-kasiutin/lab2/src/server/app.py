from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import load_config

from .engine import RagEngine, build_engine
from .llm import LlmUnavailable, PromptTooLong, list_model_ids, resolve_base_url

state: dict[str, RagEngine] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    config = load_config(os.environ.get("LAB2_CONFIG"))
    state["engine"] = build_engine(config)
    yield
    state.clear()


app = FastAPI(title="Lab2 RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_n: int | None = Field(default=None, ge=1, le=20)


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    engine = state["engine"]
    try:
        return engine.ask(request.question, request.top_n)
    except PromptTooLong as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except LlmUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/health")
def health() -> dict:
    engine = state["engine"]
    config = engine.config
    result: dict = {
        "index": {
            "collection": config.rag.collection or "active",
            "ntotal": int(engine.store.index.ntotal),
        },
        "llm": {"base_url": resolve_base_url(config.llm), "reachable": False},
    }
    try:
        result["llm"]["models"] = list_model_ids(config.llm)
        result["llm"]["reachable"] = True
        result["llm"]["selected"] = engine.model_id
    except LlmUnavailable as error:
        result["llm"]["error"] = str(error)
    return result
