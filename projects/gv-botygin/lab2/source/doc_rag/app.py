from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from doc_rag.api.routes import router
from doc_rag.config import load_config
from doc_rag.services.embeddings import EmbeddingsService
from doc_rag.services.llm import LLMService
from doc_rag.services.qdrant import QdrantService
from doc_rag.services.rag import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()  # путь к конфигу: переменная DOC_RAG_CONFIG
    collection = config.collection_name()

    qdrant = QdrantService(config.qdrant, collection)
    if not qdrant.collection_exists():
        qdrant.close()
        raise RuntimeError(
            f"Коллекция '{collection}' не найдена в Qdrant. Сначала запустите scripts/build_index.py "
            f"с теми же параметрами splitter/embeddings."
        )

    embeddings = EmbeddingsService(config.embeddings)
    llm = LLMService(config.llm)
    app.state.config = config
    app.state.rag = RAGService(qdrant, embeddings, llm, config.search)
    logger.info(f"Готово: коллекция {collection}, {qdrant.count()} точек, LLM {config.llm.model}")

    yield

    llm.close()
    qdrant.close()


app = FastAPI(
    title="Doc RAG API",
    description="Поиск по PDF-документам (dense, Qdrant) и ответы LLM с цитатами",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
