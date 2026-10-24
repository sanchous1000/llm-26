from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger

from book_rag_backend.api.routes import documents_router, search_router
from book_rag_backend.config import load_config
from book_rag_backend.services.embeddings import EmbeddingsService, SparseEmbeddingsService
from book_rag_backend.services.llm import LLMService
from book_rag_backend.services.qdrant import QdrantService, get_qdrant_client
from book_rag_backend.services.rag import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    client = get_qdrant_client(cfg.qdrant.url, cfg.qdrant.api_key)
    qdrant = QdrantService(client, cfg.qdrant.collection_name)
    if not await qdrant.exists():
        raise SystemExit(f"Коллекции '{cfg.qdrant.collection_name}' нет. "
                          f"Сначала запустите scripts/build_index.py и scripts/load_to_vector_store.py")

    vz = cfg.embeddings.vectorizer

    embeddings = EmbeddingsService.from_name(cfg.embeddings.dense_model)
    sparse = SparseEmbeddingsService.from_name() if vz in ("sparse", "hybrid") else None
    llm = LLMService(cfg.model.url, cfg.model.api_key, cfg.model.model,
                      max_tokens=cfg.model.max_tokens)

    rag = RAGService(qdrant, vz, embeddings, sparse, llm, top_k=cfg.retrieval.top_k,
                     ef_search=cfg.qdrant.hnsw_ef_search, max_context_chars=cfg.model.max_context_chars)

    app.state.config = cfg
    app.state.rag = rag
    n = await qdrant.count()
    logger.info(f"✓ Коллекция '{cfg.qdrant.collection_name}': {n} чанков, поиск {vz}")

    yield

    await llm.close()
    await client.close()


app = FastAPI(title="Lab2 RAG API", lifespan=lifespan)
app.include_router(search_router)
app.include_router(documents_router)


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)