import uuid
from datetime import datetime, timezone
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from doc_rag.config import QdrantConfig
from doc_rag.services.splitter import Chunk

POINT_NAMESPACE = uuid.UUID("6f1c1c1e-5b5e-4d3a-9a55-2f0a5b6d7e11")
UPSERT_BATCH = 128


def point_id(source: str, chunk_index: int) -> str:
    """Детерминированный ID: повторная загрузка того же чанка перезаписывает точку."""
    return str(uuid.uuid5(POINT_NAMESPACE, f"{source}:{chunk_index}"))


def _source_filter(source: str) -> Filter:
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])


class QdrantService:
    """Работа с одной коллекцией (имя определяется конфигурацией индекса)."""

    def __init__(self, config: QdrantConfig, collection: str):
        self.collection = collection
        self.client = QdrantClient(url=config.url, api_key=config.api_key, timeout=60, check_compatibility=False)

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------ коллекция

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, dim: int) -> bool:
        """Создаёт коллекцию (cosine, HNSW по умолчанию), если её нет. True - если создана."""
        if self.collection_exists():
            return False
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        return True

    def drop_collection(self) -> None:
        self.client.delete_collection(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    # ------------------------------------------------------------------ документы

    def document_exists(self, source: str) -> bool:
        return self.client.count(self.collection, count_filter=_source_filter(source), exact=True).count > 0

    def delete_document(self, source: str) -> None:
        self.client.delete(self.collection, points_selector=_source_filter(source), wait=True)

    def list_documents(self) -> list[str]:
        sources: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, with_payload=["source"], with_vectors=False, limit=1000, offset=offset
            )
            sources.update(p.payload["source"] for p in points)
            if offset is None:
                return sorted(sources)

    # ------------------------------------------------------------------ загрузка

    def upsert_chunks(
        self,
        source: str,
        path: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        index_config: dict,
    ) -> int:
        indexed_at = datetime.now(timezone.utc).isoformat()
        points = [
            PointStruct(
                id=point_id(source, c.chunk_index),
                vector=vec,
                payload={
                    "text": c.text,
                    "source": source,
                    "path": path,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "h1": c.h1,
                    "h2": c.h2,
                    "h3": c.h3,
                    "chunk_index": c.chunk_index,
                    "indexed_at": indexed_at,
                    "index_config": index_config,
                },
            )
            for c, vec in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), UPSERT_BATCH):
            self.client.upsert(self.collection, points=points[i:i + UPSERT_BATCH], wait=True)
        return len(points)

    # ------------------------------------------------------------------ поиск

    def search(self, vector: list[float], top_k: int, score_threshold: float | None = None) -> list[dict]:
        res = self.client.query_points(
            self.collection,
            query=vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return [{"score": p.score, **(p.payload or {})} for p in res.points]
