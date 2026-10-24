import uuid

from qdrant_client import AsyncQdrantClient, models


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


def get_qdrant_client(url: str, api_key: str | None) -> AsyncQdrantClient:
    kwargs = {"url": url, "timeout": 60}
    if api_key:
        kwargs["api_key"] = api_key
    return AsyncQdrantClient(**kwargs)


class QdrantService:
    def __init__(self, client: AsyncQdrantClient, collection: str):
        self.client = client
        self.collection = collection

    async def exists(self) -> bool:
        return await self.client.collection_exists(self.collection)

    async def count(self) -> int:
        return (await self.client.count(self.collection)).count

    async def drop(self) -> None:
        if await self.exists():
            await self.client.delete_collection(self.collection)

    async def create(self, vectorizer: str, dense_dim: int | None, hnsw_m: int, hnsw_ef_construct: int) -> None:
        hnsw = models.HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct)
        vectors_config = {}
        sparse_config = {}
        if vectorizer in ("dense", "hybrid"):
            vectors_config["dense"] = models.VectorParams(size=dense_dim, distance=models.Distance.COSINE, hnsw_config=hnsw)
        if vectorizer in ("sparse", "hybrid"):
            sparse_config["sparse"] = models.SparseVectorParams(modifier=models.Modifier.IDF)
        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config=vectors_config or None,
            sparse_vectors_config=sparse_config or None,
        )

    async def upsert_chunks(self, chunks: list[dict], dense: list[list[float]] | None,
                             sparse: list[tuple[list[int], list[float]]] | None, batch: int = 128) -> None:
        points = []
        for i, c in enumerate(chunks):
            vector = {}
            if dense is not None:
                vector["dense"] = dense[i]
            if sparse is not None:
                idx, val = sparse[i]
                vector["sparse"] = models.SparseVector(indices=idx, values=val)
            points.append(models.PointStruct(
                id=point_id(c["id"]), vector=vector,
                payload={"content": c["text"], "source": c["source"], "page": c["page"],
                         "date": c["date"], "heading": c.get("heading", "")},
            ))
        for i in range(0, len(points), batch):
            await self.client.upsert(collection_name=self.collection, points=points[i:i + batch], wait=True)

    async def search(self, dense_vector: list[float] | None, sparse_vector: tuple[list[int], list[float]] | None,
                      vectorizer: str, k: int, ef_search: int) -> list[dict]:
        params = models.SearchParams(hnsw_ef=ef_search)
        if vectorizer == "dense":
            res = await self.client.query_points(self.collection, query=dense_vector, using="dense",
                                                  limit=k, with_payload=True, search_params=params)
        elif vectorizer == "sparse":
            idx, val = sparse_vector
            res = await self.client.query_points(self.collection, query=models.SparseVector(indices=idx, values=val),
                                                  using="sparse", limit=k, with_payload=True)
        else:  # hybrid: RRF-слияние dense и sparse
            idx, val = sparse_vector
            res = await self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dense_vector, using="dense", limit=k, params=params),
                    models.Prefetch(query=models.SparseVector(indices=idx, values=val), using="sparse", limit=k),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k, with_payload=True,
            )
        return [{"content": p.payload["content"], "score": p.score,
                 "metadata": {k: v for k, v in p.payload.items() if k != "content"}}
                for p in res.points]

    async def list_sources(self) -> list[dict]:
        if not await self.exists():
            return []
        by_source: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = await self.client.scroll(self.collection, with_payload=True, with_vectors=False,
                                                       limit=512, offset=offset)
            for p in points:
                src = p.payload.get("source")
                entry = by_source.setdefault(src, {"source": src, "date": p.payload.get("date", ""),
                                                    "pages": set(), "n_chunks": 0})
                entry["n_chunks"] += 1
                entry["pages"].add(p.payload.get("page"))
            if offset is None:
                break
        return [{"source": v["source"], "date": v["date"], "n_pages": len(v["pages"]), "n_chunks": v["n_chunks"]}
                for v in by_source.values()]

    async def delete_source(self, source: str) -> bool:
        flt = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
        if (await self.client.count(self.collection, count_filter=flt)).count == 0:
            return False
        await self.client.delete(collection_name=self.collection, points_selector=flt, wait=True)
        return True