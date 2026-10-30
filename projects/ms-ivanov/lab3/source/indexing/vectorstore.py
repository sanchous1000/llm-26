"""Qdrant collection management and point upserts."""

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from source.indexing.config import PipelineConfig
from source.indexing.settings import InfraSettings

DENSE = "dense"
SPARSE = "sparse"
_DISTANCE = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}


@dataclass(frozen=True)
class VectorPoint:
    """One chunk's vectors plus the payload used for filtering and display."""

    chunk_id: str
    dense: list[float] | None
    sparse: dict[int, float] | None
    payload: dict


def make_client(settings: InfraSettings) -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_http_port,
        grpc_port=settings.qdrant_grpc_port,
        prefer_grpc=True,
    )


def ensure_collection(
    client: QdrantClient,
    config: PipelineConfig,
    dense_dim: int,
    *,
    recreate: bool = False,
) -> str:
    """Create the collection for ``config`` (or recreate it); returns its name."""
    name = config.collection_name
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if client.collection_exists(name):
        return name
    index, mode = config.index, config.embedding.mode
    dense = {
        DENSE: models.VectorParams(
            size=dense_dim,
            distance=_DISTANCE[index.distance],
            on_disk=index.on_disk,
            hnsw_config=models.HnswConfigDiff(
                m=index.hnsw_m, ef_construct=index.hnsw_ef_construct
            ),
        )
    }
    sparse = {SPARSE: models.SparseVectorParams()}
    client.create_collection(
        collection_name=name,
        vectors_config=dense if mode in ("dense", "hybrid") else {},
        sparse_vectors_config=sparse if mode in ("sparse", "hybrid") else None,
    )
    client.create_payload_index(name, "doc_key", models.PayloadSchemaType.KEYWORD)
    return name


def existing_ids(client: QdrantClient, name: str, ids: Sequence[str]) -> set[str]:
    """Which of ``ids`` are already stored (used to resume interrupted runs)."""
    if not ids or not client.collection_exists(name):
        return set()
    found = client.retrieve(name, ids=list(ids), with_payload=False, with_vectors=False)
    return {str(point.id) for point in found}


def upsert(
    client: QdrantClient, name: str, points: Iterable[VectorPoint], mode: str
) -> int:
    batch = [_to_point(point, mode) for point in points]
    if batch:
        client.upsert(name, points=batch, wait=True)
    return len(batch)


def _to_point(point: VectorPoint, mode: str) -> models.PointStruct:
    vector: dict = {}
    if mode in ("dense", "hybrid") and point.dense is not None:
        vector[DENSE] = point.dense
    if mode in ("sparse", "hybrid") and point.sparse is not None:
        items = sorted(point.sparse.items())
        vector[SPARSE] = models.SparseVector(
            indices=[token for token, _ in items],
            values=[weight for _, weight in items],
        )
    return models.PointStruct(id=point.chunk_id, vector=vector, payload=point.payload)


def apply_index_config(client: QdrantClient, config: PipelineConfig) -> None:
    """Change the HNSW graph of an existing collection in place (Qdrant rebuilds it).

    The stored vectors are untouched, so no embedding is recomputed.
    """
    index = config.index
    client.update_collection(
        config.collection_name,
        vectors_config={
            DENSE: models.VectorParamsDiff(
                hnsw_config=models.HnswConfigDiff(
                    m=index.hnsw_m, ef_construct=index.hnsw_ef_construct
                ),
                on_disk=index.on_disk,
            )
        },
    )


def wait_until_indexed(client: QdrantClient, name: str, timeout_s: float = 600) -> None:
    """Block until the optimizer is done (collection green, all vectors indexed)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = client.get_collection(name)
        done = (info.indexed_vectors_count or 0) >= (info.points_count or 0)
        if info.status == models.CollectionStatus.GREEN and done:
            return
        time.sleep(0.2)
    raise TimeoutError(f"collection {name} was not indexed within {timeout_s:.0f}s")


def search_params(config: PipelineConfig) -> models.SearchParams:
    """Query-time parameters derived from the configuration."""
    return models.SearchParams(hnsw_ef=config.index.hnsw_ef)


def count(client: QdrantClient, name: str) -> int:
    return client.count(name, exact=True).count if client.collection_exists(name) else 0
