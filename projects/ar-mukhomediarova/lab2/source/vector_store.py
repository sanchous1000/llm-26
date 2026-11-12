"""Работа с Qdrant: подключение, создание коллекции, загрузка точек."""
from __future__ import annotations

import atexit
import time
import uuid

from qdrant_client import QdrantClient, models

DENSE, SPARSE = "dense", "bm25"


_CLIENTS: dict[str, QdrantClient] = {}


@atexit.register
def _close_clients() -> None:
    for c in _CLIENTS.values():
        try:
            c.close()
        except Exception:
            pass
    _CLIENTS.clear()


def connect(cfg: dict) -> QdrantClient:
    """Один клиент на адрес (встроенный режим не допускает двух клиентов на один каталог)."""
    url = cfg["vector_store"]["url"]
    if url not in _CLIENTS:
        if url.startswith("path:"):  # встроенный режим без Docker (HNSW-параметры не применяются)
            import settings
            _CLIENTS[url] = QdrantClient(path=str(settings.LAB_DIR / url[5:]))
        else:
            _CLIENTS[url] = QdrantClient(url=url, timeout=60)
    return _CLIENTS[url]


def create_collection(client: QdrantClient, name: str, cfg: dict, dense_dim: int | None, has_sparse: bool):
    vs = cfg["vector_store"]
    h = vs["hnsw"]
    vectors = {}
    if dense_dim:
        vectors[DENSE] = models.VectorParams(
            size=dense_dim,
            distance=getattr(models.Distance, vs.get("distance", "Cosine").upper()),
            hnsw_config=models.HnswConfigDiff(m=h["m"], ef_construct=h["ef_construct"],
                                              full_scan_threshold=max(10, h.get("full_scan_threshold", 10000))),
            on_disk=False,
        )
    sparse = {SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)} if has_sparse else None
    quant = None
    if vs.get("quantization") == "scalar":
        quant = models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, always_ram=True))
    client.create_collection(
        collection_name=name,
        vectors_config=vectors,
        sparse_vectors_config=sparse,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=max(10, vs.get("indexing_threshold", 20000))),
        quantization_config=quant,
    )


def point_id(chunk_id: str) -> str:
    """Детерминированный id: повторная загрузка перезаписывает те же точки, а не дублирует их."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


PAYLOAD_KEYS = ("chunk_id", "doc_id", "title", "text", "heading_path", "sections", "page_start",
                "page_end", "n_tokens", "source_type", "format", "path", "url", "date", "semester",
                "page_mode")


def upsert_index(client: QdrantClient, name: str, index: dict, batch: int = 64) -> None:
    chunks = index["chunks"]
    dense = index.get("dense")
    sparse = index.get("sparse")
    for s in range(0, len(chunks), batch):
        points = []
        for i in range(s, min(s + batch, len(chunks))):
            ch = chunks[i]
            vec = {}
            if dense is not None:
                vec[DENSE] = dense[i].tolist()
            if sparse is not None and sparse[i]["indices"]:
                vec[SPARSE] = models.SparseVector(**sparse[i])
            points.append(models.PointStruct(
                id=point_id(ch["chunk_id"]), vector=vec,
                payload={k: ch.get(k) for k in PAYLOAD_KEYS}))
        client.upsert(collection_name=name, points=points, wait=True)
        print(f"\r  upsert: {min(s + batch, len(chunks))}/{len(chunks)}", end="", flush=True)
    print()


def wait_indexed(client: QdrantClient, name: str, timeout: float = 120) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        info = client.get_collection(name)
        status = str(getattr(info.status, "value", info.status))
        if status == "green":
            return status
        time.sleep(0.5)
    return status
