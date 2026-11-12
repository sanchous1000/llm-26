"""Поиск по коллекции Qdrant: dense, sparse (BM25) и гибрид (RRF)."""
from __future__ import annotations

import json
import time

from qdrant_client import models

import build_index
import settings
import vector_store as vs
from embeddings import DenseEncoder
from sparse import BM25Encoder


class Retriever:
    def __init__(self, cfg: dict, client=None):
        self.cfg = cfg
        self.client = client or vs.connect(cfg)
        self.collection = settings.collection_name(cfg)
        if not self.client.collection_exists(self.collection):
            raise RuntimeError(
                f"Коллекция {self.collection} не найдена. Сначала: python source/load_to_vector_store.py "
                "с теми же параметрами индекса.")
        meta = json.loads((build_index.index_dir(cfg) / "index.json").read_text(encoding="utf-8"))
        self.vector_type = meta["params"]["vector_type"]
        e = cfg["embeddings"]
        self.dense = DenseEncoder(e["dense_model"], e["base_url"]) if self.vector_type != "sparse" else None
        bm = meta["stats"].get("bm25") or {}
        self.bm25 = BM25Encoder(bm.get("k1", 1.2), bm.get("b", 0.75), bm.get("avgdl")) \
            if self.vector_type != "dense" else None

    def search(self, query: str, top_k: int | None = None, mode: str | None = None,
               exact: bool = False) -> tuple[list[dict], dict]:
        s = self.cfg["search"]
        top_k = top_k or s["top_k"]
        mode = mode or s["mode"]
        if mode != "sparse" and self.dense is None or mode != "dense" and self.bm25 is None:
            raise ValueError(f"режим {mode} недоступен для индекса с vector_type={self.vector_type}")
        params = models.SearchParams(hnsw_ef=s.get("ef_search", 64), exact=exact)
        timings = {}

        t0 = time.perf_counter()
        qd = self.dense.encode_query(query).tolist() if mode in ("dense", "hybrid") else None
        timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        qs = None
        if mode in ("sparse", "hybrid"):
            idx, val = self.bm25.encode_query(query)
            qs = models.SparseVector(indices=idx, values=val)

        t1 = time.perf_counter()
        if mode == "dense":
            res = self.client.query_points(self.collection, query=qd, using=vs.DENSE, limit=top_k,
                                           search_params=params, with_payload=True)
        elif mode == "sparse":
            res = self.client.query_points(self.collection, query=qs, using=vs.SPARSE, limit=top_k,
                                           with_payload=True)
        elif mode == "hybrid":
            pk = max(s.get("prefetch_k", 30), top_k)
            res = self.client.query_points(
                self.collection,
                prefetch=[models.Prefetch(query=qd, using=vs.DENSE, limit=pk, params=params),
                          models.Prefetch(query=qs, using=vs.SPARSE, limit=pk)],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k, with_payload=True)
        else:
            raise ValueError(f"неизвестный режим поиска: {mode}")
        timings["search_ms"] = round((time.perf_counter() - t1) * 1000, 1)

        hits = [{**p.payload, "score": round(float(p.score), 4), "rank": r}
                for r, p in enumerate(res.points, 1)]
        return hits, timings
