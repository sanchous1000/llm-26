from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
from langchain_core.documents import Document

from config import Config, load_config

from .embedders import ChunkEmbeddings, bm25_tokenize

MANIFEST = "manifest.json"
INDEX_FILE = "index.faiss"
DOCSTORE_FILE = "docstore.jsonl"
ACTIVE_FILE = "active.json"
RRF_K = 60

_QUANTIZERS = {
    "fp16": faiss.ScalarQuantizer.QT_fp16,
    "sq8": faiss.ScalarQuantizer.QT_8bit,
}


# ---------- имена и пути ----------

def resolve_run_id(config: Config, run_id: str | None) -> str:
    if run_id:
        return run_id
    latest = config.artifacts_dir / "latest.json"
    if not latest.is_file():
        raise FileNotFoundError("Нет artifacts/latest.json: сначала запустите python -m index.build_index")
    return json.loads(latest.read_text(encoding="utf-8"))["run_id"]


def collection_name(config: Config, run_id: str) -> str:
    i = config.index
    if i.type == "flat":
        return f"{run_id}__flat"
    suffix = "" if i.quantization == "none" else f"-{i.quantization}"
    return f"{run_id}__hnsw-M{i.hnsw.M}-efc{i.hnsw.ef_construction}{suffix}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_hash(config: Config, embeddings_sha: str) -> str:
    i = config.index
    payload = {
        "type": i.type,
        "metric": "cosine",  # ключ остаётся, чтобы уже собранные коллекции не пересобирались
        "M": i.hnsw.M,
        "ef_construction": i.hnsw.ef_construction,
        "quantization": i.quantization,
        "embeddings_sha256": embeddings_sha,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def read_active(store_dir: Path) -> str | None:
    path = store_dir / ACTIVE_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("collection")


def write_active(store_dir: Path, name: str) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / ACTIVE_FILE).write_text(json.dumps({"collection": name}, indent=2), encoding="utf-8")


# ---------- построение ----------

def make_index(config: Config, vectors: np.ndarray) -> faiss.Index:
    i = config.index
    dim = vectors.shape[1]
    if i.type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif i.quantization == "none":
        index = faiss.IndexHNSWFlat(dim, i.hnsw.M, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexHNSWSQ(dim, _QUANTIZERS[i.quantization], i.hnsw.M, faiss.METRIC_INNER_PRODUCT)
    if i.type == "hnsw":
        index.hnsw.efConstruction = i.hnsw.ef_construction
    if not index.is_trained:
        index.train(vectors)
    index.add(vectors)
    return index


def set_ef_search(index: faiss.Index, config: Config) -> None:
    if config.index.type == "hnsw":
        faiss.downcast_index(index).hnsw.efSearch = config.index.hnsw.ef_search


def is_complete(directory: Path, digest: str) -> bool:
    manifest = directory / MANIFEST
    if not manifest.is_file() or not (directory / INDEX_FILE).is_file():
        return False
    return json.loads(manifest.read_text(encoding="utf-8")).get("collection_hash") == digest


def build_collection(config: Config, run_id: str, rebuild: bool = False, activate: bool = True) -> Path:
    run_dir = config.artifacts_dir / run_id
    emb_path, chunks_path = run_dir / "embeddings.npy", run_dir / "chunks.jsonl"
    if not emb_path.is_file() or not chunks_path.is_file():
        raise FileNotFoundError(
            f"В {run_dir} нет embeddings.npy/chunks.jsonl: нужен dense или hybrid запуск п.3"
        )

    store_dir = config.faiss_dir
    name = collection_name(config, run_id)
    target = store_dir / name
    digest = collection_hash(config, file_sha256(emb_path))

    if not rebuild and is_complete(target, digest):
        print(f"Коллекция уже построена: {target} (--rebuild для пересборки)")
        if activate:
            write_active(store_dir, name)
        return target

    started = time.time()
    vectors = np.ascontiguousarray(np.load(emb_path), dtype=np.float32)
    lines = chunks_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(vectors):
        raise ValueError(f"Число чанков ({len(lines)}) не совпадает с числом векторов ({len(vectors)})")

    index = make_index(config, vectors)
    if index.ntotal != len(vectors):
        raise RuntimeError(f"ntotal={index.ntotal}, ожидалось {len(vectors)}")

    set_ef_search(index, config)
    build_seconds = round(time.time() - started, 2)

    tmp = store_dir / f".tmp-{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    faiss.write_index(index, str(tmp / INDEX_FILE))
    with (tmp / DOCSTORE_FILE).open("w", encoding="utf-8") as handle:
        for faiss_id, line in enumerate(lines):
            record = json.loads(line)
            handle.write(json.dumps({"faiss_id": faiss_id, **record}, ensure_ascii=False) + "\n")
    manifest = {
        "collection": name,
        "collection_hash": digest,
        "run_id": run_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "index": config.to_dict()["index"],
        "ntotal": int(index.ntotal),
        "dim": int(vectors.shape[1]),
        "build_seconds": build_seconds,
        "index_bytes": (tmp / INDEX_FILE).stat().st_size,
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    if activate:
        write_active(store_dir, name)
    print(
        f"Коллекция {name}: ntotal={manifest['ntotal']}, dim={manifest['dim']}, "
        f"{manifest['index_bytes'] / 1024:.0f} КБ, {build_seconds} с"
    )
    return target


# ---------- сброс и список ----------

def reset(config: Config, run_id: str | None, everything: bool) -> None:
    store_dir = config.faiss_dir
    if everything:
        shutil.rmtree(store_dir, ignore_errors=True)
        print(f"Хранилище удалено: {store_dir}")
        return
    name = collection_name(config, resolve_run_id(config, run_id))
    shutil.rmtree(store_dir / name, ignore_errors=True)
    if read_active(store_dir) == name:
        (store_dir / ACTIVE_FILE).unlink(missing_ok=True)
    print(f"Коллекция удалена: {name}")


def list_collections(config: Config) -> None:
    store_dir = config.faiss_dir
    active = read_active(store_dir)
    manifests = sorted(store_dir.glob(f"*/{MANIFEST}")) if store_dir.is_dir() else []
    if not manifests:
        print("Коллекций нет")
    for path in manifests:
        m = json.loads(path.read_text(encoding="utf-8"))
        mark = "*" if m["collection"] == active else " "
        print(
            f"{mark} {m['collection']}  ntotal={m['ntotal']}  {m['index_bytes'] / 1024:.0f} КБ"
        )


# ---------- единый поиск (dense + hybrid BM25/RRF) ----------

def load_bm25(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)["bm25"]


def rrf_scores(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    scores = rrf_scores(rankings, k)
    return sorted(scores, key=lambda idx: (-scores[idx], idx))


def dense_depth(index_type: str, ef_search: int | None, ntotal: int, min_k: int = 10) -> int:
    """Глубина поиска FAISS: для HNSW ограничена max(min_k, ef_search), чтобы не завышать efSearch."""
    if index_type == "flat" or ef_search is None:
        return ntotal
    return min(ntotal, max(min_k, ef_search))


def dense_rankings(index: faiss.Index, queries: np.ndarray, k: int) -> list[list[int]]:
    _, ids = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    rankings: list[list[int]] = []
    for row in ids:
        ranking: list[int] = []
        seen: set[int] = set()
        for idx in row.tolist():
            if idx < 0 or idx in seen:
                continue
            seen.add(idx)
            ranking.append(idx)
        rankings.append(ranking)
    return rankings


def bm25_rankings(bm25: Any, questions: list[str], n_chunks: int) -> list[list[int]]:
    rankings = []
    for text in questions:
        scores = np.asarray(bm25.get_scores(bm25_tokenize(text)), dtype=np.float64)
        if len(scores) != n_chunks:
            raise ValueError(f"BM25 вернул {len(scores)} оценок на {n_chunks} чанков")
        positive = np.flatnonzero(scores > 0)
        order = positive[np.argsort(-scores[positive], kind="mergesort")]
        rankings.append(order.tolist())
    return rankings


def fuse_rankings(
    retrieval: str,
    dense: list[list[int]],
    sparse: list[list[int]] | None,
    rrf_k: int = RRF_K,
) -> list[list[int]]:
    if retrieval == "dense":
        return dense
    if sparse is None:
        raise ValueError("Для hybrid нужен ранжир BM25")
    return [rrf_fuse([left, right], rrf_k) for left, right in zip(dense, sparse)]


def rank_indices(
    index: faiss.Index,
    queries: np.ndarray,
    questions: list[str],
    bm25: Any | None,
    depth: int,
    rrf_k: int = RRF_K,
) -> tuple[list[list[int]], list[dict[int, float]]]:
    """Порядок чанков и очки. Для dense очки пустые: их отдаёт inner product в VectorStore."""
    dense = dense_rankings(index, queries, depth)
    if bm25 is None:
        return dense, [{} for _ in dense]
    sparse = bm25_rankings(bm25, questions, index.ntotal)
    orders = fuse_rankings("hybrid", dense, sparse, rrf_k)
    score_maps = [rrf_scores([left, right], rrf_k) for left, right in zip(dense, sparse)]
    return orders, score_maps


@dataclass
class VectorStore:
    """Единый поисковый индекс поверх FAISS и (для hybrid-коллекций) BM25 с RRF."""

    index: faiss.Index
    docs: list[Document]
    embeddings: ChunkEmbeddings
    bm25: Any | None = None
    index_type: str = "hnsw"
    ef_search: int | None = None
    rrf_k: int = RRF_K

    def similarity_search_with_score(self, question: str, k: int = 5) -> list[tuple[Document, float]]:
        query = self.embeddings.embed_queries([question])
        if self.bm25 is None:
            scores, ids = self.index.search(np.ascontiguousarray(query, dtype=np.float32), k)
            return [
                (self.docs[idx], float(score))
                for score, idx in zip(scores[0].tolist(), ids[0].tolist())
                if 0 <= idx < len(self.docs)
            ]
        depth = dense_depth(self.index_type, self.ef_search, len(self.docs), min_k=k)
        orders, score_maps = rank_indices(
            self.index, query, [question], self.bm25, depth, self.rrf_k
        )
        return [(self.docs[idx], score_maps[0][idx]) for idx in orders[0][:k]]


def load_vector_store(
    config: Config,
    collection: str | None = None,
    embeddings: ChunkEmbeddings | None = None,
) -> VectorStore:
    """Загружает FAISS-коллекцию (и BM25 при hybrid-прогоне); ef_search берётся из конфига."""
    store_dir = config.faiss_dir
    collection = collection or read_active(store_dir)
    if not collection or not (store_dir / collection / INDEX_FILE).is_file():
        raise FileNotFoundError("Нет активной коллекции: запустите python -m index.load_to_vector_store")
    directory = store_dir / collection

    index = faiss.read_index(str(directory / INDEX_FILE))
    set_ef_search(index, config)

    by_id: dict[int, Document] = {}
    with (directory / DOCSTORE_FILE).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            by_id[int(record["faiss_id"])] = Document(
                page_content=record["text"], metadata=record["metadata"]
            )
    docs = [by_id[i] for i in range(len(by_id))]

    bm25 = None
    index_type = config.index.type
    manifest_path = directory / MANIFEST
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_type = manifest.get("index", {}).get("type", index_type)
        run_id = manifest.get("run_id")
        if run_id:
            bm25_path = config.artifacts_dir / run_id / "bm25.pkl"
            if bm25_path.is_file():
                bm25 = load_bm25(bm25_path)

    if embeddings is None:
        embeddings = ChunkEmbeddings(config.embedding)
    return VectorStore(
        index=index,
        docs=docs,
        embeddings=embeddings,
        bm25=bm25,
        index_type=index_type,
        ef_search=config.index.hnsw.ef_search if index_type == "hnsw" else None,
        rrf_k=config.rag.rrf_k,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка индекса в FAISS (п.4)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-id", default=None, help="запуск п.3 (по умолчанию из latest.json)")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать коллекцию")
    parser.add_argument("--reset", action="store_true", help="удалить коллекцию (с --all всё хранилище)")
    parser.add_argument("--all", action="store_true", help="вместе с --reset: удалить все коллекции")
    parser.add_argument("--list", action="store_true", help="показать коллекции")
    args = parser.parse_args()

    try:
        config = load_config(args.config, args.overrides)
        if args.list:
            list_collections(config)
        elif args.reset:
            reset(config, args.run_id, args.all)
        else:
            build_collection(config, resolve_run_id(config, args.run_id), rebuild=args.rebuild)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
