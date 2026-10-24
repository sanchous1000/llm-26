"""Шаг 4. Загружает чанки и эмбеддинги из artifacts/ в Qdrant.

Идемпотентно: если коллекция уже существует и число точек совпадает с числом чанков —
ничего не делает. --rebuild удаляет коллекцию и грузит заново. --reset просто удаляет коллекцию.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # чтобы работал import book_rag_backend

from book_rag_backend.config import Config, add_common_args, cfg_from_args, path_of
from book_rag_backend.services.qdrant import QdrantService, get_qdrant_client


def read_artifacts(art: Path):
    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in (art / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    dense = np.load(art / "dense.npy").tolist() if (art / "dense.npy").exists() else None
    sparse = None
    if (art / "sparse.jsonl").exists():
        rows = [json.loads(line) for line in (art / "sparse.jsonl").read_text(encoding="utf-8").splitlines()]
        sparse = [(r["i"], r["v"]) for r in rows]
    if dense is not None and len(dense) != len(chunks):
        raise SystemExit(f"dense.npy ({len(dense)} векторов) не совпадает по числу с chunks.jsonl "
                          f"({len(chunks)}) — пересоберите индекс: python scripts/build_index.py --force")
    if sparse is not None and len(sparse) != len(chunks):
        raise SystemExit(f"sparse.jsonl ({len(sparse)} векторов) не совпадает по числу с chunks.jsonl "
                          f"({len(chunks)}) — пересоберите индекс: python scripts/build_index.py --force")
    return manifest, chunks, dense, sparse


async def load(cfg: Config, rebuild: bool = False, reset: bool = False, client=None, log=print) -> QdrantService:
    art = path_of(cfg, "corpus", "artifacts_dir")
    if not (art / "manifest.json").exists():
        raise SystemExit("Артефактов нет. Сначала запустите scripts/build_index.py")
    manifest, chunks, dense, sparse = read_artifacts(art)

    client = client or get_qdrant_client(cfg.qdrant.url, cfg.qdrant.api_key)
    store = QdrantService(client, cfg.qdrant.collection_name)

    if reset:
        await store.drop()
        log(f"Коллекция {store.collection} удалена")
        return store

    if rebuild:
        await store.drop()

    if await store.exists() and await store.count() == len(chunks):
        log(f"Коллекция {store.collection} уже загружена ({len(chunks)} точек), пропускаю")
        return store
    if await store.exists():
        await store.drop()  # неполная загрузка — пересоздаём

    await store.create(manifest["params"]["vectorizer"], manifest["dense_dim"],
                       cfg.qdrant.hnsw_m, cfg.qdrant.hnsw_ef_construct)
    await store.upsert_chunks(chunks, dense, sparse)
    log(f"Загружено {await store.count()} точек в коллекцию {store.collection}")
    return store


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    async def _main():
        store = await load(cfg_from_args(args), rebuild=args.rebuild, reset=args.reset)
        await store.client.close()

    asyncio.run(_main())