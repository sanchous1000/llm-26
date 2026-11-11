from __future__ import annotations

import argparse
import json
import pickle
import platform
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from preprocessing.loaders import load_documents

from config import Config, config_hash, load_config
from .embedders import EmbeddingCache, ChunkEmbeddings, build_bm25
from .splitters import embedder_token_counter, split_documents

MANIFEST = "manifest.json"


def make_run_id(config: Config, digest: str) -> str:
    slug = re.sub(r"[^\w.-]+", "-", config.embedding.model.split("/")[-1]).lower()
    s = config.splitter
    return (
        f"{config.embedding.vectorization}-{slug}-cs{s.chunk_size}"
        f"-ov{s.chunk_overlap}-{digest[:8]}"
    )


def token_stats(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values)
    return {
        "min": int(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
        "mean": round(float(arr.mean()), 1),
    }


def write_chunks(path: Path, chunks) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.metadata["chunk_id"],
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_complete(run_dir: Path, digest: str) -> bool:
    manifest = run_dir / MANIFEST
    if not manifest.is_file():
        return False
    return json.loads(manifest.read_text(encoding="utf-8")).get("config_hash") == digest


def update_latest(artifacts_dir: Path, run_id: str) -> None:
    (artifacts_dir / "latest.json").write_text(
        json.dumps({"run_id": run_id, "path": run_id}, indent=2), encoding="utf-8"
    )


def build(config: Config, rebuild: bool = False, dry_run: bool = False) -> Path | None:
    digest = config_hash(config)
    run_id = make_run_id(config, digest)
    artifacts = config.artifacts_dir
    run_dir = artifacts / run_id

    if not dry_run and not rebuild and is_complete(run_dir, digest):
        print(f"Артефакты уже собраны: {run_dir} (используйте --rebuild для пересборки)")
        update_latest(artifacts, run_id)
        return run_dir

    started = time.time()
    documents = load_documents(config.sources_dir, reparse=config.sources.reparse)
    chunks = split_documents(documents, config)

    if not chunks:
        raise ValueError("Не получено ни одного чанка")

    n_tokens = [c.metadata["n_tokens"] for c in chunks]
    stats = token_stats(n_tokens)
    emb = config.embedding
    count_embed = embedder_token_counter(config)
    embedded_tokens = [c.metadata.get("embed_tokens") or count_embed(c.page_content) for c in chunks]
    truncated = sum(t > emb.max_seq_length for t in embedded_tokens)
    
    print(f"Чанков: {len(chunks)}; токены: {stats}; будут обрезаны моделью: {truncated}")
    if dry_run:
        return None

    tmp_dir = artifacts / f".tmp-{run_id}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    texts = [c.page_content for c in chunks]
    write_chunks(tmp_dir / "chunks.jsonl", chunks)

    embedder = ChunkEmbeddings(emb)
    cache = EmbeddingCache(artifacts / "_embedding_cache", emb)
    vectors, computed = cache.embed(texts, embedder)
    np.save(tmp_dir / "embeddings.npy", vectors)
    dim = int(vectors.shape[1])
    print(f"Эмбеддинги: {vectors.shape}, посчитано заново: {computed}, device={embedder.device}")
    if emb.vectorization == "hybrid":
        bm25 = build_bm25(texts, config)
        with (tmp_dir / "bm25.pkl").open("wb") as handle:
            pickle.dump(
                {"bm25": bm25, "chunk_ids": [c.metadata["chunk_id"] for c in chunks]}, handle
            )

    manifest = {
        "run_id": run_id,
        "config_hash": digest,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "n_documents": len(documents),
        "n_chunks": len(chunks),
        "chunk_tokens": stats,
        "truncated_by_model": truncated,
        "embedding_dim": dim,
        "newly_embedded": computed,
        "seconds": round(time.time() - started, 1),
        "python": platform.python_version(),
    }
    (tmp_dir / MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    shutil.rmtree(run_dir, ignore_errors=True)
    tmp_dir.rename(run_dir)
    update_latest(artifacts, run_id)
    print(f"Готово: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Чанки и эмбеддинги (п.3)")
    parser.add_argument("--config", default=None, help="путь к config.yaml (по умолчанию lab2/config.yaml)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="переопределение, например splitter.chunk_size=300")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать, даже если артефакты есть")
    parser.add_argument("--dry-run", action="store_true", help="только чанки и статистика, без эмбеддингов")
    args = parser.parse_args()
    build(load_config(args.config, args.overrides), rebuild=args.rebuild, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
