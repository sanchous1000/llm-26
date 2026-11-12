"""Шаг 3. Построение чанков и эмбеддингов (dense / sparse / hybrid).

Артефакты версионируются: artifacts/<index_id>/, где index_id кодирует параметры
(сплиттер, chunk_size, overlap, заголовки, тип векторов, модель) и хэш корпуса.
Повторный запуск с теми же параметрами ничего не пересчитывает; при смене параметров
создаётся новая версия рядом. Dense-векторы кэшируются по тексту (artifacts/_cache).

Примеры (из каталога lab2):
    python source/build_index.py
    python source/build_index.py --splitter recursive --chunk-size 256 --overlap 32
    python source/build_index.py --no-headers --vector-type dense --force
    python source/build_index.py --list
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

import chunking
import parse_docs
import settings
from embeddings import DenseEncoder
from sparse import BM25Encoder


def ensure_markdown(cfg: dict, reparse: bool = False) -> None:
    md_dir = settings.lab_path(cfg, "markdown_dir")
    if reparse or not (md_dir / "manifest.json").exists():
        print("[parse] конвертация документов в Markdown ...")
        parse_docs.parse_all(cfg)


def index_dir(cfg: dict) -> Path:
    return settings.lab_path(cfg, "artifacts_dir") / settings.index_id(cfg)


def build(cfg: dict, force: bool = False, reparse: bool = False) -> Path:
    ensure_markdown(cfg, reparse)
    out = index_dir(cfg)
    if (out / "DONE").exists() and not force:
        print(f"[build] {out.name}: уже построен (используйте --force для пересборки)")
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    params = settings.index_params(cfg)
    vt = params["vector_type"]
    print(f"[build] {out.name}")

    t0 = time.perf_counter()
    chunks, stats = chunking.chunk_corpus(cfg)
    stats["chunking_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"  чанков: {stats['chunks']} (медиана {stats['tokens_median']} ток., max {stats['tokens_max']})")
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
    texts = [ch.embed_text for ch in chunks]

    if vt in ("dense", "hybrid"):
        e = cfg["embeddings"]
        enc = DenseEncoder(e["dense_model"], e["base_url"], e.get("batch_size", 16),
                           settings.lab_path(cfg, "artifacts_dir") / "_cache" / "dense.sqlite")
        vecs, dstats = enc.encode(texts, progress=True)
        np.save(out / "dense.npy", vecs)
        stats.update({f"dense_{k}": v for k, v in dstats.items()})
        stats["dense_dim"] = int(vecs.shape[1]) if vecs.size else None
        print(f"  dense: dim={stats['dense_dim']}, посчитано {dstats['embedded']}, "
              f"из кэша {dstats['from_cache']}, {dstats['embed_seconds']} с")

    if vt in ("sparse", "hybrid"):
        t1 = time.perf_counter()
        bm = BM25Encoder(**cfg["embeddings"].get("bm25", {})).fit(texts)
        with open(out / "sparse.jsonl", "w", encoding="utf-8") as f:
            for t in texts:
                idx, val = bm.encode_doc(t)
                f.write(json.dumps({"indices": idx, "values": val}) + "\n")
        stats["sparse_seconds"] = round(time.perf_counter() - t1, 2)
        stats["bm25"] = bm.state()
        print(f"  sparse (BM25): avgdl={bm.avgdl:.1f}, {stats['sparse_seconds']} с")

    stats["build_seconds"] = round(time.perf_counter() - t0, 2)
    meta = {"index_id": out.name, "params": params, "corpus": settings.corpus_fingerprint(cfg),
            "stats": stats, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "index.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "DONE").touch()
    print(f"  готово за {stats['build_seconds']} с → {out}")
    return out


def load_index(path: Path) -> dict:
    meta = json.loads((path / "index.json").read_text(encoding="utf-8"))
    with open(path / "chunks.jsonl", encoding="utf-8") as f:
        meta["chunks"] = [json.loads(line) for line in f]
    if (path / "dense.npy").exists():
        meta["dense"] = np.load(path / "dense.npy")
    if (path / "sparse.jsonl").exists():
        with open(path / "sparse.jsonl", encoding="utf-8") as f:
            meta["sparse"] = [json.loads(line) for line in f]
    return meta


def list_indexes(cfg: dict) -> None:
    root = settings.lab_path(cfg, "artifacts_dir")
    for p in sorted(root.glob("*/index.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        s = m["stats"]
        print(f"{m['index_id']}: {s['chunks']} чанков, build {s['build_seconds']} с, {m['built_at']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    settings.add_common_args(parser)
    settings.add_index_args(parser)
    parser.add_argument("--force", action="store_true", help="пересобрать, даже если версия уже есть")
    parser.add_argument("--reparse", action="store_true", help="заново конвертировать документы в Markdown")
    parser.add_argument("--list", action="store_true", help="показать построенные версии индекса")
    args = parser.parse_args()
    cfg = settings.config_from_args(args)
    if args.list:
        list_indexes(cfg)
        return
    build(cfg, force=args.force, reparse=args.reparse)


if __name__ == "__main__":
    main()
