"""Шаг 3. Парсинг корпуса, чанкинг и эмбеддинги. Результат — в artifacts/ (один набор файлов,
без версионирования): chunks.jsonl, dense.npy и/или sparse.jsonl, manifest.json.

Повторный запуск с теми же параметрами ничего не пересчитывает (manifest.json сверяется с
текущим конфигом). Любое изменение splitter/embeddings в config.yml или через флаги — и
индекс пересобирается заново. --force пересобирает в любом случае.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # чтобы работал import book_rag_backend

from book_rag_backend.config import Config, add_common_args, cfg_from_args, path_of
from book_rag_backend.services.embeddings import EmbeddingsService, SparseEmbeddingsService
from book_rag_backend.services.parser import ParserService, write_markdown
from book_rag_backend.services.splitter import SplitterService, chunk_documents


def current_params(cfg: Config) -> dict:
    return {
        "splitter": cfg.splitter.model_dump(),
        "vectorizer": cfg.embeddings.vectorizer,
        "dense_model": cfg.embeddings.dense_model if cfg.embeddings.vectorizer in ("dense", "hybrid") else None,
    }


def build(cfg: Config, force: bool = False, log=print) -> dict:
    """Возвращает manifest (готовый или только что собранный)."""
    art = path_of(cfg, "corpus", "artifacts_dir")
    manifest_path = art / "manifest.json"
    params = current_params(cfg)

    if manifest_path.exists() and not force:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("params") == params:
            log(f"Индекс уже собран с этими параметрами ({old['n_chunks']} чанков), пересчёт не нужен")
            return old

    raw_dir = path_of(cfg, "corpus", "raw_dir")
    if not raw_dir.exists() or not ParserService.list_files(raw_dir):
        raise SystemExit(f"В {raw_dir} нет PDF. Запустите сначала scripts/download_corpus.py")

    t0 = time.perf_counter()
    docs = ParserService().parse_corpus(raw_dir)
    write_markdown(docs, path_of(cfg, "corpus", "markdown_dir"))
    log(f"Разобрано документов: {len(docs)}, страниц: {sum(len(d.pages) for d in docs)}, "
        f"знаков: {sum(len(p.markdown) for d in docs for p in d.pages)}")

    # dense-модель грузится один раз, здесь, и используется дважды: сплиттером — чтобы резать
    # по её же токенам (не влезть в её лимит), и ниже — для реальных эмбеддингов. Один и тот же
    # объект в обоих местах — чтобы подсчёт длины чанка точно совпадал с тем, что увидит модель.
    emb = EmbeddingsService.from_name(cfg.embeddings.dense_model)

    sp = cfg.splitter
    chunks = chunk_documents(docs, SplitterService(sp.chunk_size, sp.chunk_overlap, emb), sp.include_headings)
    log(f"Чанков: {len(chunks)} (в среднем {np.mean([c['n_tokens'] for c in chunks]):.0f} токенов)")

    texts = [c["embed_text"] for c in chunks]
    art.mkdir(parents=True, exist_ok=True)
    (art / "dense.npy").unlink(missing_ok=True)   # чтобы не остался файл от предыдущего
    (art / "sparse.jsonl").unlink(missing_ok=True)  # запуска с другим vectorizer/числом чанков
    dense_dim = None
    if params["vectorizer"] in ("dense", "hybrid"):
        dense = np.array(emb.embed_batch(texts), dtype=np.float32)
        dense_dim = int(dense.shape[1])
        np.save(art / "dense.npy", dense)
    if params["vectorizer"] in ("sparse", "hybrid"):
        enc = SparseEmbeddingsService.from_name()
        with open(art / "sparse.jsonl", "w", encoding="utf-8") as f:
            for i, v in enc.embed_batch(texts):
                f.write(json.dumps({"i": i, "v": v}) + "\n")

    with open(art / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    manifest = {"params": params, "n_docs": len(docs), "n_chunks": len(chunks),
                "dense_dim": dense_dim, "seconds": round(time.perf_counter() - t0, 1)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Артефакты сохранены в {art}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--force", action="store_true", help="пересобрать, даже если параметры не менялись")
    args = ap.parse_args()
    build(cfg_from_args(args), force=args.force)
    print("Дальше: python scripts/load_to_vector_store.py")