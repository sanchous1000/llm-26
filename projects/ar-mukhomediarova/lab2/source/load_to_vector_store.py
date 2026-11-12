"""Шаг 4. Загрузка индекса (чанки + векторы) в Qdrant.

Коллекция: <prefix>_<index_id>_m<M>_ef<ef_construct>[_q8]
  - dense-вектор "dense" (HNSW: M, ef_construct; ef_search задаётся при поиске),
  - sparse-вектор "bm25" (инвертированный индекс, IDF считает Qdrant),
  - payload: текст чанка, документ, раздел, страницы, путь, дата и т. д.

Идемпотентность: id точек детерминированы (uuid5 от chunk_id), поэтому повторная загрузка
перезаписывает те же точки. Если коллекция уже содержит все точки — ничего не делается.

Примеры (из каталога lab2):
    python source/load_to_vector_store.py                       # построить (если нужно) и загрузить
    python source/load_to_vector_store.py --chunk-size 256      # другая версия индекса
    python source/load_to_vector_store.py --rebuild             # пересоздать коллекцию
    python source/load_to_vector_store.py --drop                # удалить коллекцию текущей конфигурации
    python source/load_to_vector_store.py --reset-all           # удалить все коллекции с префиксом
    python source/load_to_vector_store.py --list
"""
from __future__ import annotations

import argparse
import time

import build_index
import settings
import vector_store as vs


def load(cfg: dict, rebuild: bool = False, force_build: bool = False) -> str:
    path = build_index.build(cfg, force=force_build)
    index = build_index.load_index(path)
    name = settings.collection_name(cfg)
    client = vs.connect(cfg)
    n = len(index["chunks"])

    if client.collection_exists(name):
        if rebuild or force_build:
            client.delete_collection(name)
            print(f"[qdrant] {name}: удалена для пересоздания")
        else:
            count = client.count(name, exact=True).count
            if count == n:
                print(f"[qdrant] {name}: уже загружена ({count} точек)")
                return name
            print(f"[qdrant] {name}: {count}/{n} точек — догружаю")

    t0 = time.perf_counter()
    if not client.collection_exists(name):
        dense = index.get("dense")
        vs.create_collection(client, name, cfg,
                             dense_dim=int(dense.shape[1]) if dense is not None and dense.size else None,
                             has_sparse="sparse" in index)
        print(f"[qdrant] {name}: создана")
    vs.upsert_index(client, name, index)
    status = vs.wait_indexed(client, name)
    print(f"[qdrant] {name}: {client.count(name, exact=True).count} точек, статус {status}, "
          f"{time.perf_counter() - t0:.1f} с")
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    settings.add_common_args(parser)
    settings.add_index_args(parser)
    parser.add_argument("--rebuild", "--drop-and-reindex", dest="rebuild", action="store_true",
                        help="удалить и заново создать коллекцию")
    parser.add_argument("--force-build", action="store_true", help="пересобрать и артефакты индекса")
    parser.add_argument("--drop", action="store_true", help="только удалить коллекцию")
    parser.add_argument("--reset-all", action="store_true", help="удалить все коллекции с префиксом")
    parser.add_argument("--list", action="store_true", help="показать коллекции")
    args = parser.parse_args()
    cfg = settings.config_from_args(args)
    client = vs.connect(cfg)
    prefix = cfg["vector_store"]["collection_prefix"] + "_"

    if args.list:
        for c in client.get_collections().collections:
            if c.name.startswith(prefix):
                print(f"{c.name}: {client.count(c.name, exact=True).count} точек")
        return
    if args.reset_all:
        for c in client.get_collections().collections:
            if c.name.startswith(prefix):
                client.delete_collection(c.name)
                print(f"[qdrant] удалена {c.name}")
        return
    if args.drop:
        name = settings.collection_name(cfg)
        print(f"[qdrant] {name}: " + ("удалена" if client.delete_collection(name) else "не найдена"))
        return
    load(cfg, rebuild=args.rebuild, force_build=args.force_build)


if __name__ == "__main__":
    main()
