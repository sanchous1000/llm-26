"""Шаг 5. Прогоняет конфигурации из experiments.yml по очереди: для каждой — пересобрать
индекс, загрузить в ту же коллекцию (заменяя предыдущую), прогнать eval/questions.jsonl через
retrieval. Метрики сам не считает — сохраняет только сырые найденные чанки на каждую
конфигурацию (results/retrieval_dump_<имя>.json), чтобы метрики посчитать отдельно по
присланному дампу, не доверяя автоматическому подсчёту вслепую.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # чтобы работал import book_rag_backend

from book_rag_backend.config import ROOT, apply_overrides, cfg_from_args, add_common_args
from book_rag_backend.services.embeddings import EmbeddingsService, SparseEmbeddingsService
from book_rag_backend.services.evaluation import load_questions, retrieval_dump
from book_rag_backend.services.qdrant import get_qdrant_client
from book_rag_backend.services.rag import RAGService

import build_index
import load_to_vector_store

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(ap)
ap.add_argument("--experiments", default=str(ROOT / "experiments.yml"))
ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
ap.add_argument("--k", type=int, default=10)
args = ap.parse_args()


async def main():
    base_cfg = cfg_from_args(args)
    experiments = yaml.safe_load(Path(args.experiments).read_text(encoding="utf-8"))["experiments"]
    questions = load_questions(Path(args.questions))

    for exp in experiments:
        print(f"\n=== {exp['name']}")
        cfg = apply_overrides(base_cfg, {k: v for k, v in exp.items() if k != "name"})

        manifest = build_index.build(cfg, force=True)
        client = get_qdrant_client(cfg.qdrant.url, cfg.qdrant.api_key)
        store = await load_to_vector_store.load(cfg, rebuild=True, client=client)

        vz = manifest["params"]["vectorizer"]
        embeddings = EmbeddingsService.from_name(manifest["params"]["dense_model"]) if vz in ("dense", "hybrid") else None
        sparse = SparseEmbeddingsService.from_name() if vz in ("sparse", "hybrid") else None
        rag = RAGService(store, vz, embeddings, sparse, llm=None, top_k=args.k, ef_search=cfg.qdrant.hnsw_ef_search)

        dump = await retrieval_dump(rag, questions, k=args.k)
        dump_path = ROOT / "results" / f"retrieval_dump_{exp['name']}.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps({"params": manifest["params"], "n_chunks": manifest["n_chunks"],
                                         "k": args.k, "results": dump}, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"Сохранено: {dump_path}")
        await client.close()

    print("\nГотово. Пришли файлы results/retrieval_dump_<имя>.json — посчитаю метрики и сравню конфигурации.")


asyncio.run(main())