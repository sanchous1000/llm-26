"""Шаг 5. Оценка качества retrieval для набора конфигураций (experiments.yaml).

Для каждой конфигурации индекс строится/загружается при необходимости (артефакты и коллекции
переиспользуются), затем по каждому вопросу считаются Recall@k, Precision@k, Hit@k, MRR.
Дополнительно: время построения, задержка поиска, для dense — согласие HNSW с точным поиском.

Запуск из каталога lab2:
    python source/eval_retrieval.py
    python source/eval_retrieval.py --only E3_hybrid E4_hybrid_cs256
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import pandas as pd
import yaml

import build_index
import load_to_vector_store
import relevance
import settings
from retriever import Retriever


def run_experiment(base_cfg: dict, exp: dict, questions: list[dict], ks: list[int]) -> tuple[dict, list[dict]]:
    cfg = settings.apply_overrides(base_cfg, exp.get("set"))
    collection = load_to_vector_store.load(cfg)
    index = build_index.load_index(build_index.index_dir(cfg))
    chunks = index["chunks"]
    retr = Retriever(cfg)
    mode = exp.get("mode", cfg["search"]["mode"])
    kmax = max(ks)

    rows, latencies, ann = [], [], []
    for q in questions:
        if not relevance.answerable(q, chunks):
            continue
        hits, t = retr.search(q["question"], top_k=kmax, mode=mode)
        latencies.append(t["embed_ms"] + t["search_ms"])
        m = relevance.retrieval_metrics(hits, q, ks)
        if mode == "dense":  # насколько HNSW совпадает с точным перебором
            exact, _ = retr.search(q["question"], top_k=kmax, mode=mode, exact=True)
            ann.append(len({h["chunk_id"] for h in hits} & {h["chunk_id"] for h in exact}) / kmax)
        rows.append({"experiment": exp["name"], "qid": q["id"], **m,
                     "top_chunks": " | ".join(f"{h['rank']}:{h['chunk_id']}" for h in hits[:5])})

    st = index["stats"]
    summary = {
        "experiment": exp["name"],
        "splitter": cfg["chunking"]["splitter"],
        "chunk_size": cfg["chunking"]["chunk_size"],
        "overlap": cfg["chunking"]["overlap"],
        "headers": cfg["chunking"]["include_headers"],
        "mode": mode,
        "n_questions": len(rows),
        **{k: round(statistics.mean(r[k] for r in rows), 3)
           for k in [f"{m}@{k}" for m in ("recall", "precision", "hit") for k in ks] + ["mrr"]},
        "chunks": st["chunks"],
        "tokens_mean": st["tokens_mean"],
        "build_s": st["build_seconds"],
        "dense_embed_s": st.get("dense_embed_seconds"),
        "query_ms": round(statistics.median(latencies), 1) if latencies else None,
        "ann_agreement@10": round(statistics.mean(ann), 3) if ann else None,
        "collection": collection,
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    settings.add_common_args(parser)
    parser.add_argument("--experiments", default=str(settings.SOURCE_DIR / "experiments.yaml"))
    parser.add_argument("--only", nargs="*", help="запустить только эти эксперименты")
    args = parser.parse_args()
    base = settings.load_config(args.config, args.set)
    build_index.ensure_markdown(base)
    spec = yaml.safe_load(open(args.experiments, encoding="utf-8"))
    ks = spec.get("ks", [5, 10])
    questions = relevance.load_questions()

    summaries, per_q = [], []
    for exp in spec["experiments"]:
        if args.only and exp["name"] not in args.only:
            continue
        print(f"\n=== {exp['name']} ===")
        t0 = time.perf_counter()
        s, rows = run_experiment(base, exp, questions, ks)
        print(f"  {s['n_questions']} вопросов, recall@5={s['recall@5']}, mrr={s['mrr']} "
              f"({time.perf_counter() - t0:.0f} с)")
        summaries.append(s)
        per_q += rows

    out = settings.lab_path(base, "results_dir")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summaries)
    df.to_csv(out / "retrieval_metrics.csv", index=False)
    pd.DataFrame(per_q).to_csv(out / "retrieval_per_question.csv", index=False)
    cols = [c for c in df.columns if c != "collection"]
    pq = pd.DataFrame(per_q).pivot_table(index="qid", columns="experiment", values="recall@5")
    (out / "retrieval_metrics.md").write_text(
        "# Метрики retrieval\n\n" + df[cols].to_markdown(index=False)
        + "\n\n## Recall@5 по вопросам\n\n" + pq.round(2).to_markdown() + "\n", encoding="utf-8")
    (out / "retrieval_config.json").write_text(
        json.dumps({"base_config": base, "experiments": spec}, ensure_ascii=False, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("\n", df[cols])
    print(f"\nРезультаты: {out}/retrieval_metrics.md")


if __name__ == "__main__":
    main()
