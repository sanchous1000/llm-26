"""Оценка retrieval: Recall@k, Precision@k, MRR, Hit@k на вопросах из eval/questions.json.

Чанк релевантен, если его документ совпадает с эталонным и диапазон страниц [page_start, page_end]
содержит хотя бы одну эталонную страницу. Эталон не привязан к ID чанков, поэтому
один и тот же набор вопросов подходит для любых параметров нарезки.

    Recall@k    - доля эталонных (документ, страница), покрытых чанками из топ-k
    Precision@k - доля релевантных чанков среди топ-k
    MRR         - среднее 1/rank первого релевантного чанка (в топ-max(k))
    Hit@k       - доля вопросов, где в топ-k есть хотя бы один релевантный чанк

Примеры:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --config config_v2.yml --label chunk128
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

from doc_rag.cli import add_splitter_args, apply_splitter_overrides
from doc_rag.config import SOURCE_DIR, load_config
from doc_rag.services.embeddings import EmbeddingsService
from doc_rag.services.qdrant import QdrantService

DEFAULT_QUESTIONS = SOURCE_DIR / "eval" / "questions.json"
RESULTS_DIR = SOURCE_DIR.parent / "results"
KS = (5, 10)


def is_relevant(hit: dict, targets: list[dict]) -> bool:
    return any(
        hit["source"] == t["source"] and hit["page_start"] <= p <= hit["page_end"]
        for t in targets
        for p in t["pages"]
    )


def covered_pages(hits: list[dict], targets: list[dict]) -> int:
    return sum(
        any(h["source"] == t["source"] and h["page_start"] <= p <= h["page_end"] for h in hits)
        for t in targets
        for p in t["pages"]
    )


def evaluate_question(hits: list[dict], targets: list[dict]) -> dict:
    flags = [is_relevant(h, targets) for h in hits]
    total_pages = sum(len(t["pages"]) for t in targets)
    row = {"first_rank": next((i + 1 for i, f in enumerate(flags) if f), None)}
    row["mrr"] = 1 / row["first_rank"] if row["first_rank"] else 0.0
    for k in KS:
        row[f"recall@{k}"] = covered_pages(hits[:k], targets) / total_pages
        row[f"precision@{k}"] = sum(flags[:k]) / k
        row[f"hit@{k}"] = float(any(flags[:k]))
    return row


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    add_splitter_args(p)
    p.add_argument("--collection", help="имя коллекции (по умолчанию вычисляется из конфига)")
    p.add_argument("--label", help="метка запуска для файлов результатов (по умолчанию - имя коллекции)")
    p.add_argument("--verbose", action="store_true", help="печатать топ-3 чанка для каждого вопроса")
    args = p.parse_args()

    config = apply_splitter_overrides(load_config(args.config), args)
    collection = args.collection or config.collection_name()
    label = args.label or collection
    questions = json.loads(args.questions.read_text(encoding="utf-8"))

    embeddings = EmbeddingsService(config.embeddings)
    qdrant = QdrantService(config.qdrant, collection)
    if not qdrant.collection_exists():
        sys.exit(f"Коллекция {collection} не найдена - сначала scripts/build_index.py")

    rows, latencies = [], []
    for q in questions:
        t0 = time.perf_counter()
        hits = qdrant.search(embeddings.embed(q["question"]), top_k=max(KS), score_threshold=None)
        latencies.append(time.perf_counter() - t0)
        row = {"id": q["id"], **evaluate_question(hits, q["relevant"])}
        rows.append(row)
        if args.verbose:
            print(f"\n{q['id']} {q['question']}\n  эталон: {q['relevant']}")
            for i, h in enumerate(hits[:3], 1):
                mark = "+" if is_relevant(h, q["relevant"]) else "-"
                print(f"  {mark} #{i} {h['score']:.3f} {h['source'][:25]} стр.{h['page_start']}-{h['page_end']}: "
                      f"{h['text'][:70]!r}")

    metrics = ["mrr"] + [f"{m}@{k}" for k in KS for m in ("recall", "precision", "hit")]
    summary = {m: statistics.mean(r[m] for r in rows) for m in metrics}
    summary["search_ms"] = statistics.mean(latencies) * 1000
    summary["points"] = qdrant.count()

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"eval_{label}_questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "first_rank", "mrr"] + metrics[1:])
        w.writeheader()
        w.writerows(rows)

    s = config.splitter
    stats_file = RESULTS_DIR / "index_stats.json"
    index_stats = json.loads(stats_file.read_text(encoding="utf-8")).get(collection, {}) if stats_file.exists() else {}
    summary_row = {
        "label": label, "collection": collection, "model": config.embeddings.model_name,
        "chunk_size": s.chunk_size, "overlap": s.chunk_overlap, "headers_in_text": s.include_headers_in_text,
        **{k: round(v, 3) for k, v in summary.items()},
        "index_s": index_stats.get("index_s", ""),
    }
    summary_file = RESULTS_DIR / "eval_summary.csv"
    existing = []
    if summary_file.exists():
        with open(summary_file, newline="", encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if r["label"] != label]
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row))
        w.writeheader()
        w.writerows(existing + [summary_row])

    print(f"\nКоллекция {collection} ({summary['points']} точек), вопросов: {len(rows)}, "
          f"chunk_size={s.chunk_size} overlap={s.chunk_overlap} headers_in_text={s.include_headers_in_text}")
    print(f"{'метрика':<14}{'значение':>9}")
    for m in metrics:
        print(f"{m:<14}{summary[m]:>9.3f}")
    print(f"{'search_ms':<14}{summary['search_ms']:>9.1f}")
    miss = [r["id"] for r in rows if not r["hit@10"]]
    print(f"Без релевантного чанка в топ-10: {miss or 'нет'}")
    qdrant.close()


if __name__ == "__main__":
    main()
