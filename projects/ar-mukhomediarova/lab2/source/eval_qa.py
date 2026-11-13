"""Оценка ответов RAG-агента на оценочных вопросах.

Метрики: key_fact_recall — доля ключевых фактов эталона в ответе; citation_precision — доля
процитированных фрагментов, которые релевантны по разметке; no_answer — агент сказал, что ответа нет.
Все ответы сохраняются в results/qa_answers.md для ручной проверки.

    python source/eval_qa.py                               # конфигурация из config.yaml
    python source/eval_qa.py --set chunking.chunk_size=256 --mode hybrid
"""
from __future__ import annotations

import argparse
import statistics

import pandas as pd

import build_index
import relevance
import settings
from rag import RAGEngine, format_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    settings.add_common_args(parser)
    settings.add_index_args(parser)
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"])
    args = parser.parse_args()
    cfg = settings.config_from_args(args)
    engine = RAGEngine(cfg)
    chunks = build_index.load_index(build_index.index_dir(cfg))["chunks"]
    by_id = {c["chunk_id"]: c for c in chunks}

    rows, md = [], [f"# Ответы RAG-агента\n\nКоллекция: `{engine.retriever.collection}`, "
                    f"LLM: `{cfg['llm']['model']}`, режим: `{args.mode or cfg['search']['mode']}`\n"]
    for q in relevance.load_questions():
        if not relevance.answerable(q, chunks):
            print(f"[skip] {q['id']}: нет документов для разметки")
            continue
        res = engine.ask(q["question"], mode=args.mode)
        cited = [c for c in res["retrieved"] if c["cited"]]
        cited_rel = [bool(relevance.groups_hit(by_id[c["chunk_id"]], q)) for c in cited]
        row = {
            "qid": q["id"],
            "key_fact_recall": relevance.key_fact_recall(res["answer"], q.get("key_facts", [])),
            "citation_precision": (sum(cited_rel) / len(cited_rel)) if cited_rel else 0.0,
            "n_cited": len(cited),
            "retrieved_relevant": sum(bool(relevance.groups_hit(by_id[c["chunk_id"]], q))
                                      for c in res["retrieved"]),
            "no_answer": "нет ответа" in res["answer"].lower(),
            "llm_s": round(res["timings"]["llm_ms"] / 1000, 1),
            "prompt_tokens": (res["tokens"] or {}).get("prompt"),
            "completion_tokens": (res["tokens"] or {}).get("completion"),
        }
        rows.append(row)
        print(f"[{q['id']}] facts={row['key_fact_recall']:.2f} cit_prec={row['citation_precision']:.2f} "
              f"llm={row['llm_s']} с")
        md += [f"## {q['id']}. {q['question']}\n", f"**Эталон:** {q['answer']}\n",
               f"**Метрики:** key_fact_recall={row['key_fact_recall']:.2f}, "
               f"citation_precision={row['citation_precision']:.2f}\n",
               "```text", format_answer(res, show_context=True), "```\n"]

    df = pd.DataFrame(rows)
    out = settings.lab_path(cfg, "results_dir")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "qa_results.csv", index=False)
    summary = {k: round(statistics.mean(df[k]), 3) for k in ("key_fact_recall", "citation_precision")}
    summary.update({"no_answer": int(df["no_answer"].sum()), "llm_s_median": float(df["llm_s"].median()),
                    "n": len(df)})
    md.insert(1, "**Итого:** " + ", ".join(f"{k}={v}" for k, v in summary.items()) + "\n\n"
              + df.to_markdown(index=False) + "\n")
    (out / "qa_answers.md").write_text("\n".join(md), encoding="utf-8")
    print("\n", summary, f"\nОтветы: {out}/qa_answers.md")


if __name__ == "__main__":
    main()
