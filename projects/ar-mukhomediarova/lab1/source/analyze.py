"""Агрегация сырых прогонов в таблицы для отчёта.

Запуск из каталога lab1:
    python source/analyze.py
Создаёт в results/:
    runs_scored.csv  — все прогоны с метриками качества
    summary.csv      — агрегаты по модель × промпт × режим
    summary.md       — те же таблицы в markdown (для README)
    responses.md     — все ответы моделей в читаемом виде
"""
import itertools
import json
from difflib import SequenceMatcher

import pandas as pd

import config
from evaluate import evaluate
from prompts import PROMPTS


def load_runs() -> pd.DataFrame:
    with open(config.RAW_RESULTS, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    if "error" in df:
        errors = df[df["error"].notna()]
        if len(errors):
            print(f"Внимание: {len(errors)} прогонов с ошибкой, они исключены")
        df = df[df["error"].isna()].copy()
    return df


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    ev = [evaluate(p, r) for p, r in zip(df["prompt_id"], df["response"])]
    df["score"] = [e["score"] for e in ev]
    df["repeated_trigrams"] = [e["repeated_trigrams"] for e in ev]
    df["format_ok"] = [e.get("format_ok", e.get("json_only", e.get("length_ok"))) for e in ev]
    df["details"] = [json.dumps({k: v for k, v in e.items()
                                 if k not in ("score", "repeated_trigrams")}, ensure_ascii=False)
                     for e in ev]
    return df


def pairwise_similarity(texts: list[str]) -> float:
    """Средняя попарная похожесть ответов из повторов (1.0 — полностью совпадают)."""
    pairs = list(itertools.combinations(texts, 2))
    if not pairs:
        return float("nan")
    return round(sum(SequenceMatcher(None, a, b).ratio() for a, b in pairs) / len(pairs), 3)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, prompt_id, mode), g in df.groupby(["model", "prompt_id", "mode"], sort=False):
        rows.append({
            "model": model,
            "prompt": prompt_id,
            "mode": mode,
            "n": len(g),
            "score": round(g["score"].mean(), 3),
            "format_ok": f"{int(g['format_ok'].fillna(False).sum())}/{len(g)}",
            "out_tok_mean": round(g["completion_tokens"].mean(), 1),
            "out_tok_std": round(g["completion_tokens"].std(ddof=0), 1),
            "in_tok": g["prompt_tokens"].iloc[0],
            "latency_s": round(g["latency_s"].mean(), 2),
            "ttft_s": round(g["ttft_s"].mean(), 2),
            "prefill_tps": round(g["prefill_tps"].mean(), 1),
            "decode_tps": round(g["decode_tps"].mean(), 1),
            "unique_answers": f"{g['response'].nunique()}/{len(g)}",
            "similarity": pairwise_similarity(g["response"].tolist()),
            "rep_trigrams": round(g["repeated_trigrams"].mean(), 3),
            "truncated": int((g["finish_reason"] == "length").sum()),
        })
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, df: pd.DataFrame) -> None:
    lines = ["# Сводные результаты\n"]
    for prompt_id, g in summary.groupby("prompt", sort=False):
        lines += [f"## {prompt_id}\n", g.drop(columns="prompt").to_markdown(index=False), ""]

    speed = (df.groupby(["model", "mode"])[["prefill_tps", "decode_tps", "latency_s"]]
             .mean().round(1).reset_index())
    lines += ["## Скорость по моделям (среднее по всем промптам)\n", speed.to_markdown(index=False), ""]

    quality = summary.pivot_table(index="model", columns=["prompt", "mode"], values="score").round(2)
    quality.columns = [f"{p} {m}" for p, m in quality.columns]
    lines += ["## Качество (score) модель × промпт × режим\n", quality.to_markdown(), ""]

    (config.RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_responses(df: pd.DataFrame) -> None:
    out = ["# Ответы моделей\n"]
    for prompt_id, prompt in PROMPTS.items():
        out += [f"## {prompt_id}\n", "**Промпт:**\n", "```text", prompt, "```\n"]
        for _, r in df[df["prompt_id"] == prompt_id].iterrows():
            out += [
                f"### {r['model']} · {r['mode']} · повтор {r['repeat']}\n",
                f"score={r['score']}, токены вх/вых={r['prompt_tokens']}/{r['completion_tokens']}, "
                f"время={r['latency_s']} с, finish={r['finish_reason']}, детали: `{r['details']}`\n",
                "```text", str(r["response"]).strip(), "```\n",
            ]
    (config.RESULTS_DIR / "responses.md").write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    df = add_scores(load_runs())
    df.drop(columns=["params"]).to_csv(config.RESULTS_DIR / "runs_scored.csv", index=False)
    summary = summarize(df)
    summary.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
    write_markdown(summary, df)
    write_responses(df)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary)
    print(f"\nТаблицы сохранены в {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
