import json
from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd

import config
from prompts import PROMPTS


def load_runs() -> pd.DataFrame:
    rows = [json.loads(line) for line in config.RAW_RESULTS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if "error" in df:
        errors = df[df["error"].notna()]
        if len(errors):
            print(f"Пропущено прогонов с ошибкой: {len(errors)}")
        df = df[df["error"].isna()].drop(columns="error")
    df = df.drop_duplicates(subset=["model", "prompt_id", "mode", "repeat"], keep="last")

    quality = [PROMPTS[p]["evaluate"](r) for p, r in zip(df["prompt_id"], df["response"])]
    df["quality"] = [q["score"] for q in quality]
    df["quality_details"] = [json.dumps({k: v for k, v in q.items() if k != "score"}, ensure_ascii=False) for q in quality]
    return df.reset_index(drop=True)


def mean_pairwise_similarity(texts: list[str]) -> float:
    """Средняя попарная схожесть ответов между повторами: 1.0 — ответы идентичны."""
    pairs = list(combinations(texts, 2))
    if not pairs:
        return float("nan")
    return sum(SequenceMatcher(None, a, b).ratio() for a, b in pairs) / len(pairs)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        texts = g["response"].tolist()
        return pd.Series({
            "n": len(g),
            "quality": g["quality"].mean(),
            "out_tokens": g["completion_tokens"].mean(),
            "out_tokens_std": g["completion_tokens"].std(ddof=0),
            "in_tokens": g["prompt_tokens"].mean(),
            "latency_s": g["latency_s"].mean(),
            "ttft_s": g["ttft_s"].mean(),
            "prompt_tps": g["prompt_tps"].mean(),
            "gen_tps": g["gen_tps"].mean(),
            "unique_answers": len(set(texts)),
            "similarity": mean_pairwise_similarity(texts),
            "truncated": (g["finish_reason"] == "length").sum(),
        })

    return (
        df.groupby(["model", "prompt_id", "mode"], sort=True)
        .apply(agg)
        .reset_index()
        .round(3)
    )


def write_markdown(summary: pd.DataFrame, df: pd.DataFrame) -> None:
    lines = ["# Сводные результаты", ""]
    lines += [
        "Колонки: `quality` — автоматическая оценка качества (0–1); `out_tokens` — длина ответа в токенах "
        "(среднее и std по повторам); `ttft_s` — время до первого токена; `gen_tps` — скорость генерации, ток/с; "
        "`prompt_tps` — скорость обработки промпта, ток/с; `unique_answers` — число различных ответов среди "
        "повторов; `similarity` — средняя попарная схожесть ответов (1 = идентичны); `truncated` — ответы, "
        "обрезанные по `max_tokens`.",
        "",
    ]
    for prompt_id, part in summary.groupby("prompt_id"):
        lines += [f"## {prompt_id}: {PROMPTS[prompt_id]['task']}", "", part.drop(columns="prompt_id").to_markdown(index=False), ""]

    if {"A", "B"} <= set(summary["mode"]):
        lines += ["## Сравнение режимов (B − A) по моделям, усреднено по промптам", ""]
        cols = ["quality", "out_tokens", "latency_s", "gen_tps", "similarity"]
        by_mode = summary.groupby(["model", "mode"])[cols].mean().unstack("mode")
        delta = pd.DataFrame({c: by_mode[(c, "B")] - by_mode[(c, "A")] for c in cols}).round(3)
        lines += [delta.to_markdown(), ""]

    lines += ["## Итог по моделям (оба режима)", ""]
    per_model = df.groupby("model")[["quality", "completion_tokens", "latency_s", "ttft_s", "gen_tps"]].mean().round(3)
    lines += [per_model.to_markdown(), ""]

    (config.RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_responses(df: pd.DataFrame) -> None:
    lines = ["# Ответы моделей", ""]
    for prompt_id, pdf in df.groupby("prompt_id"):
        lines += [f"## {prompt_id}: {PROMPTS[prompt_id]['task']}", "", "Промпт:", "", "```text", PROMPTS[prompt_id]["text"], "```", ""]
        for (model, mode), g in pdf.groupby(["model", "mode"]):
            params = g["params"].iloc[0] or "дефолты движка"
            lines += [f"### {model} — режим {mode}", "", f"Параметры: `{params}`", ""]
            for _, r in g.sort_values("repeat").iterrows():
                lines += [
                    f"**Повтор {r['repeat']}** — quality={r['quality']:.2f}, {r['completion_tokens']} ток., "
                    f"{r['latency_s']} с, finish_reason={r['finish_reason']}, детали: `{r['quality_details']}`",
                    "",
                    "~~~~text",  # тильды, т.к. ответы сами часто содержат ```json
                    r["response"].strip(),
                    "~~~~",
                    "",
                ]
    (config.RESULTS_DIR / "responses.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_runs()
    summary = summarize(df)

    df.drop(columns=["response"]).to_csv(config.RESULTS_DIR / "runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(config.RESULTS_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    write_markdown(summary, df)
    write_responses(df)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary)
    print(f"\nФайлы сохранены в {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
