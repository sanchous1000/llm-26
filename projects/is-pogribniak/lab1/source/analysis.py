"""Анализ results/: runs.jsonl + models.json → summary.md, answers.md, plots/."""

import json
from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd

from .config import RESULTS, build_messages
from .plots import draw_all
from .scoring import score

CELL = ["model", "prompt_id", "mode"]
COMMON = ["model", "prompt_id", "mode", "prompt_tokens", "completion_mean", "completion_std", "truncated",
          "unique_answers", "text_similarity", "ttft_s", "latency_s", "gen_tps"]
TASK = { 
    "P1": ["quality_mean", "quality_std", "hallucinations", "forbidden_hits"],
    "P2": ["quality_mean", "quality_std", "parse_ok", "acc_in_scope", "unknown_recall",
           "false_unknown", "forced_labels", "hallucinations"],
    "P3": ["quality_mean", "quality_std", "toxic_words", "refused", "forbidden_hits", "hallucinations"],
}


def read_runs() -> pd.DataFrame:
    path = RESULTS / "runs.jsonl"
    return pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line])


def score_runs(runs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    messages = {p["id"]: build_messages(p) for p in cfg["prompts"]}
    scores = pd.DataFrame([score(r.prompt_id, r.response, messages[r.prompt_id]) for r in runs.itertuples()])
    df = pd.concat([runs, scores], axis=1)
    gen_time = df["latency_s"] - df["ttft_s"].fillna(0)
    df["gen_tps"] = (df["completion_tokens"] / gen_time).where(gen_time > 0)
    return df


def similarity(texts: pd.Series) -> float:
    """Среднее попарное сходство повторов одной ячейки (1 — все одинаковые)."""
    pairs = list(combinations(texts, 2))
    return sum(SequenceMatcher(None, a, b).ratio() for a, b in pairs) / len(pairs) if pairs else 1.0


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    spec = {
        "prompt_tokens": ("prompt_tokens", "mean"),
        "completion_mean": ("completion_tokens", "mean"),
        "completion_std": ("completion_tokens", "std"),
        "truncated": ("finish_reason", lambda s: (s == "length").sum()),
        "unique_answers": ("canonical", "nunique"),
        "text_similarity": ("response", similarity),
        "quality_mean": ("quality", "mean"),
        "quality_std": ("quality", "std"),
        "parse_ok": ("parse_ok", "mean"),
        "hallucinations": ("hallucinations", "sum"),
        "forbidden_hits": ("forbidden_hits", "sum"),
        "refused": ("refused", "sum"),
        "toxic_words": ("toxic_words", "mean"),
        "acc_in_scope": ("acc_in_scope", "mean"),
        "unknown_recall": ("unknown_recall", "mean"),
        "false_unknown": ("false_unknown", "sum"),
        "forced_labels": ("forced_labels", "sum"),
        "ttft_s": ("ttft_s", "mean"),
        "latency_s": ("latency_s", "mean"),
        "gen_tps": ("gen_tps", "mean"),
    }
    return df.groupby(CELL).agg(**{k: v for k, v in spec.items() if v[0] in df}).reset_index()


def model_speed(df: pd.DataFrame) -> pd.DataFrame:
    speed = df.groupby("model").agg(gen_tps=("gen_tps", "mean"), gen_tps_std=("gen_tps", "std"))
    path = RESULTS / "models.json"
    if path.exists():
        prefill = {name: pd.Series([r["prompt_tps"] for r in m.get("prefill") or []], dtype=float)
                   for name, m in json.loads(path.read_text(encoding="utf-8")).items()}
        speed["prompt_tps"] = speed.index.map(lambda n: prefill.get(n, pd.Series(dtype=float)).mean())
        speed["prompt_tps_std"] = speed.index.map(lambda n: prefill.get(n, pd.Series(dtype=float)).std())
    return speed.reset_index()


def md(df: pd.DataFrame, cols: list | None = None) -> str:
    return (df[[c for c in cols if c in df]] if cols else df).to_markdown(index=False, floatfmt=".2f")


def params_section(cfg: dict) -> str:
    lines = ["Режим A — без гиперпараметров в запросе (дефолты из Modelfile, см. models.json)."]
    lines += [f"Всегда у {m['name']}: `{json.dumps(m['extra_params'], ensure_ascii=False)}`"
              for m in cfg["models"] if m.get("extra_params")]
    lines += [f"Режим B, {p['id']}: `{json.dumps(p['params_b'], ensure_ascii=False)}`"
              for p in cfg["prompts"] if p.get("params_b")]
    return "\n".join(f"- {line}" for line in lines)


def write_answers(df: pd.DataFrame) -> None:
    out = ["# Все ответы моделей\n"]
    for (model, pid, mode), g in df.groupby(CELL, sort=False):
        out.append(f"## {model} · {pid} · режим {mode}\n")
        for r in g.itertuples():
            out.append(f"### Прогон {r.rep} — {r.completion_tokens} ток., {r.latency_s:.2f} c, "
                       f"качество {r.quality:.2f}\n\n```\n{r.response}\n```\n")
    (RESULTS / "answers.md").write_text("\n".join(out), encoding="utf-8")


def analyze(cfg: dict) -> None:
    df = score_runs(read_runs(), cfg)
    summary = summarize(df)
    speed = model_speed(df)
    write_answers(df)

    parts = ["# Сводка", "## Параметры\n\n" + params_section(cfg),
             "## Длина, стабильность, время\n\n" + md(summary, COMMON)]
    for pid, cols in TASK.items():
        sub = summary[summary["prompt_id"] == pid]
        if not sub.empty:
            parts.append(f"## Качество {pid}\n\n" + md(sub, ["model", "mode"] + cols))
    parts.append("## Скорость по моделям\n\ngen_tps — по основным прогонам; prompt_tps (prefill) — "
                 "3 уникальных текста по 1500 слов с max_tokens=1.\n\n" + md(speed))
    if {"A", "B"} <= set(summary["mode"]):
        ab = summary.pivot_table(index=["model", "prompt_id"], columns="mode",
                                 values=["completion_mean", "quality_mean", "text_similarity", "latency_s"])
        ab.columns = [f"{m}_{mode}" for m, mode in ab.columns]
        parts.append("## A против B\n\n" + md(ab.reset_index()))
    (RESULTS / "summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")

    draw_all(summary, speed, RESULTS / "plots")
    print(f"Готово: {RESULTS}/summary.md, answers.md, plots/")
