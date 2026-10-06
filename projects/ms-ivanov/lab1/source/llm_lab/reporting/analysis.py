"""Сводный эмпирический анализ всех моделей: цифры для раздела «Результаты» README.

Главная метрика — доля ПРИЕМЛЕМЫХ ответов: ответ не пуст, пройдены все
ключевые для задачи автопроверки (KEY_CHECKS) и ручной разбор
(results/manual_review.json) не нашёл в нём критической проблемы.
Средний балл автопроверок приводится рядом, но он смешивает суть
и форму и хуже различает модели.

Запуск:  python source/main.py analyze   →  results/analysis.md, results/analysis.json
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable

from llm_lab.evaluation.checks import CODE_SWITCH_MARKER, HALLUCINATION_MARKER
from llm_lab.evaluation.text import mean_pairwise_similarity
from llm_lab.experiment.config import ALL_PRESETS
from llm_lab.experiment.prompts import ALL_PROMPTS
from llm_lab.reporting.markdown import fmt, table
from llm_lab.storage import RUNS_FILENAME, iter_model_dirs, load_runs, utc_now_iso, write_text

Record = dict[str, Any]

# Ключевые проверки: без них ответ непригоден по сути задачи. Лимит слов,
# подпись и формат нумерации — вторичны и учитываются только в среднем балле.
KEY_CHECKS: dict[str, tuple[str, ...]] = {
    "P1": ("указана причина", "предложение о разовых заявках", "нет выдуманных сумм и фамилий",
           "числа только из условия", "только русский язык"),
    "P2": ("точность классификации",),
    "P3": ("схема, значения и null-поля", "без markdown-обёртки"),
    "P4": ("нет выдуманных чисел", "указана первопричина", "указаны меры", "ключевой факт не искажён",
           "только русский язык"),
    "P5": ("верный итог (1431 ₽)",),
    "P6": ("честность вместо выдумки", "числа только из вопроса", "только русский язык"),
}
LENGTH_LIMITED_PRESET = "G_budget"
SWEEP_PRESETS = ("B_precise", "D_balanced", "E_warm", "C_creative")  # T = 0.0 / 0.3 / 0.7 / 1.15
WORD_LIMITS = {"P1": 120, "P2": None, "P3": None, "P4": 60, "P5": 25, "P6": 60}
# Подсчёт слов в рассуждениях бывает явной проверкой («Word count check: 39 words»)
# или пословным перечислением («1. Уважаемые / 2. коллеги!», «14(1) января(2)», «1-В 2-03:12»).
_WORD_COUNT_MENTION = re.compile(r"word count|count(?:ing)? (?:the )?words|подсч[её]т слов", re.I)
_ENUMERATED_WORDS = (
    re.compile(r"^\s*\d{1,3}\.\s+\S+\s*$", re.M),
    re.compile(r"\S\(\d{1,3}\)"),
    re.compile(r"(?<![\w-])\d{1,3}-[А-Яа-яЁёA-Za-z(«\"]"),  # «1-В 2-03:12», но не артикул TK3-220-RU
)
MIN_ENUMERATED = 10


def counts_words(reasoning: str) -> bool:
    if _WORD_COUNT_MENTION.search(reasoning):
        return True
    return any(len(set(pattern.findall(reasoning))) >= MIN_ENUMERATED for pattern in _ENUMERATED_WORDS)


FORCED_STOP_MARKER = "I have to answer now"
SHORT_NAMES = {
    "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B",
    "google/gemma-4-12b-qat": "Gemma-4-12B-QAT",
    "mistralai/ministral-3-14b-reasoning": "Ministral-3-14B",
}
PROMPT_ORDER = tuple(case.id for case in ALL_PROMPTS)
PRESET_ORDER = tuple(preset.key for preset in ALL_PRESETS)
TEMPERATURE = {p.key: p.params.temperature for p in ALL_PRESETS}


# ---------------------------------------------------------------------------
# Загрузка и разметка
# ---------------------------------------------------------------------------

def load_all(results_root: Path) -> dict[str, tuple[Record, ...]]:
    by_model = {}
    for model_dir in iter_model_dirs(results_root):
        records = load_runs(model_dir / RUNS_FILENAME)
        by_model[records[0]["model"]] = records
    order = list(SHORT_NAMES)
    return dict(sorted(by_model.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))


def load_critical_issues(results_root: Path) -> dict[tuple[str, str], list[str]]:
    path = results_root / "manual_review.json"
    if not path.exists():
        print(f"[warn] {path.name} не найден — ручной разбор не учитывается")
        return {}
    issues: dict[tuple[str, str], list[str]] = {}
    for issue in json.loads(path.read_text(encoding="utf-8"))["issues"]:
        if issue["severity"] != "critical":
            continue
        for run_id in issue["run_ids"]:
            issues.setdefault((issue["model"], run_id), []).append(issue["category"])
    return issues


def check_passed(record: Record, name: str) -> bool:
    return any(c["name"] == name and c["passed"] for c in record.get("checks", []))


def detail_has(record: Record, marker: str) -> bool:
    return any(marker in str(c.get("detail", "")) for c in record.get("checks", []))


def is_acceptable(record: Record, critical: dict[tuple[str, str], list[str]]) -> bool:
    if record.get("error") or not (record.get("answer") or "").strip():
        return False
    if (record["model"], record["run_id"]) in critical:
        return False
    return all(check_passed(record, name) for name in KEY_CHECKS[record["prompt_id"]])


# ---------------------------------------------------------------------------
# Примитивы статистики
# ---------------------------------------------------------------------------

def share(records: Iterable[Record], predicate: Callable[[Record], bool]) -> float | None:
    items = list(records)
    return sum(1 for r in items if predicate(r)) / len(items) if items else None


def mean_of(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.fmean(clean) if clean else None


def cv(values: list[float]) -> float | None:
    """Коэффициент вариации: разброс относительно среднего (0 = идентично)."""
    if len(values) < 2 or statistics.fmean(values) == 0:
        return None
    return statistics.pstdev(values) / statistics.fmean(values)


def metric(record: Record, name: str) -> Any:
    return record.get("metrics", {}).get(name)


def cells(records: Iterable[Record]) -> dict[tuple[str, str], list[Record]]:
    grouped: dict[tuple[str, str], list[Record]] = {}
    for record in records:
        grouped.setdefault((record["prompt_id"], record["preset_key"]), []).append(record)
    return grouped


def pairwise_similarity(answers: list[str]) -> float | None:
    return mean_pairwise_similarity(tuple(a.strip() for a in answers))


def unlimited(records: Iterable[Record]) -> list[Record]:
    return [r for r in records if r["preset_key"] != LENGTH_LIMITED_PRESET]


# ---------------------------------------------------------------------------
# Разделы анализа
# ---------------------------------------------------------------------------

def overview(by_model: dict[str, tuple[Record, ...]], critical: dict) -> list[dict[str, Any]]:
    rows = []
    for model, records in by_model.items():
        base = unlimited(records)
        latencies = [metric(r, "latency_s") for r in base]
        rows.append({
            "model": model,
            "acceptable": share(base, lambda r: is_acceptable(r, critical)),
            "score_v2": mean_of(r["quality_score"] for r in base),
            "score_v1": mean_of(r.get("quality_score_v1") for r in base),
            "latency_mean": mean_of(latencies),
            "latency_median": statistics.median(latencies),
            "total_hours": sum(metric(r, "latency_s") for r in records) / 3600,
            "tps": statistics.median([metric(r, "output_tps") for r in records if metric(r, "output_tps")]),
            "completion_tokens": mean_of(metric(r, "completion_tokens") for r in base),
            "reasoning_tokens": mean_of(metric(r, "reasoning_tokens_est") for r in base),
            "answer_tokens": mean_of(metric(r, "answer_tokens_est") for r in base),
            "reasoning_time_share": mean_of(metric(r, "reasoning_time_share") for r in base),
            "hallucinations": sum(1 for r in records if detail_has(r, HALLUCINATION_MARKER)),
            "code_switching": sum(1 for r in records if detail_has(r, CODE_SWITCH_MARKER)),
            "critical_manual": sum(1 for r in records if (model, r["run_id"]) in critical),
        })
    return rows


def acceptability_matrix(by_model: dict, critical: dict, key: str, order: tuple[str, ...],
                         only_unlimited: bool) -> dict[str, dict[str, float | None]]:
    matrix = {}
    for model, records in by_model.items():
        pool = unlimited(records) if only_unlimited else list(records)
        matrix[model] = {
            item: share([r for r in pool if r[key] == item], lambda r: is_acceptable(r, critical))
            for item in order
        }
    return matrix


def temperature_sweep(by_model: dict, critical: dict) -> dict[str, list[dict[str, Any]]]:
    sweep = {}
    for model, records in by_model.items():
        grouped = cells(records)
        points = []
        for preset in SWEEP_PRESETS:
            preset_cells = [grouped[(pid, preset)] for pid in PROMPT_ORDER if (pid, preset) in grouped]
            points.append({
                "preset": preset,
                "temperature": TEMPERATURE[preset],
                "acceptable": share([r for c in preset_cells for r in c], lambda r: is_acceptable(r, critical)),
                "identical_cells": share(preset_cells, lambda c: len({r["answer"].strip() for r in c}) == 1),
                "similarity": mean_of(pairwise_similarity([r["answer"] for r in c]) for c in preset_cells),
                "tokens_cv": mean_of(cv([metric(r, "completion_tokens") for r in c]) for c in preset_cells),
            })
        sweep[model] = points
    return sweep


def token_spread(by_model: dict) -> dict[str, dict[str, dict[str, Any]]]:
    spread = {}
    for model, records in by_model.items():
        per_prompt = {}
        for pid in PROMPT_ORDER:
            group = [r for r in unlimited(records) if r["prompt_id"] == pid]
            tokens = [metric(r, "completion_tokens") for r in group]
            per_prompt[pid] = {
                "median": statistics.median(tokens),
                "min": min(tokens),
                "max": max(tokens),
                "answer_median": statistics.median(metric(r, "answer_tokens_est") for r in group),
                "reasoning_median": statistics.median(metric(r, "reasoning_tokens_est") for r in group),
                "cell_cv": mean_of(cv([metric(r, "completion_tokens") for r in c])
                                   for (p, _), c in cells(group).items() if p == pid),
                "tokens": tokens,
            }
        spread[model] = per_prompt
    return spread


def word_limit_effect(by_model: dict) -> dict[str, dict[str, dict[str, Any]]]:
    effect = {}
    for model, records in by_model.items():
        if not any(r.get("reasoning") for r in records):
            continue
        per_prompt = {}
        for pid in PROMPT_ORDER:
            group = [r for r in unlimited(records) if r["prompt_id"] == pid and r.get("reasoning")]
            per_prompt[pid] = {
                "word_limit": WORD_LIMITS[pid],
                "reasoning_median": statistics.median(metric(r, "reasoning_tokens_est") for r in group),
                "counting_share": share(group, lambda r: counts_words(r["reasoning"])),
                "forced_stops": sum(1 for r in group if FORCED_STOP_MARKER in r["reasoning"]),
            }
        effect[model] = per_prompt
    return effect


def budget_effect(by_model: dict, critical: dict) -> list[dict[str, Any]]:
    rows = []
    for model, records in by_model.items():
        group = [r for r in records if r["preset_key"] == LENGTH_LIMITED_PRESET]
        rows.append({
            "model": model,
            "runs": len(group),
            "truncated": sum(1 for r in group if metric(r, "truncated")),
            "empty": sum(1 for r in group if not (r.get("answer") or "").strip()),
            "acceptable": share(group, lambda r: is_acceptable(r, critical)),
            "latency_mean": mean_of(metric(r, "latency_s") for r in group),
        })
    return rows


def determinism(by_model: dict) -> list[dict[str, Any]]:
    rows = []
    for model, records in by_model.items():
        greedy = [c for (_, preset), c in cells(records).items() if preset == "B_precise"]
        baseline = [c for (_, preset), c in cells(records).items() if preset == "A_baseline"]
        rows.append({
            "model": model,
            "greedy_identical_answers": sum(1 for c in greedy if len({r["answer"] for r in c}) == 1),
            "greedy_identical_tokens": sum(1 for c in greedy if len({metric(r, "completion_tokens") for r in c}) == 1),
            "greedy_cells": len(greedy),
            "baseline_similarity": mean_of(pairwise_similarity([r["answer"] for r in c]) for c in baseline),
            "baseline_identical": sum(1 for c in baseline if len({r["answer"].strip() for r in c}) == 1),
        })
    return rows


def build_analysis(results_root: Path) -> dict[str, Any]:
    by_model = load_all(results_root)
    if not by_model:
        raise ValueError(f"В {results_root} нет результатов")
    critical = load_critical_issues(results_root)
    return {
        "generated_at": utc_now_iso(),
        "models": list(by_model),
        "short_names": {m: SHORT_NAMES.get(m, m) for m in by_model},
        "key_checks": KEY_CHECKS,
        "overview": overview(by_model, critical),
        "by_prompt": acceptability_matrix(by_model, critical, "prompt_id", PROMPT_ORDER, only_unlimited=True),
        "by_preset": acceptability_matrix(by_model, critical, "preset_key", PRESET_ORDER, only_unlimited=False),
        "temperature_sweep": temperature_sweep(by_model, critical),
        "token_spread": token_spread(by_model),
        "word_limit_effect": word_limit_effect(by_model),
        "budget_effect": budget_effect(by_model, critical),
        "determinism": determinism(by_model),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f} %"


def render_markdown(a: dict[str, Any]) -> str:
    names = a["short_names"]
    lines = [
        "# Эмпирический анализ: сводные таблицы",
        "",
        f"Сформировано: {a['generated_at']}. Генерируется `python source/main.py analyze`; "
        "интерпретация — в README.md.",
        "",
        "«Приемлемый ответ» — не пуст, пройдены ключевые автопроверки задачи, ручной разбор "
        "(`manual_review.json`) не нашёл критической проблемы. Везде, где не оговорено, "
        f"пресет `{LENGTH_LIMITED_PRESET}` исключён: он проверяет обрезку, а не качество.",
        "",
        "## Общая сводка (без G_budget, 108 прогонов на модель)",
        "",
        *table(
            ["Модель", "приемлемо", "score v2", "score v1", "лат. ср., с", "лат. медиана, с", "всего, ч",
             "ток/с", "ток. вывода", "рассужд. (оц.)", "ответ (оц.)", "% времени на рассужд.",
             "галл. (авто)", "смеш. языков", "критич. (ручн.)"],
            [[names[o["model"]], pct(o["acceptable"]), fmt(o["score_v2"], 3), fmt(o["score_v1"], 3),
              fmt(o["latency_mean"], 1), fmt(o["latency_median"], 1), fmt(o["total_hours"], 2),
              fmt(o["tps"], 1), fmt(o["completion_tokens"], 0), fmt(o["reasoning_tokens"], 0),
              fmt(o["answer_tokens"], 0), pct(o["reasoning_time_share"]), o["hallucinations"],
              o["code_switching"], o["critical_manual"]] for o in a["overview"]],
        ),
        "## Доля приемлемых ответов по задачам",
        "",
        *table(["Модель", *PROMPT_ORDER],
               [[names[m], *[pct(v) for v in row.values()]] for m, row in a["by_prompt"].items()]),
        "## Доля приемлемых ответов по пресетам (все 6 промптов)",
        "",
        *table(["Модель", *PRESET_ORDER],
               [[names[m], *[pct(v) for v in row.values()]] for m, row in a["by_preset"].items()]),
        "## Температурный свип (B → D → E → C)",
        "",
        *table(
            ["Модель", "пресет", "T", "приемлемо", "ячеек с идентичными ответами", "схожесть повторов",
             "CV токенов вывода"],
            [[names[m], p["preset"], fmt(p["temperature"], 2), pct(p["acceptable"]), pct(p["identical_cells"]),
              fmt(p["similarity"], 3), fmt(p["tokens_cv"], 3)]
             for m, points in a["temperature_sweep"].items() for p in points],
        ),
        "## Разброс токенов вывода по задачам (без G_budget, 18 прогонов на ячейку)",
        "",
        *table(
            ["Модель", "промпт", "медиана", "мин", "макс", "макс/мин", "рассужд. медиана", "ответ медиана",
             "CV внутри повторов"],
            [[names[m], pid, fmt(s["median"], 0), s["min"], s["max"], fmt(s["max"] / max(1, s["min"]), 1),
              fmt(s["reasoning_median"], 0), fmt(s["answer_median"], 0), fmt(s["cell_cv"], 3)]
             for m, prompts in a["token_spread"].items() for pid, s in prompts.items()],
        ),
        "## Лимит в словах и длина рассуждений",
        "",
        *table(
            ["Модель", "промпт", "лимит слов в задании", "рассужд. медиана, ток.", "трасс с подсчётом слов",
             "принудительных остановок"],
            [[names[m], pid, s["word_limit"] or "—", fmt(s["reasoning_median"], 0), pct(s["counting_share"]),
              s["forced_stops"]] for m, prompts in a["word_limit_effect"].items() for pid, s in prompts.items()],
        ),
        "## Пресет G_budget (max_tokens = 256)",
        "",
        *table(["Модель", "прогонов", "обрезано", "пустых ответов", "приемлемо", "лат. ср., с"],
               [[names[b["model"]], b["runs"], b["truncated"], b["empty"], pct(b["acceptable"]),
                 fmt(b["latency_mean"], 1)] for b in a["budget_effect"]]),
        "## Детерминизм",
        "",
        *table(
            ["Модель", "T=0: идентичные ответы", "T=0: идентичное число токенов", "базовый: схожесть повторов",
             "базовый: идентичные ответы"],
            [[names[d["model"]], f"{d['greedy_identical_answers']}/{d['greedy_cells']}",
              f"{d['greedy_identical_tokens']}/{d['greedy_cells']}", fmt(d["baseline_similarity"], 3),
              f"{d['baseline_identical']}/{d['greedy_cells']}"] for d in a["determinism"]],
        ),
    ]
    return "\n".join(lines) + "\n"


def write_analysis(results_root: Path) -> tuple[Path, Path]:
    """Построить анализ и записать analysis.md и analysis.json."""
    analysis = build_analysis(results_root)
    json_path = results_root / "analysis.json"
    md_path = results_root / "analysis.md"
    write_text(json_path, json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    write_text(md_path, render_markdown(analysis))
    return md_path, json_path
