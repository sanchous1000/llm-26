"""Сводный отчёт-сравнение по всем моделям, для которых есть результаты.

Запуск:  python source/main.py compare [--results DIR]
"""

from __future__ import annotations

from pathlib import Path

from llm_lab.evaluation.checks import CODE_SWITCH_MARKER
from llm_lab.reporting.markdown import fmt, table
from llm_lab.reporting.stats import (
    Record,
    group_by,
    has_hallucination,
    metric,
    ok_records,
    ordered_preset_keys,
    ordered_prompt_ids,
    stability,
    summarize,
)
from llm_lab.storage import (
    COMPARISON_FILENAME,
    RUNS_FILENAME,
    iter_model_dirs,
    load_runs,
    utc_now_iso,
    write_text,
)


def collect_by_model(results_root: Path) -> dict[str, tuple[Record, ...]]:
    collected: dict[str, tuple[Record, ...]] = {}
    for model_dir in iter_model_dirs(results_root):
        try:
            records = load_runs(model_dir / RUNS_FILENAME)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[warn] {model_dir.name}: {exc}")
            continue
        collected[records[0]["model"]] = records
    return collected


def _overall_table(by_model: dict[str, tuple[Record, ...]]) -> list[str]:
    rows = []
    for model, records in by_model.items():
        s = summarize(model, records)
        rows.append([
            f"`{model}`",
            s.runs,
            fmt(s.quality, 3),
            fmt(s.completion_tokens, 0),
            fmt(s.latency_s, 2),
            fmt(s.output_tps, 1),
            f"{s.truncated}/{s.runs}",
            f"{s.hallucinations}/{s.runs}",
            s.errors,
        ])
    return table(
        ["Модель", "прогонов", "score", "ток. вывода", "лат., с", "ток/с", "обрезано", "галл.", "ошибок"],
        rows,
    )


def _by_prompt_table(by_model: dict[str, tuple[Record, ...]]) -> list[str]:
    prompt_ids: list[str] = []
    for records in by_model.values():
        for pid in ordered_prompt_ids(records):
            if pid not in prompt_ids:
                prompt_ids.append(pid)
    rows = []
    for model, records in by_model.items():
        grouped = group_by(records, lambda r: r["prompt_id"])
        row: list[object] = [f"`{model}`"]
        for pid in prompt_ids:
            group = grouped.get(pid, ())
            row.append(fmt(summarize(pid, group).quality, 2) if group else "—")
        rows.append(row)
    return table(["Модель", *[f"`{p}`" for p in prompt_ids]], rows)


def _by_mode_table(by_model: dict[str, tuple[Record, ...]]) -> list[str]:
    """Базовый против тюнинга. Пресеты с лимитом max_tokens вынесены отдельно:
    у reasoning-моделей они обнуляют ответы и маскируют влияние остальных параметров."""
    rows = []
    for model, records in by_model.items():
        base = summarize("baseline", [r for r in records if r["mode"] == "baseline"])
        tuned = summarize("tuned", [r for r in records if r["mode"] == "tuned" and not _limits_length(r)])
        limited = summarize("limited", [r for r in records if _limits_length(r)])
        delta = tuned.quality - base.quality if tuned.quality is not None and base.quality is not None else None
        rows.append([
            f"`{model}`",
            fmt(base.quality, 3),
            fmt(base.completion_tokens, 0),
            fmt(tuned.quality, 3),
            fmt(tuned.completion_tokens, 0),
            "—" if delta is None else f"{delta:+.3f}",
            fmt(limited.quality, 3),
            f"{limited.reasoning_cut}/{limited.runs}",
        ])
    return table(
        ["Модель", "score базовый", "ток. базовый", "score тюнинг (без лимита)", "ток. тюнинг", "Δscore",
         "score с max_tokens", "ответ съеден рассужд."],
        rows,
    )


def _limits_length(record: Record) -> bool:
    return (record.get("params") or {}).get("max_tokens") is not None


def _determinism_table(by_model: dict[str, tuple[Record, ...]]) -> list[str]:
    rows = []
    for model, records in by_model.items():
        cells = group_by(records, lambda r: (r["prompt_id"], r["preset_key"], r["mode"]))
        per_mode: dict[str, list[float]] = {"baseline": [], "tuned": []}
        identical: dict[str, list[int]] = {"baseline": [], "tuned": []}
        for (_, _, mode), group in cells.items():
            st = stability(group)
            if st is None or st.repeats < 2:
                continue
            if st.mean_similarity is not None:
                per_mode[mode].append(st.mean_similarity)
            identical[mode].append(1 if st.identical else 0)
        rows.append([
            f"`{model}`",
            fmt(sum(per_mode["baseline"]) / len(per_mode["baseline"]), 3) if per_mode["baseline"] else "—",
            f"{sum(identical['baseline'])}/{len(identical['baseline'])}" if identical["baseline"] else "—",
            fmt(sum(per_mode["tuned"]) / len(per_mode["tuned"]), 3) if per_mode["tuned"] else "—",
            f"{sum(identical['tuned'])}/{len(identical['tuned'])}" if identical["tuned"] else "—",
        ])
    return table(
        ["Модель", "схожесть (базовый)", "идентичны (базовый)", "схожесть (тюнинг)", "идентичны (тюнинг)"],
        rows,
    )


def _hallucination_table(by_model: dict[str, tuple[Record, ...]]) -> list[str]:
    rows = []
    for model, records in by_model.items():
        grouped = group_by(ok_records(records), lambda r: r["prompt_id"])
        row: list[object] = [f"`{model}`"]
        for pid in ("P1", "P3", "P4", "P5", "P6"):
            group = grouped.get(pid, ())
            row.append(f"{sum(1 for r in group if has_hallucination(r))}/{len(group)}" if group else "—")
        switched = sum(1 for r in records if any(CODE_SWITCH_MARKER in str(c.get("detail")) for c in r.get("checks", [])))
        row.append(f"{switched}/{len(records)}")
        rows.append(row)
    return table(
        ["Модель", "P1 (письмо)", "P3 (извлечение)", "P4 (суммаризация)", "P5 (расчёт)", "P6 (ловушка)",
         "смешение языков"],
        rows,
    )


def _speed_table(by_model: dict[str, tuple[Record, ...]]) -> list[str]:
    rows = []
    for model, records in by_model.items():
        good = ok_records(records)
        latencies = sorted(metric(r, "latency_s") or 0.0 for r in good)
        if not latencies:
            continue
        p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
        s = summarize(model, records)
        share = s.reasoning_time_share * 100 if s.reasoning_time_share is not None else None
        rows.append([
            f"`{model}`",
            fmt(s.prompt_tokens, 0),
            fmt(s.completion_tokens, 0),
            fmt(s.reasoning_tokens, 0),
            fmt(s.answer_tokens, 0),
            fmt(s.latency_p50, 1),
            fmt(p95, 1),
            fmt(s.ttft_s, 2),
            fmt(s.ttfa_s, 1),
            fmt(share, 0),
            fmt(s.output_tps, 1),
            ", ".join(s.tps_methods) or "—",
        ])
    return table(
        ["Модель", "ток. ввода", "ток. вывода", "из них рассужд. (оц.)", "ответа (оц.)", "лат. p50, с",
         "лат. p95, с", "TTFT, с", "до ответа, с", "% на рассужд.", "ток/с", "метод ток/с"],
        rows,
    )


def build_comparison(by_model: dict[str, tuple[Record, ...]]) -> str:
    lines = [
        "# Сводное сравнение моделей (лабораторная №1)",
        "",
        f"Сформировано: {utc_now_iso()}",
        "",
        f"Моделей в сравнении: **{len(by_model)}**. "
        "Отчёты по каждой модели лежат в `results/<модель>/report.md`, "
        "полные ответы — в `results/<модель>/answers/`.",
        "",
        "## 1. Общая сводка",
        "",
        *_overall_table(by_model),
        "## 2. Качество по типам задач (средний балл)",
        "",
        *_by_prompt_table(by_model),
        "## 3. Базовый режим против тюнинга",
        "",
        *_by_mode_table(by_model),
        "## 4. Стабильность повторов",
        "",
        *_determinism_table(by_model),
        "## 5. Галлюцинации и смешение языков (автопроверки v2)",
        "",
        *_hallucination_table(by_model),
        "## 6. Производительность",
        "",
        *_speed_table(by_model),
        "## 7. Покрытие пресетов",
        "",
        *table(
            ["Модель", "Пресеты"],
            [[f"`{m}`", ", ".join(f"`{k}`" for k in ordered_preset_keys(r))] for m, r in by_model.items()],
        ),
    ]
    return "\n".join(lines) + "\n"


def write_comparison(results_root: Path) -> tuple[Path, tuple[str, ...]]:
    """Собрать результаты всех моделей и записать comparison.md; вернуть путь и список моделей."""
    by_model = collect_by_model(results_root)
    if not by_model:
        raise FileNotFoundError(f"В {results_root} нет результатов. Сначала запустите `python source/main.py run`.")
    output = results_root / COMPARISON_FILENAME
    write_text(output, build_comparison(by_model))
    return output, tuple(by_model)
