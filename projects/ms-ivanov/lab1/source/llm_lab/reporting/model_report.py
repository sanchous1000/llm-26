"""Генерация markdown-отчёта по одной модели (по всем тестам)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_lab.evaluation.text import truncate
from llm_lab.reporting.markdown import fmt, table
from llm_lab.reporting.stats import (
    Record,
    best_run,
    check_pass_rates,
    group_by,
    hallucination_details,
    has_hallucination,
    metric,
    ok_records,
    ordered_preset_keys,
    ordered_prompt_ids,
    stability,
    summarize,
    worst_run,
)
from llm_lab.storage import ModelPaths, answer_filename, load_runs, write_text

EXCERPT_LIMIT = 700


def _load_manifest(paths: ModelPaths) -> dict[str, Any]:
    if not paths.manifest_file.exists():
        return {}
    try:
        return json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[warn] манифест не прочитан: {exc}")
        return {}


def build_report(records: tuple[Record, ...], manifest: dict[str, Any]) -> str:
    model = records[0]["model"]
    preset_keys = ordered_preset_keys(records)
    prompt_ids = ordered_prompt_ids(records)
    preset_labels = {r["preset_key"]: r["preset_label"] for r in records}
    prompt_titles = {r["prompt_id"]: r["prompt_title"] for r in records}

    lines: list[str] = []
    lines += _header(model, records, manifest)
    lines += _presets_section(records, preset_keys, preset_labels)
    lines += _modes_summary(records, preset_keys, preset_labels)
    lines += _matrix_section(records, prompt_ids, preset_keys, prompt_titles)
    lines += _stability_section(records, prompt_ids, preset_keys)
    lines += _hallucination_section(records)
    lines += _per_prompt_sections(records, prompt_ids, preset_keys)
    lines += _errors_section(records)
    lines += _observations(records, preset_keys, preset_labels)
    return "\n".join(lines) + "\n"


def _header(model: str, records: tuple[Record, ...], manifest: dict[str, Any]) -> list[str]:
    good = ok_records(records)
    failed = len(records) - len(good)
    total_out = sum(metric(r, "completion_tokens") or 0 for r in good)
    total_in = sum(metric(r, "prompt_tokens") or 0 for r in good)
    total_time = sum(metric(r, "latency_s") or 0.0 for r in good)
    total_reasoning = sum(metric(r, "reasoning_tokens_est") or 0 for r in good)
    estimated = any(metric(r, "tokens_estimated") for r in good)

    lines = [
        f"# Отчёт по модели `{model}`",
        "",
        "## 1. Параметры эксперимента",
        "",
    ]
    lines += table(
        ["Параметр", "Значение"],
        [
            ["Модель", f"`{model}`"],
            ["Эндпоинт", manifest.get("endpoint", records[0].get("endpoint", "—"))],
            ["Профиль пресетов", manifest.get("profile", "—")],
            ["Запуск начат", manifest.get("started_at", records[0]["timestamp"])],
            ["Последний прогон", manifest.get("finished_at") or max(r["timestamp"] for r in records)],
            ["Промптов", len(ordered_prompt_ids(records))],
            ["Пресетов параметров", len(ordered_preset_keys(records))],
            ["Повторов на конфигурацию", records[0].get("repeats_total", "—")],
            ["Всего прогонов", len(records)],
            ["Из них с ошибкой запроса", failed],
            ["Режим стриминга", "да (TTFT измеряется)" if records[0].get("stream") else "нет"],
            ["Суммарно токенов ввода", total_in],
            ["Суммарно токенов вывода", total_out],
            ["из них на рассуждения (оценка)", total_reasoning],
            ["Суммарное время генерации, с", f"{total_time:.1f} ({total_time / 3600:.2f} ч)"],
            ["Источник счётчика токенов", "оценка по символам" if estimated else "поле usage сервера"],
        ],
    )
    lines += _sessions_block(manifest)
    return lines


def _sessions_block(manifest: dict[str, Any]) -> list[str]:
    """История сессий, если результаты склеены из нескольких запусков (команда merge)."""
    sessions = manifest.get("sessions") or []
    if not sessions:
        return []
    lines = [
        f"Результаты собраны из {len(sessions)} сессий запуска (`main.py merge`):",
        "",
        *table(
            ["Источник", "Промпты", "Записей", "Использовано", "Первый прогон", "Последний прогон"],
            [
                [
                    f"`{s['source']}`",
                    ", ".join(s.get("prompts", [])),
                    s.get("records_loaded"),
                    s.get("records_used"),
                    s.get("first_run_at"),
                    s.get("last_run_at"),
                ]
                for s in sessions
            ],
        ),
    ]
    for dup in manifest.get("dropped_duplicates") or []:
        lines += [
            f"- Отброшен дубль `{dup['run_id']}` из `{dup['source']}` ({dup['timestamp']}, "
            f"score {dup.get('quality_score')}) — заменён записью из `{dup['replaced_by']}`, "
            "чтобы все повторы конфигурации были из одной сессии.",
        ]
    return lines + [""] if manifest.get("dropped_duplicates") else lines


def _presets_section(
    records: tuple[Record, ...],
    preset_keys: tuple[str, ...],
    preset_labels: dict[str, str],
) -> list[str]:
    rows = []
    by_preset = group_by(records, lambda r: r["preset_key"])
    for key in preset_keys:
        sample = by_preset[key][0]
        rows.append([
            f"`{key}`",
            preset_labels[key],
            sample["mode"],
            f"`{sample['params_desc']}`",
        ])
    return ["## 2. Режимы генерации (пресеты гиперпараметров)", "", *table(
        ["Ключ", "Название", "Режим", "Переданные параметры"], rows
    )]


def _modes_summary(
    records: tuple[Record, ...],
    preset_keys: tuple[str, ...],
    preset_labels: dict[str, str],
) -> list[str]:
    by_preset = group_by(records, lambda r: r["preset_key"])
    stats = [(key, summarize(preset_labels[key], by_preset[key])) for key in preset_keys]
    any_reasoning = any(s.reasoning_chars for _, s in stats)

    quality_rows = [
        [
            f"`{key}`",
            fmt(s.quality, 3),
            fmt(s.completion_tokens, 0),
            fmt(s.completion_tokens_sd, 0),
            f"{fmt(s.completion_tokens_min)}–{fmt(s.completion_tokens_max)}",
            fmt(s.reasoning_tokens, 0),
            fmt(s.answer_tokens, 0),
            fmt(s.words, 0),
            f"{s.truncated}/{s.runs}",
            f"{s.reasoning_cut}/{s.runs}",
            f"{s.hallucinations}/{s.runs}",
            s.errors,
        ]
        for key, s in stats
    ]
    speed_rows = [
        [
            f"`{key}`",
            fmt(s.latency_s, 1),
            fmt(s.latency_p50, 1),
            fmt(s.ttft_s, 2),
            fmt(s.ttfa_s, 1),
            fmt(s.reasoning_time_share * 100 if s.reasoning_time_share is not None else None, 0),
            fmt(s.output_tps, 1),
            ", ".join(s.tps_methods) or "—",
        ]
        for key, s in stats
    ]

    lines = [
        "## 3. Сводка по режимам (все промпты)",
        "",
        "### 3.1. Качество и длина вывода",
        "",
        "`score` — средневзвешенный балл автопроверок в [0; 1]; "
        "`σ(tok)` — стандартное отклонение числа токенов вывода; "
        "`ток. рассужд.` / `ток. ответа` — оценка разделения вывода на рассуждения и видимый ответ; "
        "`съеден рассужд.` — прогоны без ответа, где лимит израсходован на рассуждения; "
        "`галл.` — прогоны с сработавшим детектором выдумок.",
        "",
        *table(
            ["Пресет", "score", "ток. вывода", "σ(tok)", "мин–макс", "ток. рассужд.", "ток. ответа",
             "слов", "обрезано", "съеден рассужд.", "галл.", "ошибок"],
            quality_rows,
        ),
        "### 3.2. Производительность",
        "",
        "`TTFT` — время до первого токена любого вида; `до ответа` — время до первого токена "
        "видимого ответа (у reasoning-моделей ≈ время рассуждений); `% на рассужд.` — доля времени "
        "запроса до начала ответа; `ток/с` — токены вывода за время декодирования "
        "(`decode` — без TTFT, `end_to_end` — по полному времени запроса, включая prefill).",
        "",
        *table(
            ["Пресет", "лат. ср., с", "лат. p50, с", "TTFT, с", "до ответа, с", "% на рассужд.",
             "ток/с", "метод ток/с"],
            speed_rows,
        ),
    ]
    if any_reasoning:
        lines += [
            "> Модель выдаёт блоки рассуждений. Они исключены из оценки качества и подсчёта слов, "
            "но входят в `completion_tokens` и во время генерации — при сравнении с обычными "
            "моделями длина вывода у этой модели завышена.",
            "",
        ]
    return lines


def _matrix_section(
    records: tuple[Record, ...],
    prompt_ids: tuple[str, ...],
    preset_keys: tuple[str, ...],
    prompt_titles: dict[str, str],
) -> list[str]:
    cells = group_by(records, lambda r: (r["prompt_id"], r["preset_key"]))

    quality_rows = []
    tokens_rows = []
    latency_rows = []
    for prompt_id in prompt_ids:
        q_row: list[Any] = [f"`{prompt_id}` {prompt_titles[prompt_id]}"]
        t_row: list[Any] = [f"`{prompt_id}`"]
        l_row: list[Any] = [f"`{prompt_id}`"]
        for preset_key in preset_keys:
            group = cells.get((prompt_id, preset_key), ())
            if not group:
                q_row.append("—")
                t_row.append("—")
                l_row.append("—")
                continue
            s = summarize(preset_key, group)
            q_row.append(fmt(s.quality, 2))
            t_row.append(fmt(s.completion_tokens, 0))
            l_row.append(fmt(s.latency_s, 1))
        quality_rows.append(q_row)
        tokens_rows.append(t_row)
        latency_rows.append(l_row)

    header_tail = [f"`{k}`" for k in preset_keys]
    return [
        "## 4. Матрицы «промпт × режим»",
        "",
        "### 4.1. Качество (средний балл автопроверок)",
        "",
        *table(["Промпт", *header_tail], quality_rows),
        "### 4.2. Длина ответа, токенов вывода (среднее)",
        "",
        *table(["Промпт", *header_tail], tokens_rows),
        "### 4.3. Время отклика, с (среднее)",
        "",
        *table(["Промпт", *header_tail], latency_rows),
    ]


def _stability_section(
    records: tuple[Record, ...],
    prompt_ids: tuple[str, ...],
    preset_keys: tuple[str, ...],
) -> list[str]:
    cells = group_by(records, lambda r: (r["prompt_id"], r["preset_key"]))
    rows = []
    for prompt_id in prompt_ids:
        for preset_key in preset_keys:
            group = cells.get((prompt_id, preset_key), ())
            st = stability(group)
            if st is None:
                continue
            rows.append([
                f"`{prompt_id}`",
                f"`{preset_key}`",
                st.repeats,
                st.distinct_answers,
                "да" if st.identical else "нет",
                fmt(st.mean_similarity, 3),
                fmt(st.quality_spread, 2),
                fmt(st.tokens_spread, 0),
            ])
    return [
        "## 5. Стабильность и детерминизм",
        "",
        "Повторы одной конфигурации: сколько получилось различных ответов, "
        "средняя попарная схожесть текстов и разброс метрик.",
        "",
        *table(
            ["Промпт", "Пресет", "повторов", "различных", "идентичны", "ср. схожесть", "Δscore", "Δток."],
            rows,
        ),
    ]


def _hallucination_section(records: tuple[Record, ...]) -> list[str]:
    flagged = [r for r in ok_records(records) if has_hallucination(r)]
    lines = [
        "## 6. Фактические ошибки и галлюцинации",
        "",
        "Детектор срабатывает на: числа, которых нет в исходном тексте; заполнение "
        "заведомо отсутствующих полей; пересказ несуществующего документа; выдуманные "
        "суммы и фамилии в деловом письме.",
        "",
    ]
    if not flagged:
        lines += ["Автоматические детекторы галлюцинаций не сработали ни на одном прогоне.", ""]
        return lines
    rows = []
    for record in flagged:
        rows.append([
            f"`{record['prompt_id']}`",
            f"`{record['preset_key']}`",
            f"r{record['repeat']}",
            "; ".join(hallucination_details(record))[:220],
        ])
    lines += table(["Промпт", "Пресет", "Повтор", "Что обнаружено"], rows)
    lines += [f"Итого прогонов с признаками выдумки: **{len(flagged)}** из {len(ok_records(records))}.", ""]
    return lines


def _per_prompt_sections(
    records: tuple[Record, ...],
    prompt_ids: tuple[str, ...],
    preset_keys: tuple[str, ...],
) -> list[str]:
    by_prompt = group_by(records, lambda r: r["prompt_id"])
    lines = ["## 7. Детально по промптам", ""]
    for prompt_id in prompt_ids:
        group = by_prompt[prompt_id]
        sample = group[0]
        lines += [
            f"### {prompt_id}. {sample['prompt_title']} (`{sample['prompt_type']}`)",
            "",
            "<details><summary>Текст промпта</summary>",
            "",
            "```text",
            (sample.get("system") or "(system не задан)"),
            "---",
            sample.get("user", ""),
            "```",
            "",
            "</details>",
            "",
        ]
        if sample.get("reference"):
            lines += [f"**Эталон:** {sample['reference']}", ""]

        rates = check_pass_rates(group)
        if rates:
            lines += ["**Прохождение проверок по всем режимам:**", ""]
            lines += table(
                ["Проверка", "Пройдено"],
                [[name, f"{passed}/{total}"] for name, (passed, total) in rates.items()],
            )

        lines += _prompt_preset_table(group, preset_keys)
        lines += _prompt_examples(group)
    return lines


def _prompt_preset_table(group: tuple[Record, ...], preset_keys: tuple[str, ...]) -> list[str]:
    by_preset = group_by(group, lambda r: r["preset_key"])
    rows = []
    for key in preset_keys:
        subset = by_preset.get(key, ())
        if not subset:
            continue
        s = summarize(key, subset)
        failed_checks = sorted({
            c["name"]
            for r in ok_records(subset)
            for c in r.get("checks", [])
            if not c.get("passed")
        })
        rows.append([
            f"`{key}`",
            fmt(s.quality, 2),
            fmt(s.completion_tokens, 0),
            fmt(s.latency_s, 1),
            f"{s.truncated}/{s.runs}",
            ", ".join(failed_checks) or "—",
        ])
    return table(["Пресет", "score", "ток.", "лат., с", "обрезано", "Проваленные проверки"], rows)


def _prompt_examples(group: tuple[Record, ...]) -> list[str]:
    best = best_run(group)
    worst = worst_run(group)
    lines: list[str] = []
    for title, record in (("Лучший ответ", best), ("Худший ответ", worst)):
        if record is None:
            continue
        if best is worst and title == "Худший ответ":
            continue
        file_ref = f"answers/{answer_filename(record['prompt_id'], record['preset_key'], record['repeat'])}"
        lines += [
            f"**{title}** — `{record['preset_key']}` / r{record['repeat']}, "
            f"score {fmt(record.get('quality_score'), 2)} "
            f"([полная карточка]({file_ref}))",
            "",
            "```text",
            truncate(record.get("answer") or "(пусто)", EXCERPT_LIMIT),
            "```",
            "",
        ]
    return lines


def _errors_section(records: tuple[Record, ...]) -> list[str]:
    failed = [r for r in records if r.get("error")]
    lines = ["## 8. Ошибки запросов", ""]
    if not failed:
        lines += ["Ошибок транспорта и HTTP не зафиксировано.", ""]
        return lines
    rows = [
        [f"`{r['prompt_id']}`", f"`{r['preset_key']}`", f"r{r['repeat']}", r.get("attempts", 1), r["error"][:200]]
        for r in failed
    ]
    lines += table(["Промпт", "Пресет", "Повтор", "Попыток", "Ошибка"], rows)
    return lines


def _observations(
    records: tuple[Record, ...],
    preset_keys: tuple[str, ...],
    preset_labels: dict[str, str],
) -> list[str]:
    """Автоматические наблюдения — фактура для раздела «Выводы» в README."""
    by_preset = group_by(records, lambda r: r["preset_key"])
    stats = {key: summarize(preset_labels[key], by_preset[key]) for key in preset_keys}
    notes: list[str] = []

    baseline = next((s for k, s in stats.items() if by_preset[k][0]["mode"] == "baseline"), None)
    tuned = {k: s for k, s in stats.items() if by_preset[k][0]["mode"] == "tuned"}

    if baseline and tuned:
        best_key = max(tuned, key=lambda k: tuned[k].quality or 0.0)
        best = tuned[best_key]
        if baseline.quality is not None and best.quality is not None:
            delta = best.quality - baseline.quality
            notes.append(
                f"Лучший тюнингованный режим `{best_key}` даёт score {best.quality:.3f} против "
                f"{baseline.quality:.3f} у базового ({delta:+.3f})."
            )
        if baseline.completion_tokens and best.completion_tokens:
            ratio = best.completion_tokens / baseline.completion_tokens
            notes.append(
                f"Средняя длина ответа: базовый {baseline.completion_tokens:.0f} ток., "
                f"`{best_key}` {best.completion_tokens:.0f} ток. (×{ratio:.2f})."
            )

    identical_by_preset: dict[str, tuple[int, int]] = {}
    cells = group_by(records, lambda r: (r["prompt_id"], r["preset_key"]))
    for (prompt_id, preset_key), group in cells.items():
        st = stability(group)
        if st is None or st.repeats < 2:
            continue
        done, total = identical_by_preset.get(preset_key, (0, 0))
        identical_by_preset[preset_key] = (done + (1 if st.identical else 0), total + 1)
    for key in preset_keys:
        if key in identical_by_preset:
            done, total = identical_by_preset[key]
            notes.append(f"Детерминизм `{key}`: полностью совпали повторы в {done} из {total} ячеек.")

    truncated_total = sum(s.truncated for s in stats.values())
    if truncated_total:
        worst_key = max(stats, key=lambda k: stats[k].truncated)
        notes.append(
            f"Прогонов, обрезанных по max_tokens: {truncated_total}; чаще всего в `{worst_key}` "
            f"({stats[worst_key].truncated} шт.)."
        )

    reasoning_cut_total = sum(s.reasoning_cut for s in stats.values())
    if reasoning_cut_total:
        worst_key = max(stats, key=lambda k: stats[k].reasoning_cut)
        notes.append(
            f"**ВНИМАНИЕ.** Прогонов, где лимит длины израсходован блоком рассуждений и видимого "
            f"ответа не осталось: {reasoning_cut_total} (чаще всего `{worst_key}`). Нулевой балл "
            f"в этих ячейках отражает неудачный max_tokens, а не качество модели — "
            f"поднимите `--budget-max-tokens` и перезапустите."
        )

    hallucinations = sum(s.hallucinations for s in stats.values())
    if hallucinations:
        worst_key = max(stats, key=lambda k: stats[k].hallucinations)
        notes.append(
            f"Прогонов с признаками галлюцинаций: {hallucinations}; максимум в `{worst_key}` "
            f"({stats[worst_key].hallucinations} шт.)."
        )
    else:
        notes.append("Признаков галлюцинаций автопроверки не выявили.")

    speeds = {k: s.output_tps for k, s in stats.items() if s.output_tps}
    if speeds:
        fastest = max(speeds, key=lambda k: speeds[k])
        notes.append(
            f"Скорость генерации: {min(speeds.values()):.1f}–{max(speeds.values()):.1f} ток/с "
            f"(быстрее всего `{fastest}`)."
        )

    return ["## 9. Автоматические наблюдения", "", *[f"- {n}" for n in notes], ""]


def generate_report(paths: ModelPaths) -> Path:
    """Считать runs.jsonl и записать report.md рядом с ним."""
    records = load_runs(paths.runs_file)
    content = build_report(records, _load_manifest(paths))
    write_text(paths.report_file, content)
    return paths.report_file
