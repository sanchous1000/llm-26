#!/usr/bin/env python3
"""Сводит метрики и проводит простую автоматическую проверку качества ответов."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "results"


def extract_json(text: str) -> Any | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).lower().replace("ё", "е")).strip()


def is_strict_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text.strip()), dict)
    except json.JSONDecodeError:
        return False


def failed_checks(checks: dict[str, bool]) -> list[str]:
    return [name for name, passed in checks.items() if not passed]


def evaluate_p1(text: str) -> dict[str, Any]:
    normalized = normalize(text)
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]
    checks = {
        "название": "практика локальных llm" in normalized,
        "дата": "15 ноября 2026" in normalized,
        "адрес": "49" in normalized and "кронверк" in normalized,
        "бесплатно": "бесплат" in normalized,
        "регистрация": "10 ноября" in normalized,
        "150 мест": "150" in normalized,
        "заголовок и один абзац": len(blocks) == 2,
    }
    words = re.findall(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)?", text)
    checks["90–120 слов"] = 90 <= len(words) <= 120
    return {
        "score": round(sum(checks.values()) / len(checks), 3),
        "checks": checks,
        "failed_checks": failed_checks(checks),
        "word_count": len(words),
        "valid_json": None,
    }


def evaluate_p2(text: str) -> dict[str, Any]:
    parsed = extract_json(text)
    expected = {
        "A": "оплата",
        "B": "техподдержка",
        "C": "продажи",
        "D": "аккаунт",
    }
    checks = {
        key: isinstance(parsed, dict) and normalize(parsed.get(key)) == label
        for key, label in expected.items()
    }
    checks["формат: только JSON"] = is_strict_json_object(text)
    return {
        "score": round(sum(checks.values()) / len(checks), 3),
        "checks": checks,
        "failed_checks": failed_checks(checks),
        "word_count": len(text.split()),
        "valid_json": isinstance(parsed, dict),
    }


def evaluate_p3(text: str) -> dict[str, Any]:
    parsed = extract_json(text)
    data = parsed if isinstance(parsed, dict) else {}
    materials = normalize(data.get("materials"))
    features = normalize(data.get("features"))
    glass_ok = "стекл" in materials or "glass" in materials
    stainless_ok = "нержав" in materials or "stainless" in materials
    auto_off_ok = "автоотключ" in features or "auto-off" in features
    dry_run_ok = "без вод" in features or "dry" in features
    checks = {
        "name": "nordic kettle x2" in normalize(data.get("name")),
        "volume_l": normalize(data.get("volume_l")).replace(",", ".") in {"1.7", "1.7 l", "1.7 л"},
        "power_w": "2200" in normalize(data.get("power_w")),
        "materials": glass_ok and stainless_ok,
        "features": auto_off_ok and dry_run_ok,
        "price_rub": "4590" in normalize(data.get("price_rub")),
        "warranty_months": "24" in normalize(data.get("warranty_months")),
        "smartphone_control": data.get("smartphone_control") is False
        or normalize(data.get("smartphone_control")) in {"нет", "false"},
        "summary": isinstance(data.get("summary"), str)
        and 1 <= len(data["summary"].split()) <= 25,
        "формат: только JSON": is_strict_json_object(text),
    }
    return {
        "score": round(sum(checks.values()) / len(checks), 3),
        "checks": checks,
        "failed_checks": failed_checks(checks),
        "word_count": len(text.split()),
        "valid_json": isinstance(parsed, dict),
    }


def evaluate(record: dict[str, Any]) -> dict[str, Any]:
    prompt_id = record["prompt"]["id"]
    evaluation = {
        "P1": evaluate_p1,
        "P2": evaluate_p2,
        "P3": evaluate_p3,
    }[prompt_id](record["response"])
    evaluation["checks"]["ответ не обрезан"] = record.get("finish_reason") != "length"
    evaluation["failed_checks"] = failed_checks(evaluation["checks"])
    evaluation["score"] = round(
        sum(evaluation["checks"].values()) / len(evaluation["checks"]), 3
    )
    return evaluation


def mean(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(statistics.mean(present), 3) if present else None


def markdown_escape(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


ERROR_LABELS = {
    "P1": {
        "название": "нет точного названия",
        "дата": "нет даты",
        "адрес": "нет адреса",
        "бесплатно": "не найдено указание «бесплатно»",
        "регистрация": "нет срока регистрации",
        "150 мест": "нет числа мест",
        "заголовок и один абзац": "нет структуры «заголовок + один абзац»",
        "90–120 слов": "нарушен объём 90–120 слов",
        "ответ не обрезан": "ответ обрезан по max_tokens",
    },
    "P2": {
        "A": "неверная метка A",
        "B": "неверная метка B",
        "C": "неверная метка C",
        "D": "неверная метка D",
        "формат: только JSON": "Markdown/текст вместо чистого JSON",
        "ответ не обрезан": "ответ обрезан по max_tokens",
    },
    "P3": {
        "name": "неверное название",
        "volume_l": "неверный объём",
        "power_w": "неверная мощность",
        "materials": "неверные материалы",
        "features": "неверные функции",
        "price_rub": "неверная цена",
        "warranty_months": "неверная гарантия",
        "smartphone_control": "неверное управление со смартфона",
        "summary": "summary отсутствует или длиннее 25 слов",
        "формат: только JSON": "Markdown/текст вместо чистого JSON",
        "ответ не обрезан": "ответ обрезан по max_tokens",
    },
}


def error_types(record: dict[str, Any]) -> str:
    prompt_id = record["prompt"]["id"]
    failed = record["evaluation"].get("failed_checks", [])
    if not failed:
        return "нет формальных ошибок"
    labels = ERROR_LABELS[prompt_id]
    return "; ".join(labels.get(name, name) for name in failed)


def load_records(results_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((results_dir / "responses").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["evaluation"] = evaluate(record)
        records.append(record)
    return records


def write_analysis_json(records: list[dict[str, Any]], results_dir: Path) -> None:
    compact = []
    for record in records:
        compact.append(
            {
                "run_id": record["run_id"],
                "model": record["model"]["id"],
                "family": record["model"]["family"],
                "prompt": record["prompt"]["id"],
                "mode": record["mode"],
                "repeat": record.get("repeat", 1),
                "usage": record["usage"],
                "metrics": {
                    "wall_time_s": record["metrics"].get("wall_time_s"),
                    "end_to_end_completion_tokens_per_s": record["metrics"].get(
                        "end_to_end_completion_tokens_per_s"
                    ),
                    "cold_start_excluded": record["metrics"].get(
                        "cold_start_excluded", False
                    ),
                },
                "evaluation": record["evaluation"],
            }
        )
    (results_dir / "analysis.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def tuned_parameters_label(records: list[dict[str, Any]]) -> str:
    parameter_sets = {
        json.dumps(record.get("parameters", {}), ensure_ascii=False, sort_keys=True)
        for record in records
        if record.get("mode") == "tuned"
    }
    if len(parameter_sets) != 1:
        return "параметры tuned-прогонов различаются"
    parameters = json.loads(parameter_sets.pop())
    return ", ".join(f"`{key}={value}`" for key, value in parameters.items())


def write_summary(records: list[dict[str, Any]], results_dir: Path) -> None:
    tuned_label = tuned_parameters_label(records)
    cold_start_excluded = all(
        record.get("metrics", {}).get("cold_start_excluded") is True for record in records
    )
    lines = [
        "# Сводка эксперимента",
        "",
        f"Получено ответов: **{len(records)}**.",
        "",
        "Все ответы получены через `POST /v1/chat/completions`. Для baseline в тело",
        f"запроса не добавлялись гиперпараметры. Tuned: {tuned_label}.",
        (
            "Перед измерениями выполнен прогрев каждой модели; холодный старт исключён."
            if cold_start_excluded
            else "Внимание: текущие результаты получены до добавления прогрева; времена нужно перезаписать."
        ),
        "",
        "## Все прогоны",
        "",
        "| Модель | Промпт | Режим | Повтор | Input | Output | Время, с | Output ток/с | Автопроверка | Тип ошибок |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        usage = record["usage"]
        metrics = record["metrics"]
        evaluation = record["evaluation"]
        lines.append(
            "| {model} | {prompt} | {mode} | {repeat} | {input} | {output} | {time} | {speed} | "
            "{quality:.1%} | {errors} |".format(
                model=markdown_escape(record["model"]["id"]),
                prompt=record["prompt"]["id"],
                mode=record["mode"],
                repeat=record.get("repeat", 1),
                input=usage.get("prompt_tokens", "—"),
                output=usage.get("completion_tokens", "—"),
                time=metrics.get("wall_time_s", "—"),
                speed=metrics.get("end_to_end_completion_tokens_per_s", "—"),
                quality=evaluation["score"],
                errors=markdown_escape(error_types(record)),
            )
        )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["model"]["id"], record["mode"])].append(record)

    lines.extend(
        [
            "",
            "## Средние по моделям и режимам",
            "",
            "| Модель | Режим | Output tokens | Время, с | Output ток/с | Автопроверка |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for (model_id, mode), group in sorted(groups.items()):
        lines.append(
            "| {model} | {mode} | {tokens} | {time} | {speed} | {quality:.1%} |".format(
                model=model_id,
                mode=mode,
                tokens=mean([item["usage"].get("completion_tokens") for item in group]),
                time=mean([item["metrics"].get("wall_time_s") for item in group]),
                speed=mean(
                    [item["metrics"].get("end_to_end_completion_tokens_per_s") for item in group]
                ),
                quality=mean([item["evaluation"]["score"] for item in group]) or 0,
            )
        )

    lines.extend(
        [
            "",
            "## Ограничение автоматической оценки",
            "",
            "Автопроверка учитывает наличие заданных фактов, полей и меток, но не заменяет",
            "ручную оценку языка и галлюцинаций. Например, формально полный JSON может",
            "содержать искажённое резюме, а пресс-релиз — неподтверждённые имена.",
            "",
            "## Ограничение метрики скорости",
            "",
            "OpenAI-совместимый endpoint Ollama возвращает число input/output токенов,",
            "но не возвращает отдельные длительности prompt-eval и generation. Поэтому",
            "`ток/с` выше — воспроизводимая end-to-end оценка `completion_tokens / wall_time`,",
            (
                "включающая обработку запроса и генерацию; холодная загрузка исключена прогревом."
                if cold_start_excluded
                else "включающая холодную загрузку в старых результатах; требуется повторный прогон."
            ),
            "",
        ]
    )
    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(records: list[dict[str, Any]], results_dir: Path) -> None:
    lines = ["# Полные ответы моделей", ""]
    has_repeats = any(record.get("repeat", 1) > 1 for record in records)
    for record in records:
        repeat_suffix = (
            f" — повтор {record.get('repeat', 1)}"
            if has_repeats
            else ""
        )
        lines.extend(
            [
                f"## {record['model']['id']} — {record['prompt']['id']} — "
                f"{record['mode']}{repeat_suffix}",
                "",
                f"- Входных токенов: {record['usage'].get('prompt_tokens', '—')}",
                f"- Выходных токенов: {record['usage'].get('completion_tokens', '—')}",
                f"- Время: {record['metrics'].get('wall_time_s', '—')} с",
                f"- Автооценка: {record['evaluation']['score']:.1%}",
                f"- Ошибки автопроверки: {error_types(record)}",
                "",
                "````text",
                record["response"].strip(),
                "````",
                "",
            ]
        )
    (results_dir / "full_outputs.md").write_text("\n".join(lines), encoding="utf-8")


def coefficient_of_variation(values: list[float | int]) -> float:
    if not values:
        return 0.0
    average = statistics.mean(values)
    if average == 0:
        return 0.0
    return statistics.pstdev(values) / average


def canonical_content(text: str) -> str:
    parsed = extract_json(text)
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    return normalize(text)


def variability_metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
    responses = [record["response"].strip() for record in group]
    normalized = [normalize(response) for response in responses]
    pairs = list(combinations(range(len(responses)), 2))
    exact_matches = [responses[left] == responses[right] for left, right in pairs]
    similarities = [
        SequenceMatcher(None, normalized[left], normalized[right]).ratio()
        for left, right in pairs
    ]
    lengths = [record["usage"].get("completion_tokens", 0) for record in group]
    times = [record["metrics"].get("wall_time_s", 0) for record in group]
    return {
        "repeats": len(group),
        "unique_texts": len(set(responses)),
        "unique_contents": len({canonical_content(response) for response in responses}),
        "exact_match_rate": statistics.mean(exact_matches) if exact_matches else 1.0,
        "mean_similarity": statistics.mean(similarities) if similarities else 1.0,
        "mean_tokens": statistics.mean(lengths),
        "token_cv": coefficient_of_variation(lengths),
        "time_cv": coefficient_of_variation(times),
        "length_capped_runs": sum(record.get("finish_reason") == "length" for record in group),
    }


def write_variability(records: list[dict[str, Any]], results_dir: Path) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["model"]["id"], record["prompt"]["id"], record["mode"])].append(
            record
        )
    if not any(len(group) > 1 for group in groups.values()):
        return

    lines = [
        "# Вариабельность и повторяемость",
        "",
        "Каждая комбинация модели, промпта и режима запускалась несколько раз с",
        "неизменными параметрами. Чем меньше уникальных ответов и коэффициент вариации",
        "длины, а также чем выше точное совпадение и текстовое сходство, тем стабильнее модель.",
        "",
        "| Модель | Промпт | Режим | Повторы | Уникальные тексты | Уникальное содержание | Точные совпадения | Среднее сходство | Средняя длина | CV длины | CV времени | Обрезано лимитом |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    machine_readable = []
    for (model_id, prompt_id, mode), group in sorted(groups.items()):
        metrics = variability_metrics(group)
        lines.append(
            "| {model} | {prompt} | {mode} | {repeats} | {unique_texts} | "
            "{unique_contents} | {exact:.1%} | {similarity:.1%} | {tokens:.1f} | "
            "{token_cv:.1%} | {time_cv:.1%} | {capped} |".format(
                model=model_id,
                prompt=prompt_id,
                mode=mode,
                repeats=metrics["repeats"],
                unique_texts=metrics["unique_texts"],
                unique_contents=metrics["unique_contents"],
                exact=metrics["exact_match_rate"],
                similarity=metrics["mean_similarity"],
                tokens=metrics["mean_tokens"],
                token_cv=metrics["token_cv"],
                time_cv=metrics["time_cv"],
                capped=metrics["length_capped_runs"],
            )
        )
        machine_readable.append(
            {
                "model": model_id,
                "prompt": prompt_id,
                "mode": mode,
                **metrics,
            }
        )
    lines.extend(
        [
            "",
            "Для JSON-задач «уникальное содержание» сравнивает разобранные объекты и",
            "игнорирует пробелы и Markdown-обёртку. Текстовое сходство рассчитано через",
            "SequenceMatcher; это техническая мера, а не оценка фактической правильности.",
            "CV длины не интерпретируется как стабильность, если ответы обрезаны лимитом.",
            "",
        ]
    )
    (results_dir / "variability.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "variability.json").write_text(
        json.dumps(machine_readable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    records = load_records(args.results)
    if not records:
        raise SystemExit(f"Не найдены JSON-ответы в {args.results / 'responses'}")
    write_analysis_json(records, args.results)
    write_summary(records, args.results)
    write_outputs(records, args.results)
    write_variability(records, args.results)
    print(f"Проанализировано ответов: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
