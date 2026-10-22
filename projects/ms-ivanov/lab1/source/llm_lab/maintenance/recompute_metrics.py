"""Пересчёт метрик скорости в сохранённых прогонах — без повторного запуска моделей.

Исправляет прогоны с метриками версии 1, где TTFT засекался по первому
токену поля `content`. См. подробности в llm_lab/inference/speed.py.

Запуск:
    python source/main.py recompute-metrics --all
    python source/main.py recompute-metrics --model qwen3.5-35b-a3b --dry-run
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from llm_lab.experiment.runner import DUPLICATED_TEXT_FIELDS
from llm_lab.inference.speed import METRICS_VERSION, SpeedInputs, derive_speed_metrics, estimate_answer_tokens
from llm_lab.reporting.model_report import generate_report
from llm_lab.storage import (
    ModelPaths,
    load_runs,
    rewrite_answers,
    write_runs,
)

# Если старый «TTFT» / полное время ≥ этой доли от доли токенов рассуждений,
# значит рассуждения шли отдельным полем и старое значение — время до ОТВЕТА.
# На реальных данных отношение 0.93–1.0; у inline-<think> оно было бы около 0.
SEPARATE_REASONING_MIN_RATIO = 0.5

Record = dict[str, Any]


def classify_v1_timing(metrics: dict[str, Any], stream: bool) -> tuple[float | None, float | None, str]:
    """Восстановить (ttft, ttfa, вид случая) из метрик версии 1."""
    old = metrics.get("ttft_s")
    latency = float(metrics.get("latency_s") or 0.0)
    if not stream:
        return None, None, "non_stream"
    if old is None:
        return None, None, "no_answer_tokens"
    reasoning_chars = int(metrics.get("reasoning_chars") or 0)
    if reasoning_chars == 0:
        return old, old, "no_reasoning"

    tokens = int(metrics.get("completion_tokens") or 0)
    answer_tokens = estimate_answer_tokens(int(metrics.get("chars") or 0), tokens)
    reasoning_share = (tokens - answer_tokens) / tokens if tokens else 0.0
    if latency > 0 and old / latency >= SEPARATE_REASONING_MIN_RATIO * reasoning_share:
        # Реальный TTFT не сохранялся: поле засекало начало видимого ответа.
        return None, old, "separate_reasoning"
    return old, None, "inline_reasoning"


def migrate_record(record: Record) -> tuple[Record, str]:
    """Вернуть НОВУЮ запись с метриками версии 2 и вид исправленного случая."""
    metrics = record.get("metrics") or {}
    if int(metrics.get("metrics_version") or 1) >= METRICS_VERSION:
        return record, "up_to_date"
    if record.get("error"):
        cleaned = {k: v for k, v in metrics.items() if k not in DUPLICATED_TEXT_FIELDS}
        return {**record, "metrics": {**cleaned, "metrics_version": METRICS_VERSION}}, "error_run"

    ttft, ttfa, label = classify_v1_timing(metrics, bool(record.get("stream")))
    speed = derive_speed_metrics(SpeedInputs(
        latency_s=float(metrics.get("latency_s") or 0.0),
        ttft_s=ttft,
        ttfa_s=ttfa,
        completion_tokens=int(metrics.get("completion_tokens") or 0),
        answer_chars=int(metrics.get("chars") or 0),
        reasoning_chars=int(metrics.get("reasoning_chars") or 0),
    ))
    cleaned = {k: v for k, v in metrics.items() if k not in DUPLICATED_TEXT_FIELDS}
    return {**record, "metrics": {**cleaned, **speed}}, label


def _median_tps(records: tuple[Record, ...]) -> float | None:
    values = [r["metrics"].get("output_tps") for r in records if r.get("metrics", {}).get("output_tps")]
    return round(statistics.median(values), 1) if values else None


def recompute_model(paths: ModelPaths, dry_run: bool) -> int:
    try:
        records = load_runs(paths.runs_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {paths.model_dir.name}: {exc}")
        return 1

    migrated_pairs = [migrate_record(r) for r in records]
    migrated = tuple(record for record, _ in migrated_pairs)
    labels = Counter(label for _, label in migrated_pairs)

    before, after = _median_tps(records), _median_tps(migrated)
    print(f"\n== {records[0]['model']} ({len(records)} прогонов)")
    print(f"   случаи: {dict(labels)}")
    print(f"   медиана ток/с: {before} → {after}")

    if labels.keys() == {"up_to_date"}:
        print("   метрики уже актуальны — файл не меняется")
        return 0
    if dry_run:
        print("   --dry-run: изменения не записаны")
        return 0

    write_runs(paths.runs_file, migrated)
    rewrite_answers(paths, migrated)
    report = generate_report(paths)
    print(f"   записано: {paths.runs_file.name}, answers/ ({len(migrated)}), {report.name}")
    return 0
