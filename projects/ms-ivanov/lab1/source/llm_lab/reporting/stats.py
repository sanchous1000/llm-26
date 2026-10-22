"""Агрегация метрик по прогонам (чистые функции над записями JSONL)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from llm_lab.evaluation.checks import HALLUCINATION_MARKER
from llm_lab.evaluation.text import mean_pairwise_similarity

Record = dict[str, Any]


def mean(values: Sequence[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def stdev(values: Sequence[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.pstdev(clean) if len(clean) > 1 else 0.0 if clean else None


def group_by(records: Iterable[Record], key: Callable[[Record], Any]) -> dict[Any, tuple[Record, ...]]:
    grouped: dict[Any, list[Record]] = {}
    for record in records:
        grouped.setdefault(key(record), []).append(record)
    return {k: tuple(v) for k, v in grouped.items()}


def ok_records(records: Iterable[Record]) -> tuple[Record, ...]:
    return tuple(r for r in records if not r.get("error"))


def metric(record: Record, name: str) -> Any:
    return (record.get("metrics") or {}).get(name)


def has_hallucination(record: Record) -> bool:
    """Хотя бы одна проверка явно пометила ответ как выдумку."""
    return any(HALLUCINATION_MARKER in str(c.get("detail", "")) for c in record.get("checks", []))


def hallucination_details(record: Record) -> tuple[str, ...]:
    return tuple(
        str(c.get("detail", ""))
        for c in record.get("checks", [])
        if HALLUCINATION_MARKER in str(c.get("detail", ""))
    )


@dataclass(frozen=True)
class GroupStats:
    """Сводные метрики по произвольной группе прогонов."""

    label: str
    runs: int
    errors: int
    quality: float | None
    quality_min: float | None
    completion_tokens: float | None
    completion_tokens_sd: float | None
    prompt_tokens: float | None
    latency_s: float | None
    ttft_s: float | None
    output_tps: float | None
    truncated: int
    hallucinations: int
    words: float | None
    repeat_ngram_ratio: float | None
    empty_answers: int
    reasoning_cut: int
    reasoning_chars: float | None
    ttfa_s: float | None
    latency_p50: float | None
    reasoning_tokens: float | None
    answer_tokens: float | None
    reasoning_time_share: float | None
    completion_tokens_min: int | None
    completion_tokens_max: int | None
    tps_methods: tuple[str, ...]

    @property
    def error_rate(self) -> float:
        return self.errors / self.runs if self.runs else 0.0


def median(values: Sequence[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def is_empty_answer(record: Record) -> bool:
    return not (record.get("answer") or "").strip()


def is_reasoning_cut(record: Record) -> bool:
    """Ответа нет, потому что лимит длины израсходован блоком рассуждений.

    Для reasoning-моделей это не качество модели, а неудачный max_tokens —
    такие прогоны в отчёте помечаются отдельно.
    """
    return is_empty_answer(record) and (metric(record, "reasoning_chars") or 0) > 0


def summarize(label: str, records: Sequence[Record]) -> GroupStats:
    good = ok_records(records)
    qualities = [r.get("quality_score") for r in good]
    tokens = [t for t in (metric(r, "completion_tokens") for r in good) if t is not None]
    return GroupStats(
        label=label,
        runs=len(records),
        errors=len(records) - len(good),
        quality=mean(qualities),
        quality_min=min(qualities) if qualities else None,
        completion_tokens=mean([metric(r, "completion_tokens") for r in good]),
        completion_tokens_sd=stdev([metric(r, "completion_tokens") for r in good]),
        prompt_tokens=mean([metric(r, "prompt_tokens") for r in good]),
        latency_s=mean([metric(r, "latency_s") for r in good]),
        ttft_s=mean([metric(r, "ttft_s") for r in good]),
        output_tps=mean([metric(r, "output_tps") for r in good]),
        truncated=sum(1 for r in good if metric(r, "truncated")),
        hallucinations=sum(1 for r in good if has_hallucination(r)),
        words=mean([metric(r, "words") for r in good]),
        repeat_ngram_ratio=mean([metric(r, "repeat_ngram_ratio") for r in good]),
        empty_answers=sum(1 for r in good if is_empty_answer(r)),
        reasoning_cut=sum(1 for r in good if is_reasoning_cut(r)),
        reasoning_chars=mean([metric(r, "reasoning_chars") for r in good]),
        ttfa_s=mean([metric(r, "ttfa_s") for r in good]),
        latency_p50=median([metric(r, "latency_s") for r in good]),
        reasoning_tokens=mean([metric(r, "reasoning_tokens_est") for r in good]),
        answer_tokens=mean([metric(r, "answer_tokens_est") for r in good]),
        reasoning_time_share=mean([metric(r, "reasoning_time_share") for r in good]),
        completion_tokens_min=min(tokens) if tokens else None,
        completion_tokens_max=max(tokens) if tokens else None,
        tps_methods=tuple(sorted({str(metric(r, "tps_method")) for r in good if metric(r, "tps_method")})),
    )


@dataclass(frozen=True)
class StabilityStats:
    """Повторяемость ответов внутри одной ячейки (промпт × пресет)."""

    prompt_id: str
    preset_key: str
    repeats: int
    distinct_answers: int
    identical: bool
    mean_similarity: float | None
    quality_spread: float | None
    tokens_spread: float | None


def stability(records: Sequence[Record]) -> StabilityStats | None:
    """Считает разброс для группы повторов одной конфигурации."""
    good = ok_records(records)
    if not good:
        return None
    answers = tuple((r.get("answer") or "").strip() for r in good)
    qualities = [r.get("quality_score") for r in good if r.get("quality_score") is not None]
    tokens = [metric(r, "completion_tokens") for r in good]
    tokens = [t for t in tokens if t is not None]
    return StabilityStats(
        prompt_id=good[0]["prompt_id"],
        preset_key=good[0]["preset_key"],
        repeats=len(good),
        distinct_answers=len(set(answers)),
        identical=len(set(answers)) == 1 and len(answers) > 1,
        mean_similarity=mean_pairwise_similarity(answers),
        quality_spread=(max(qualities) - min(qualities)) if len(qualities) > 1 else 0.0,
        tokens_spread=(max(tokens) - min(tokens)) if len(tokens) > 1 else 0.0,
    )


def check_pass_rates(records: Sequence[Record]) -> dict[str, tuple[int, int]]:
    """Имя проверки -> (пройдено, всего) по всем прогонам группы."""
    rates: dict[str, list[int]] = {}
    for record in ok_records(records):
        for check in record.get("checks", []):
            entry = rates.setdefault(check["name"], [0, 0])
            entry[0] += 1 if check.get("passed") else 0
            entry[1] += 1
    return {name: (passed, total) for name, (passed, total) in rates.items()}


def best_run(records: Sequence[Record]) -> Record | None:
    """Прогон с максимальным баллом (при равенстве — более короткий)."""
    good = ok_records(records)
    if not good:
        return None
    return max(
        good,
        key=lambda r: (r.get("quality_score") or 0.0, -(metric(r, "completion_tokens") or 0)),
    )


def worst_run(records: Sequence[Record]) -> Record | None:
    good = ok_records(records)
    if not good:
        return None
    return min(good, key=lambda r: (r.get("quality_score") or 0.0))


def ordered_preset_keys(records: Sequence[Record]) -> tuple[str, ...]:
    seen: list[str] = []
    for record in records:
        key = record["preset_key"]
        if key not in seen:
            seen.append(key)
    return tuple(seen)


def ordered_prompt_ids(records: Sequence[Record]) -> tuple[str, ...]:
    seen: list[str] = []
    for record in records:
        key = record["prompt_id"]
        if key not in seen:
            seen.append(key)
    return tuple(seen)
