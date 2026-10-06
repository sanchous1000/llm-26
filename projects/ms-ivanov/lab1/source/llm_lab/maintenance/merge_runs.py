"""Склейка частичных прогонов одной модели в единый набор результатов.

Нужен, когда эксперимент по модели запускался несколькими сессиями без
--append: каждый запуск перезаписывает answers/, manifest.json и report.md.
Скрипт собирает runs*.jsonl в один runs.jsonl, пересобирает answers/,
восстанавливает manifest.json (с историей сессий) и строит отчёт.

Пример:
    python source/main.py merge --model google/gemma-4-12b-qat \\
        --parts runs_1_4.jsonl runs_5.jsonl runs.jsonl --cleanup

Порядок --parts важен: при дублях (один и тот же промпт × пресет × повтор)
остаётся запись из БОЛЕЕ ПОЗДНЕЙ части — так все повторы одной конфигурации
приходят из одной сессии и сравнение стабильности остаётся корректным.
Отброшенные дубли сохраняются в манифесте, а не теряются молча.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_lab.experiment.config import ALL_PRESETS, PROFILES, preset_by_key
from llm_lab.experiment.prompts import ALL_PROMPTS, resolve_prompts
from llm_lab.experiment.runner import ExperimentConfig, build_manifest
from llm_lab.inference.params import EXTRA_PARAM_NAMES, EndpointConfig
from llm_lab.storage import (
    load_runs,
    utc_now_iso,
)

Record = dict[str, Any]
CellKey = tuple[str, str, int]

PROMPT_ORDER = {case.id: index for index, case in enumerate(ALL_PROMPTS)}
PRESET_ORDER = {preset.key: index for index, preset in enumerate(ALL_PRESETS)}


@dataclass(frozen=True)
class Part:
    path: Path
    records: tuple[Record, ...]

    @property
    def suffix(self) -> str:
        """runs_1_4.jsonl → "_1_4"; runs.jsonl → ""."""
        return self.path.stem[len("runs"):] if self.path.stem.startswith("runs") else ""

    @property
    def manifest(self) -> dict[str, Any]:
        candidate = self.path.parent / f"manifest{self.suffix}.json"
        if not candidate.exists():
            return {}
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] {candidate.name} не прочитан: {exc}")
            return {}


def cell_key(record: Record) -> CellKey:
    return record["prompt_id"], record["preset_key"], int(record["repeat"])


def sort_key(record: Record) -> tuple[int, int, int]:
    return (
        PROMPT_ORDER.get(record["prompt_id"], len(PROMPT_ORDER)),
        PRESET_ORDER.get(record["preset_key"], len(PRESET_ORDER)),
        int(record["repeat"]),
    )


def load_parts(model_dir: Path, names: list[str], model_id: str) -> tuple[Part, ...]:
    parts: list[Part] = []
    for name in names:
        path = Path(name) if Path(name).is_absolute() else model_dir / name
        records = load_runs(path)
        foreign = {r["model"] for r in records} - {model_id}
        if foreign:
            raise ValueError(f"{path.name}: записи другой модели {sorted(foreign)}")
        parts.append(Part(path=path, records=records))
    return tuple(parts)


def merge(parts: tuple[Part, ...]) -> tuple[tuple[Record, ...], list[dict[str, Any]], dict[str, str]]:
    """Склеить части; вернуть (записи, отброшенные дубли, источник каждой записи)."""
    kept: dict[CellKey, Record] = {}
    source: dict[CellKey, str] = {}
    dropped: list[dict[str, Any]] = []
    for part in parts:
        for record in part.records:
            key = cell_key(record)
            if key in kept:
                old = kept[key]
                dropped.append({
                    "run_id": old["run_id"],
                    "timestamp": old["timestamp"],
                    "source": source[key],
                    "replaced_by": part.path.name,
                    "quality_score": old.get("quality_score"),
                    "latency_s": old.get("metrics", {}).get("latency_s"),
                    "completion_tokens": old.get("metrics", {}).get("completion_tokens"),
                    "answer": old.get("answer"),
                })
            kept[key] = record
            source[key] = part.path.name
    merged = tuple(sorted(kept.values(), key=sort_key))
    return merged, dropped, {r["run_id"]: source[cell_key(r)] for r in merged}


def missing_cells(records: tuple[Record, ...]) -> list[str]:
    prompts = {r["prompt_id"] for r in records}
    presets = {r["preset_key"] for r in records}
    repeats = max(int(r.get("repeats_total") or 1) for r in records)
    present = {cell_key(r) for r in records}
    return [
        f"{p}__{k}__r{n}"
        for p in sorted(prompts, key=lambda x: PROMPT_ORDER.get(x, 99))
        for k in sorted(presets, key=lambda x: PRESET_ORDER.get(x, 99))
        for n in range(1, repeats + 1)
        if (p, k, n) not in present
    ]


def check_params_match_config(records: tuple[Record, ...], include_extra: bool) -> list[str]:
    problems = []
    for record in records:
        expected = preset_by_key(record["preset_key"]).params.to_payload(include_extra=include_extra)
        if record.get("params") != expected:
            problems.append(f"{record['run_id']}: {record.get('params')} ≠ {expected}")
    return problems


def build_merged_manifest(
    records: tuple[Record, ...],
    parts: tuple[Part, ...],
    dropped: list[dict[str, Any]],
    sources: dict[str, str],
) -> dict[str, Any]:
    include_extra = any(set(r.get("params", {})) & set(EXTRA_PARAM_NAMES) for r in records)
    preset_keys = sorted({r["preset_key"] for r in records}, key=lambda k: PRESET_ORDER.get(k, 99))
    prompt_ids = sorted({r["prompt_id"] for r in records}, key=lambda p: PROMPT_ORDER.get(p, 99))
    manifests = [part.manifest for part in parts if part.manifest]
    profile = next(
        (m["profile"] for m in manifests if m.get("profile")),
        next((name for name, keys in PROFILES.items() if list(keys) == preset_keys), "custom"),
    )

    cfg = ExperimentConfig(
        model=records[0]["model"],
        endpoint=EndpointConfig(base_url=records[0].get("endpoint", EndpointConfig().base_url)),
        presets=tuple(preset_by_key(key) for key in preset_keys),
        prompts=resolve_prompts(tuple(prompt_ids)),
        repeats=max(int(r.get("repeats_total") or 1) for r in records),
        stream=all(bool(r.get("stream")) for r in records),
        safety_max_tokens=next((m.get("safety_max_tokens") for m in manifests), None),
        include_extra_params=include_extra,
        profile=profile,
    )
    starts = [m["started_at"] for m in manifests if m.get("started_at")]
    timestamps = [r["timestamp"] for r in records]

    return {
        **build_manifest(cfg),
        "started_at": min(starts + timestamps),
        "finished_at": max(timestamps),
        "total_runs": len(records),
        "merged_at": utc_now_iso(),
        "sessions": [
            {
                "source": part.path.name,
                "manifest_started_at": part.manifest.get("started_at"),
                "records_loaded": len(part.records),
                "records_used": sum(1 for r in records if sources[r["run_id"]] == part.path.name),
                "prompts": sorted({r["prompt_id"] for r in part.records}, key=lambda p: PROMPT_ORDER.get(p, 99)),
                "first_run_at": min(r["timestamp"] for r in part.records),
                "last_run_at": max(r["timestamp"] for r in part.records),
            }
            for part in parts
        ],
        "dropped_duplicates": dropped,
    }


def cleanup(model_dir: Path, parts: tuple[Part, ...]) -> list[str]:
    """Удалить частичные файлы (runs_X.jsonl и одноимённые report_X.md, manifest_X.json, answers_X/)."""
    removed: list[str] = []
    root = model_dir.resolve()
    for part in parts:
        if not part.suffix:
            continue  # runs.jsonl — это и есть итоговый файл
        candidates = (
            part.path,
            model_dir / f"report{part.suffix}.md",
            model_dir / f"manifest{part.suffix}.json",
            model_dir / f"answers{part.suffix}",
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if root not in resolved.parents or not candidate.exists():
                continue
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
            removed.append(candidate.name)
    return removed
