"""Пересчёт автопроверок по сохранённым ответам — без повторного запуска моделей.

Проверки — чистые функции от текста ответа, поэтому после исправления
набора проверок (experiment/prompts.py / evaluation/checks.py) баллы можно пересчитать на уже
собранных данных. Первый пересчёт сохраняет исходные значения в полях
`quality_score_v1` и `checks_v1` — сравнение версий остаётся воспроизводимым.

Запуск:
    python source/main.py rescore --all
    python source/main.py rescore --model qwen3.5-35b-a3b --dry-run
"""

from __future__ import annotations

from typing import Any

from llm_lab.evaluation.checks import CHECKS_VERSION, aggregate_score, evaluate_all
from llm_lab.experiment.prompts import ALL_PROMPTS
from llm_lab.reporting.model_report import generate_report
from llm_lab.storage import (
    RUNS_FILENAME,
    ModelPaths,
    load_runs,
    rewrite_answers,
    write_runs,
)

Record = dict[str, Any]
PROMPTS_BY_ID = {case.id: case for case in ALL_PROMPTS}


def rescore_record(record: Record) -> Record:
    """Новая запись с проверками текущей версии (исходная не изменяется)."""
    case = PROMPTS_BY_ID.get(record["prompt_id"])
    if case is None:
        raise ValueError(f"{record['run_id']}: промпт {record['prompt_id']} отсутствует в llm_lab/experiment/prompts.py")
    if record.get("user") != case.user:
        raise ValueError(f"{record['run_id']}: текст промпта в данных не совпадает с llm_lab/experiment/prompts.py")

    results = evaluate_all(case.checks, record.get("answer") or "") if not record.get("error") else ()
    history = (
        {}
        if "quality_score_v1" in record
        else {"quality_score_v1": record.get("quality_score"), "checks_v1": record.get("checks", [])}
    )
    return {
        **record,
        **history,
        "checks": [r.as_dict() for r in results],
        "checks_version": CHECKS_VERSION,
        "quality_score": round(aggregate_score(results), 4),
    }


def rescore_model(paths: ModelPaths, dry_run: bool) -> int:
    try:
        records = load_runs(paths.runs_file)
        rescored = tuple(rescore_record(r) for r in records)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {paths.model_dir.name}: {exc}")
        return 1

    changed = [
        (old["run_id"], old.get("quality_score"), new["quality_score"])
        for old, new in zip(records, rescored)
        if old.get("quality_score") != new["quality_score"]
    ]
    mean_before = sum(r.get("quality_score_v1", r.get("quality_score")) or 0 for r in rescored) / len(rescored)
    mean_after = sum(r["quality_score"] for r in rescored) / len(rescored)
    print(f"\n== {records[0]['model']}: средний балл v1 {mean_before:.3f} → v{CHECKS_VERSION} {mean_after:.3f}; "
          f"изменилось прогонов: {len(changed)}")
    for run_id, before, after in changed[:8]:
        print(f"   {run_id}: {before} → {after}")
    if len(changed) > 8:
        print(f"   … и ещё {len(changed) - 8}")

    if dry_run:
        print("   --dry-run: изменения не записаны")
        return 0
    write_runs(paths.runs_file, rescored)
    rewrite_answers(paths, rescored)
    generate_report(paths)
    print(f"   записано: {RUNS_FILENAME}, answers/, report.md")
    return 0
