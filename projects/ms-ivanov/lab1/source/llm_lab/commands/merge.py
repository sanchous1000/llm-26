"""Команда merge: склейка прогонов одной модели из нескольких сессий запуска.

    python source/main.py merge --model google/gemma-4-12b-qat \\
        --parts runs_1_4.jsonl runs_5.jsonl runs.jsonl --cleanup

Порядок --parts хронологический: при дублях остаётся запись из более поздней части.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_lab.maintenance.merge_runs import (
    build_merged_manifest,
    check_params_match_config,
    cleanup,
    load_parts,
    merge,
    missing_cells,
)
from llm_lab.reporting.model_report import generate_report
from llm_lab.storage import (
    MANIFEST_FILENAME,
    RUNS_FILENAME,
    ModelPaths,
    default_results_root,
    load_runs,
    rewrite_answers,
    write_manifest,
    write_runs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Склейка частичных прогонов одной модели")
    parser.add_argument("--model", required=True, help="ID модели")
    parser.add_argument("--parts", nargs="+", required=True,
                        help="файлы прогонов в хронологическом порядке (имена внутри директории модели)")
    parser.add_argument("--results", type=Path, default=default_results_root())
    parser.add_argument("--cleanup", action="store_true",
                        help="после успешной склейки удалить частичные файлы")
    args = parser.parse_args(argv)

    paths = ModelPaths(root=args.results, model_id=args.model)
    try:
        parts = load_parts(paths.model_dir, args.parts, args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}")
        return 1

    records, dropped, sources = merge(parts)
    for part in parts:
        print(f"  {part.path.name}: {len(part.records)} записей")
    for item in dropped:
        print(f"  [дубль] {item['run_id']} из {item['source']} ({item['timestamp']}) заменён записью из {item['replaced_by']}")

    gaps = missing_cells(records)
    if gaps:
        print(f"[warn] не хватает {len(gaps)} прогонов: {', '.join(gaps[:10])}")
    manifest = build_merged_manifest(records, parts, dropped, sources)
    mismatched = check_params_match_config(records, manifest["include_extra_params"])
    if mismatched:
        print(f"[warn] параметры {len(mismatched)} прогонов не совпадают с текущими пресетами: {mismatched[:3]}")

    write_runs(paths.runs_file, records)
    reloaded = load_runs(paths.runs_file)
    if len(reloaded) != len(records):
        print(f"[error] проверка записи не прошла: {len(reloaded)} ≠ {len(records)}; частичные файлы не удалены")
        return 1
    write_manifest(paths, manifest)
    rewrite_answers(paths, records)
    report = generate_report(paths)
    print(f"Итого: {len(records)} прогонов → {RUNS_FILENAME}, {MANIFEST_FILENAME}, answers/, {report.name}")

    if args.cleanup:
        removed = cleanup(paths.model_dir, parts)
        print(f"Удалено: {', '.join(removed) or 'нечего удалять'}")
    return 0
