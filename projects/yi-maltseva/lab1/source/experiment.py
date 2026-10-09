from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from config import FIELDNAMES, MODELS, MODES, OLLAMA_BASE_URL, TUNED_OPTIONS
from evaluation import score_answer
from ollama_client import check_api, completion_request, ensure_server_and_models, native_probe, warmup_model
from test_cases import PROMPTS
from reporting import analyze_rows, write_csv


def run_experiment(base_url: str, output_dir: Path, repeats: int, skip_native_probe: bool) -> int:
    try:
        ensure_server_and_models(base_url)
    except (RuntimeError, requests.RequestException) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    telemetry_by_group: dict[tuple[str, str, str], dict[str, Any]] = {}
    load_by_model: dict[str, float | None] = {}
    planned = len(MODELS) * len(PROMPTS) * len(MODES) * repeats
    completed = 0
    print(f"Запуск: {len(MODELS)} модели × {len(PROMPTS)} промпта × {len(MODES)} режима × {repeats} повтора = {planned} ответов.")
    print("Первый запуск каждой модели может занять больше времени из-за загрузки в память.")

    for model_info in MODELS:
        model = model_info["model"]
        print(f"\n=== {model_info['family']} / {model} ===")
        load_by_model[model] = warmup_model(base_url, model)
        for prompt_id, prompt_info in PROMPTS.items():
            for mode in MODES:
                params = {} if mode == "A_baseline" else dict(TUNED_OPTIONS)
                for repeat_index in range(1, repeats + 1):
                    completed += 1
                    run_id = f"{model_info['family']}_{prompt_id}_{mode}_R{repeat_index}"
                    print(f"[{completed}/{planned}] {prompt_id} {mode} повтор {repeat_index} ...", flush=True)
                    row: dict[str, Any] = {
                        "run_id": run_id,
                        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "model": model,
                        "family": model_info["family"],
                        "prompt_id": prompt_id,
                        "prompt_title": prompt_info["title"],
                        "mode": mode,
                        "repeat_index": repeat_index,
                        "parameters_json": json.dumps(params, ensure_ascii=False),
                        "answer": "",
                        "response_time_s": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "token_count_source": "OpenAI usage",
                        "end_to_end_output_tokens_per_sec": None,
                        "prompt_eval_tokens": None,
                        "prompt_eval_time_s": None,
                        "prompt_eval_tokens_per_sec": None,
                        "probe_output_tokens": None,
                        "probe_eval_time_s": None,
                        "probe_generation_tokens_per_sec": None,
                        "model_load_time_s": load_by_model.get(model),
                        "task_score": None,
                        "score_details_json": "{}",
                        "error": "",
                    }
                    try:
                        result = completion_request(base_url, model, prompt_info["prompt"], mode)
                        answer = result["answer"]
                        task_score, score_details = score_answer(prompt_id, answer)
                        row.update({
                            "answer": answer,
                            "response_time_s": round(result["response_time_s"], 6),
                            "prompt_tokens": result["prompt_tokens"],
                            "completion_tokens": result["completion_tokens"],
                            "total_tokens": result["total_tokens"],
                            "task_score": round(task_score, 4) if task_score is not None else None,
                            "score_details_json": json.dumps(score_details, ensure_ascii=False),
                        })
                        if result["completion_tokens"] is not None and result["response_time_s"] > 0:
                            row["end_to_end_output_tokens_per_sec"] = round(result["completion_tokens"] / result["response_time_s"], 4)
                    except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
                        row["error"] = str(exc)
                        print(f"  Ошибка запроса: {exc}", file=sys.stderr)
                    rows.append(row)
                    write_csv(output_dir / "runs.csv", rows, FIELDNAMES)

                if not skip_native_probe:
                    print(f"  Диагностический замер prompt/decode: {prompt_id} {mode} ...", flush=True)
                    telemetry = native_probe(base_url, model, prompt_info["prompt"], mode)
                    telemetry_by_group[(model, prompt_id, mode)] = telemetry
                    if telemetry.get("error"):
                        print(f"  [Предупреждение] Нет этапных метрик: {telemetry['error']}", file=sys.stderr)
                    for row in rows:
                        if row["model"] == model and row["prompt_id"] == prompt_id and row["mode"] == mode:
                            for key in (
                                "prompt_eval_tokens", "prompt_eval_time_s", "prompt_eval_tokens_per_sec",
                                "probe_output_tokens", "probe_eval_time_s", "probe_generation_tokens_per_sec",
                            ):
                                row[key] = telemetry.get(key)
                            if telemetry.get("model_load_time_s") is not None and row.get("model_load_time_s") is None:
                                row["model_load_time_s"] = telemetry["model_load_time_s"]
                    write_csv(output_dir / "runs.csv", rows, FIELDNAMES)

    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_url": base_url,
        "models": MODELS,
        "prompt_definitions": PROMPTS,
        "baseline": "A_baseline: параметры temperature/top_p/max_tokens не передаются явно",
        "tuned": {"mode": "B_tuned", "parameters": TUNED_OPTIONS},
        "repeats": repeats,
        "runs": rows,
        "diagnostic_metrics": [
            {"model": model, "prompt_id": prompt_id, "mode": mode, **metrics}
            for (model, prompt_id, mode), metrics in telemetry_by_group.items()
        ],
    }
    (output_dir / "answers.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_rows, summary_text = analyze_rows(rows)
    summary_fields = list(summary_rows[0].keys()) if summary_rows else ["model", "family", "mode", "runs"]
    write_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    (output_dir / "summary.md").write_text(summary_text, encoding="utf-8")

    manual_rows = [{
        "run_id": row["run_id"],
        "model": row["model"],
        "prompt_id": row["prompt_id"],
        "mode": row["mode"],
        "repeat_index": row["repeat_index"],
        "adequacy_1_to_5": "",
        "factual_errors_yes_no": "",
        "hallucinations_yes_no": "",
        "repeated_phrases_yes_no": "",
        "notes": "",
    } for row in rows]
    write_csv(output_dir / "manual_review.csv", manual_rows, list(manual_rows[0].keys()) if manual_rows else [])

    errors = sum(bool(row.get("error")) for row in rows)
    print("\nГотово.")
    print(f"Ответы: {output_dir / 'answers.json'}")
    print(f"Таблица прогонов: {output_dir / 'runs.csv'}")
    print(f"Сводка: {output_dir / 'summary.md'}")
    print(f"Шаблон ручной оценки: {output_dir / 'manual_review.csv'}")
    print(f"Строк с ошибками: {errors} из {len(rows)}")
    return 2 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Лабораторная № 1: эмпирическое сравнение LLM через Ollama.")
    parser.add_argument("command", choices=("check", "run"), nargs="?", default="run")
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--skip-native-probe", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats должен быть не меньше 1")
    if args.command == "check":
        return check_api(args.base_url)
    return run_experiment(args.base_url, Path(args.output_dir), args.repeats, args.skip_native_probe)


if __name__ == "__main__":
    raise SystemExit(main())
