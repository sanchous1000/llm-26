"""Добавляет сохранённые оценки к существующим сессиям и трейсам Langfuse без запуска агентов."""

import argparse
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv

REVIEW_VERSION = "lab3-manual-v2"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def stable_id(text):
    return uuid5(NAMESPACE_URL, text).hex


def build_plan(directory, agent=None, task=None):
    runs_dir = Path(__file__).resolve().parents[2] / "artifacts/eval"
    turns = [
        r
        for r in read_rows(directory / "per_turn.jsonl")
        if (agent is None or r["agent"] == agent)
        and (task is None or r["task_id"] == task)
    ]
    tasks = [
        r
        for r in read_rows(directory / "per_task.jsonl")
        if (agent is None or r["agent"] == agent)
        and (task is None or r["task_id"] == task)
    ]
    metrics = read_json(directory / "metrics.json")
    scores = []
    sessions = {}
    for name in {row["agent"] for row in tasks}:
        run_id = metrics[name]["run_id"]
        for record in read_rows(runs_dir / run_id / "results.jsonl"):
            sessions[(name, record["task_id"])] = record["session_id"]

    def add(
        trace_id, name, value, metadata, comment=None, boolean=False, session_id=None
    ):
        target = {"session_id": session_id} if session_id else {"trace_id": trace_id}
        identity = "session:" + session_id if session_id else trace_id
        scores.append(
            {
                "id": stable_id(f"{REVIEW_VERSION}:{identity}:{name}"),
                **target,
                "name": name,
                "value": float(value),
                "data_type": "BOOLEAN" if boolean else "NUMERIC",
                "comment": comment,
                "metadata": {"review_version": REVIEW_VERSION, **metadata},
            }
        )

    for row in turns:
        meta = {key: row[key] for key in ["agent", "run_id", "task_id", "turn_id"]}
        add(
            row["trace_id"],
            "turn_success",
            row["manual_success"],
            meta,
            row["failure_reason"] or "Критерии реплики выполнены.",
            boolean=True,
        )
        if row["history_dependent"]:
            add(
                row["trace_id"],
                "history_turn_success",
                row["manual_success"],
                meta,
                "Полная оценка ответа на реплику, зависящую от истории.",
                boolean=True,
            )
            add(
                row["trace_id"],
                "history_correct",
                row["history_correct"],
                meta,
                "Сохранение условий истории отдельно от правильности новой теории.",
                boolean=True,
            )
        if row["tool_correctness"] is not None:
            add(
                row["trace_id"],
                "tool_correctness",
                row["tool_correctness"],
                meta,
                "Проверены обязательные имена; аргументы и запреты входят в ручную оценку.",
                boolean=True,
            )

    for row in tasks:
        run_id = metrics[row["agent"]]["run_id"]
        meta = {"agent": row["agent"], "run_id": run_id, "task_id": row["task_id"]}
        session_id = sessions[(row["agent"], row["task_id"])]
        add(
            None,
            "task_success",
            row["task_success"],
            meta,
            (
                "Проваленные реплики: " + ", ".join(row["failed_turns"])
                if row["failed_turns"]
                else "Все реплики выполнены."
            ),
            boolean=True,
            session_id=session_id,
        )
        for name in [
            "local_elapsed_seconds",
            "tokens_observed",
            "cost_usd_observed",
            "tool_calls",
        ]:
            add(
                None,
                "task." + name,
                row[name],
                meta,
                "Сумма по задаче; cost/tokens — только зарегистрированные значения.",
                session_id=session_id,
            )
        if row["retrieval_diagnostic"]:
            for name, value in row["retrieval_diagnostic"].items():
                add(
                    None,
                    "task.retrieval_diagnostic." + name,
                    value,
                    meta,
                    "Диагностический retrieval по общим эталонным страницам задачи.",
                    session_id=session_id,
                )

    return {"review_version": REVIEW_VERSION, "scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("artifacts/eval/analysis-20261113-v2"),
        help="Каталог итоговых метрик; относительно lab3 или абсолютный путь.",
    )
    parser.add_argument("--agent", choices=["custom", "smolagents"])
    parser.add_argument("--task")
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    directory = project / args.analysis_dir
    plan = build_plan(directory, args.agent, args.task)
    if not plan["scores"]:
        parser.error("Для выбранного агента/задачи нет оценок.")
    plan_path = directory / "scores_upload_plan.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Оценок:", len(plan["scores"]), "План:", plan_path)
    if not args.send:
        print("Только подготовка. Для отправки добавь --send.")
        return

    from langfuse import Langfuse
    from langfuse.api.scores.types.create_score_request import CreateScoreRequest

    load_dotenv(project / ".env")
    client = Langfuse(tracing_enabled=False)
    try:
        for start in range(0, len(plan["scores"]), 100):
            batch = plan["scores"][start : start + 100]
            response = client.api.scores.create(
                request=[CreateScoreRequest(**score) for score in batch],
                request_options={"timeout_in_seconds": 20, "max_retries": 0},
            )
            if getattr(response, "rejected", 0):
                print(response.model_dump_json())
                raise SystemExit(
                    "Часть оценок отклонена; проверь ответ API, не повторяй вслепую."
                )
            print("Принято к обработке:", len(batch))
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
