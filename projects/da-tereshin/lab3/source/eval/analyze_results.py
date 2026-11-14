"""Считает метрики двух прогонов по ручным оценкам и выгруженным наблюдениям Langfuse."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def decode(value):
    return json.loads(value) if isinstance(value, str) and value else value


def aggregate(values):
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "max": max(values) if values else None,
    }


def rate(values):
    return {
        "passed": sum(values),
        "total": len(values),
        "rate": mean(values) if values else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("artifacts/eval/analysis-20261113-v2"),
        help="Каталог ручных оценок, наблюдений Langfuse и итоговых метрик; относительно lab3 или абсолютный путь.",
    )
    parser.add_argument(
        "--custom-run",
        default="20261113T092058Z-custom-5bf056a4",
        help="ID прогона своего агента в artifacts/eval/.",
    )
    parser.add_argument(
        "--smolagents-run",
        default="20261113T104017Z-smolagents-c197d102",
        help="ID прогона smolagents в artifacts/eval/.",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    output = project / args.analysis_dir
    review = read_json(output / "manual_review.json")
    tasks = read_json(project / "docs/evaluation/tasks.json")
    turns = {turn["id"]: turn for task in tasks for turn in task["turns"]}
    runs = {
        "custom": args.custom_run,
        "smolagents": args.smolagents_run,
    }
    summary = {}
    per_turn, per_task = [], []
    for agent, run_id in runs.items():
        records = [
            json.loads(line)
            for line in (project / "artifacts/eval" / run_id / "results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert {r["turn_id"] for r in records} == set(turns)
        task_rows = {}
        reductions_seen, summaries_seen = {}, {}
        for record in records:
            observations = read_json(
                output / "langfuse" / (record["trace_id"] + ".json")
            )
            assert len({o["id"] for o in observations}) == len(observations)
            by_id = {o["id"]: o for o in observations}
            generations = [o for o in observations if o["type"] == "GENERATION"]
            tools = [
                o
                for o in observations
                if o["type"] == "TOOL"
                and o["name"] in {"search_theory", "get_scene", "save_study_plan"}
            ]
            prepare_stats = [
                decode(o["output"])["context_stats"]
                for o in observations
                if o["name"] == "prepare" and o.get("output")
            ]
            reduced = any(
                s.get("fits") and any(step["applied"] for step in s["steps"])
                for s in prepare_stats
            )
            updated_summary = any(
                step.get("updated") for s in prepare_stats for step in s["steps"]
            )
            session = record["session_id"]
            reductions_seen[session] = reductions_seen.get(session, False) or reduced
            summaries_seen[session] = (
                summaries_seen.get(session, False) or updated_summary
            )
            required = set(turns[record["turn_id"]]["metadata"]["expected_tools"])
            called = {o["name"] for o in tools}
            reason = review["failed_turns"][agent].get(record["turn_id"])
            row = {
                "agent": agent,
                "run_id": run_id,
                "task_id": record["task_id"],
                "turn_id": record["turn_id"],
                "trace_id": record["trace_id"],
                "execution_ok": record["status"] == "ok",
                "manual_success": reason is None,
                "failure_reason": reason,
                "history_dependent": turns[record["turn_id"]]["metadata"][
                    "history_dependent"
                ],
                "history_correct": record["turn_id"]
                not in review["history_failed_turns"][agent],
                "after_reduction": reductions_seen[session],
                "after_summary": summaries_seen[session],
                "required_tools": sorted(required),
                "called_tools": sorted(called),
                "tool_correctness": required <= called if required else None,
                "tool_calls": len(tools),
                "tool_counts": dict(Counter(o["name"] for o in tools)),
                "generation_count": len(generations),
                "generation_errors": sum(
                    o.get("level") == "ERROR" for o in generations
                ),
                "summary_generation_count": sum(
                    by_id.get(o["parentObservationId"], {}).get("name")
                    == "context.summary"
                    for o in generations
                ),
                "input_tokens": sum(o.get("inputUsage", 0) or 0 for o in generations),
                "output_tokens": sum(o.get("outputUsage", 0) or 0 for o in generations),
                "total_tokens": sum(o.get("totalUsage", 0) or 0 for o in generations),
                "cost_usd_observed": sum(
                    o.get("totalCost", 0) or 0 for o in generations
                ),
                "cost_missing_generations": sum(
                    o.get("totalCost") is None for o in generations
                ),
                "local_elapsed_seconds": record["elapsed_seconds"],
                "root_latency_seconds": next(
                    (
                        o["latency"]
                        for o in observations
                        if o["name"]
                        == ("user-turn" if agent == "custom" else "smolagents-turn")
                    ),
                    None,
                ),
                "max_context_before": max(
                    (s["estimated_tokens_before"] for s in prepare_stats), default=None
                ),
                "max_context_after": max(
                    (s["estimated_tokens_after"] for s in prepare_stats), default=None
                ),
                "context_reduction_calls": sum(
                    step["applied"] for s in prepare_stats for step in s["steps"]
                ),
                "sliding_window_applied": any(
                    step["applied"] and step["strategy"] == "sliding_window"
                    for s in prepare_stats
                    for step in s["steps"]
                ),
                "iterations": record.get("iterations"),
                "context_limit": any(
                    o.get("statusMessage") == "context_limit" for o in observations
                ),
                "iteration_limit": (
                    record.get("iteration_limit", False)
                    or any(
                        o.get("statusMessage") == "iteration_limit"
                        for o in observations
                    )
                    or "Reached max steps."
                    in (project / "artifacts/eval" / run_id / record["log"]).read_text(
                        encoding="utf-8"
                    )
                ),
                "timeout": record["status"] == "timeout",
            }
            task_rows.setdefault(record["task_id"], []).append((row, observations))
            per_turn.append(row)
        current_tasks = []
        for task in tasks:
            pairs = task_rows[task["id"]]
            rows = [pair[0] for pair in pairs]
            gold = {(r["source"], r["page"]) for r in task["metadata"]["references"]}
            retrieval = []
            for _, observations in pairs:
                for observation in observations:
                    if observation["type"] != "RETRIEVER":
                        continue
                    chunks = decode(observation["output"])
                    if not isinstance(chunks, list):
                        continue
                    chunks = sorted(chunks, key=lambda c: c["rank"])[:10]
                    found = {(c["source"], c["page"]) for c in chunks}
                    first = next(
                        (
                            i
                            for i, c in enumerate(chunks, 1)
                            if (c["source"], c["page"]) in gold
                        ),
                        None,
                    )
                    if gold:
                        retrieval.append(
                            {
                                "recall_at_10": len(found & gold) / len(gold),
                                "precision_at_10": sum(
                                    (c["source"], c["page"]) in gold for c in chunks
                                )
                                / 10,
                                "mrr_at_10": 1 / first if first else 0,
                            }
                        )
            metrics = (
                {
                    key: mean([r[key] for r in retrieval]) if retrieval else 0
                    for key in ["recall_at_10", "precision_at_10", "mrr_at_10"]
                }
                if gold
                else None
            )
            row = {
                "agent": agent,
                "task_id": task["id"],
                "trace_ids": [r["trace_id"] for r in rows],
                "type": task["metadata"]["type"],
                "difficulty": task["metadata"]["difficulty"],
                "task_success": all(r["manual_success"] for r in rows),
                "execution_ok": all(r["execution_ok"] for r in rows),
                "failed_turns": [r["turn_id"] for r in rows if not r["manual_success"]],
                "local_elapsed_seconds": sum(r["local_elapsed_seconds"] for r in rows),
                "root_latency_seconds": sum(
                    r["root_latency_seconds"]
                    for r in rows
                    if r["root_latency_seconds"] is not None
                ),
                "tokens_observed": sum(r["total_tokens"] for r in rows),
                "cost_usd_observed": sum(r["cost_usd_observed"] for r in rows),
                "tool_calls": sum(r["tool_calls"] for r in rows),
                "retrieval_calls_scored": len(retrieval),
                "retrieval_diagnostic": metrics,
            }
            current_tasks.append(row)
            per_task.append(row)
        rows = [r for r in per_turn if r["agent"] == agent]
        history = [r for r in rows if r["history_dependent"]]
        retrieval_tasks = [
            t for t in current_tasks if t["retrieval_diagnostic"] is not None
        ]
        complete_generations = all(r["generation_count"] > 0 for r in rows)
        summary[agent] = {
            "run_id": run_id,
            "task_success": rate([t["task_success"] for t in current_tasks]),
            "turn_success": rate([r["manual_success"] for r in rows]),
            "execution_success": rate([r["execution_ok"] for r in rows]),
            "history_compliance": rate([r["history_correct"] for r in history]),
            "history_turn_success": rate([r["manual_success"] for r in history]),
            "by_type": {
                kind: rate(
                    [t["task_success"] for t in current_tasks if t["type"] == kind]
                )
                for kind in sorted({t["type"] for t in current_tasks})
            },
            "by_difficulty": {
                kind: rate(
                    [
                        t["task_success"]
                        for t in current_tasks
                        if t["difficulty"] == kind
                    ]
                )
                for kind in sorted({t["difficulty"] for t in current_tasks})
            },
            "history_before_reduction": (
                rate([r["manual_success"] for r in history if not r["after_reduction"]])
                if agent == "custom"
                else None
            ),
            "history_after_reduction": (
                rate([r["manual_success"] for r in history if r["after_reduction"]])
                if agent == "custom"
                else None
            ),
            "tool_correctness": rate(
                [
                    r["tool_correctness"]
                    for r in rows
                    if r["tool_correctness"] is not None
                ]
            ),
            "total_tool_calls": sum(r["tool_calls"] for r in rows),
            "tool_counts": dict(
                sum((Counter(r["tool_counts"]) for r in rows), Counter())
            ),
            "main_model_calls": (
                sum(r["iterations"] or 0 for r in rows) if agent == "custom" else None
            ),
            "generations_observed": sum(r["generation_count"] for r in rows),
            "generation_errors": sum(r["generation_errors"] for r in rows),
            "cost_missing_generations": sum(
                r["cost_missing_generations"] for r in rows
            ),
            "generation_coverage": rate([r["generation_count"] > 0 for r in rows]),
            "summary_generations": sum(r["summary_generation_count"] for r in rows),
            "input_tokens": (
                sum(r["input_tokens"] for r in rows) if complete_generations else None
            ),
            "output_tokens": (
                sum(r["output_tokens"] for r in rows) if complete_generations else None
            ),
            "total_tokens": (
                sum(r["total_tokens"] for r in rows) if complete_generations else None
            ),
            "cost_usd": (
                sum(r["cost_usd_observed"] for r in rows)
                if complete_generations
                else None
            ),
            "mean_task_cost_usd": (
                mean([t["cost_usd_observed"] for t in current_tasks])
                if complete_generations
                else None
            ),
            "partial_tokens_observed": sum(r["total_tokens"] for r in rows),
            "partial_cost_usd_observed": sum(r["cost_usd_observed"] for r in rows),
            "task_local_elapsed_seconds": aggregate(
                [t["local_elapsed_seconds"] for t in current_tasks]
            ),
            "task_root_latency_seconds": aggregate(
                [t["root_latency_seconds"] for t in current_tasks]
            ),
            "turn_local_elapsed_seconds": aggregate(
                [r["local_elapsed_seconds"] for r in rows]
            ),
            "max_context_before": max(
                (
                    r["max_context_before"]
                    for r in rows
                    if r["max_context_before"] is not None
                ),
                default=None,
            ),
            "max_context_after": max(
                (
                    r["max_context_after"]
                    for r in rows
                    if r["max_context_after"] is not None
                ),
                default=None,
            ),
            "tasks_with_execution_error_or_iteration_limit": rate(
                [
                    not t["execution_ok"]
                    or any(
                        r["iteration_limit"] or r["generation_errors"]
                        for r in rows
                        if r["task_id"] == t["task_id"]
                    )
                    for t in current_tasks
                ]
            ),
            "timeouts": sum(r["timeout"] for r in rows),
            "context_limits": sum(r["context_limit"] for r in rows),
            "iteration_limits_observed": sum(r["iteration_limit"] for r in rows),
            "retrieval_macro_diagnostic": {
                key: mean([t["retrieval_diagnostic"][key] for t in retrieval_tasks])
                for key in ["recall_at_10", "precision_at_10", "mrr_at_10"]
            },
            "retrieval_tasks_scored": len(retrieval_tasks),
        }
    for name, data in [("per_turn.jsonl", per_turn), ("per_task.jsonl", per_task)]:
        (output / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data),
            encoding="utf-8",
        )
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
