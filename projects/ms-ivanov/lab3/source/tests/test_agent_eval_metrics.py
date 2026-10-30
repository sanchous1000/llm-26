import pytest

from source.agent.evaluation.dataset import parse_tasks
from source.agent.evaluation.metrics import (
    ranked_docs,
    retrieval_scores,
    summarise,
    summarise_by,
    task_success,
    tool_recall,
    turn_rows,
)


def spec(tid, kind, n_turns, **turn_extra):
    turns = [
        {
            "user": f"q{i}",
            "expected_output": "e",
            "depends_on_history": i > 0,
            "memory_probe": i == n_turns - 1 and n_turns > 2,
            **turn_extra,
        }
        for i in range(n_turns)
    ]
    return {
        "id": tid, "type": kind, "difficulty": "easy", "category": "c",
        "source": "own", "language": "en", "turns": turns,
    }  # fmt: skip


def record(task_id, turns, run="r1", agent="own"):
    return {
        "run": run,
        "agent": agent,
        "config": "cfg",
        "task_id": task_id,
        "turns": turns,
    }


def turn(i, correct, *, reduced=False, status="ok", cost=0.01, docs=(), tools=()):
    return {
        "turn": i, "status": status, "cost_usd": cost, "latency_s": 10.0,
        "input_tokens": 100, "output_tokens": 10, "steps": 2, "tool_calls": len(tools),
        "context_tokens_max": 5000, "reduced": reduced, "retrieved_docs": [list(docs)],
        "tools_used": list(tools), "judge": {"correct": correct, "score": float(correct)},
    }  # fmt: skip


def test_ranked_docs_keep_first_occurrence_order():
    assert ranked_docs([["a", "b"], ["b", None, "c"], ["a"]]) == ["a", "b", "c"]


def test_retrieval_scores():
    scores = retrieval_scores(["x", "ref1", "y", "ref2"], ["ref1", "ref2", "ref3"])

    assert scores["recall@5"] == pytest.approx(2 / 3)
    assert scores["precision@5"] == pytest.approx(2 / 5)
    assert scores["mrr"] == pytest.approx(1 / 2)
    assert retrieval_scores(["x"], ["ref"])["mrr"] == 0.0


def test_tool_recall():
    assert tool_recall(["search_papers", "calculator"], ["search_papers"]) == 0.5
    assert tool_recall([], ["x"]) is None


def test_rows_mark_turns_after_the_first_reduction():
    (task,) = parse_tasks(
        [
            spec(
                "D1",
                "dialog",
                3,
                reference_docs=["ref"],
                expected_tools=["search_papers"],
            )
        ]
    )
    rec = record("D1", [
        turn(0, True, docs=["ref"], tools=["search_papers"]),
        turn(1, False, reduced=True),
        turn(2, True),
    ])  # fmt: skip

    rows = turn_rows(task, rec)

    assert [r.after_reduction for r in rows] == [False, True, True]
    assert rows[0].retrieval["recall@5"] == 1.0 and rows[0].tool_recall == 1.0
    assert rows[1].retrieval["recall@5"] == 0.0 and rows[1].tool_recall == 0.0
    assert not task_success(rows)


def test_summary_splits_single_and_dialog_tasks_and_history_turns():
    tasks = parse_tasks(
        [spec("S1", "single", 1), spec("S2", "single", 1), spec("D1", "dialog", 3)]
    )
    rows = (
        turn_rows(tasks[0], record("S1", [turn(0, True, cost=0.02)]))
        + turn_rows(tasks[1], record("S2", [turn(0, False, status="step_limit")]))
        + turn_rows(tasks[2], record("D1", [turn(0, True), turn(1, True, reduced=True), turn(2, False)]))
    )  # fmt: skip

    stats = summarise(rows)

    assert stats["tasks"] == 3 and stats["turns"] == 5
    assert stats["success_single"] == 0.5
    assert stats["success_dialog"] == 0.0
    assert stats["success_rate"] == pytest.approx(1 / 3)
    assert stats["history_accuracy"] == 0.5
    assert stats["history_accuracy_before_reduction"] is None
    assert stats["history_accuracy_after_reduction"] == 0.5
    assert stats["memory_probe_accuracy"] == 0.0
    assert stats["step_limit_rate"] == pytest.approx(1 / 5)
    assert stats["cost_per_task_usd"] == pytest.approx((0.02 + 0.01 + 0.03) / 3)
    by_type = summarise_by(rows, lambda r: r.task_type)
    assert set(by_type) == {"dialog", "single"}


def test_report_tables_compare_runs(tmp_path):
    import json

    from source.agent.evaluation.report import write_report

    tasks = {
        t.id: t for t in parse_tasks([spec("S1", "single", 1), spec("D1", "dialog", 2)])
    }
    runs = {
        "own": [record("S1", [turn(0, True)]), record("D1", [turn(0, True), turn(1, True, reduced=True)])],
        "opencode": [record("S1", [turn(0, False, status="step_limit")], agent="opencode")],
    }  # fmt: skip
    out = tmp_path / "RESULTS.md"

    summaries = write_report(runs, tasks, out)

    text = out.read_text(encoding="utf-8")
    assert "| Метрика | own | opencode |" in text
    assert "| Task success | 100 % | 0 % |" in text
    assert "| S1 | ✓ 100 % | ✗ 0 % |" in text
    assert "| D1 | ✓ 100 % | — |" in text
    assert json.loads(out.with_suffix(".json").read_text())["own"]["tasks"] == 2
    assert summaries["opencode"]["step_limit_rate"] == 1.0
