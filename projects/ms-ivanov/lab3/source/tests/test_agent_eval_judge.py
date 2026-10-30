import asyncio

from source.agent.evaluation.dataset import parse_tasks
from source.agent.evaluation.judge import (
    judge_prompt,
    judge_records,
    load_judge,
    parse_verdict,
    push_scores,
)


def make_tasks():
    raw = [
        {
            "id": "S1", "type": "single", "difficulty": "easy", "category": "catalog",
            "source": "own", "language": "en",
            "turns": [{"user": "How many?", "expected_output": "12", "must_include": [["12"]], "check": "exact"}],
        },
        {
            "id": "D1", "type": "dialog", "difficulty": "hard", "category": "long_dialog",
            "source": "own", "language": "en",
            "turns": [
                {"user": "first", "expected_output": "Chazal 2011", "must_include": [["Chazal"]]},
                {"user": "and then?", "expected_output": "Theorem 3.5", "depends_on_history": True},
            ],
        },
    ]  # fmt: skip
    return {t.id: t for t in parse_tasks(raw)}


def record(task_id, answers, statuses=None):
    statuses = statuses or ["ok"] * len(answers)
    return {
        "run": "r", "agent": "own", "config": "c", "task_id": task_id, "item_trace_id": "trace-item",
        "turns": [
            {"turn": i, "answer": a, "status": s, "observation_id": f"obs-{i}", "trace_id": "trace-item"}
            for i, (a, s) in enumerate(zip(answers, statuses, strict=True))
        ],
    }  # fmt: skip


def test_exact_turns_need_no_llm_and_empty_answers_fail():
    calls = []

    async def ask(prompt):
        calls.append(prompt)
        return '{"verdict": "correct", "reason": "ok"}'

    tasks = make_tasks()
    records = [
        record("S1", ["There are 12."]),
        record("D1", ["(OpenCode stopped)", "Theorem 3.5"]),
    ]
    _, template = load_judge()

    judged = asyncio.run(judge_records(records, tasks, ask, template))

    s1 = judged[0]["turns"][0]["judge"]
    assert s1["method"] == "exact" and s1["correct"] and s1["score"] == 1.0
    d1 = [t["judge"] for t in judged[1]["turns"]]
    assert d1[0]["method"] == "rule" and not d1[0]["correct"]
    assert d1[1]["method"] == "llm" and d1[1]["correct"]
    assert len(calls) == 1 and "- first" in calls[0]  # earlier user turn shown
    assert "judge" not in records[0]["turns"][0]  # input not mutated


def test_verdict_parsing():
    assert (
        parse_verdict('think...\n{"verdict": "partially_correct", "reason": "x"}')[
            "verdict"
        ]
        == "partially_correct"
    )
    broken = parse_verdict("I believe it is right")
    assert broken["verdict"] == "incorrect" and "unparsable" in broken["reason"]


def test_prompt_marks_found_and_missing_facts():
    tasks = make_tasks()
    _, template = load_judge()

    prompt = judge_prompt(template, tasks["D1"], 0, "Chazal et al. proved it")

    assert "Chazal: found" in prompt
    assert "(none)" in prompt  # no earlier user turns


class FakeClient:
    def __init__(self):
        self.scores = []

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        pass


def test_scores_go_to_turns_tasks_and_the_run():
    tasks = make_tasks()
    judged = [
        {**record("S1", ["12"]), "turns": [{**record("S1", ["12"])["turns"][0], "judge": {"correct": True, "score": 1.0, "reason": "ok"}}]},
    ]  # fmt: skip
    client = FakeClient()

    summary = push_scores(client, judged, tasks, experiment_id="exp-1")

    names = [s["name"] for s in client.scores]
    assert (
        "turn_correct" in names and "task_success" in names and "success_rate" in names
    )
    turn_score = next(s for s in client.scores if s["name"] == "turn_correct")
    assert (
        turn_score["observation_id"] == "obs-0"
        and turn_score["trace_id"] == "trace-item"
    )
    run_score = next(s for s in client.scores if s["name"] == "success_rate")
    assert run_score["dataset_run_id"] == "exp-1" and run_score["value"] == 1.0
    assert summary["success_rate"] == 1.0


def test_verdicts_with_latex_backslashes_are_read():
    text = '{"verdict": "correct", "reason": "states \\(m^{-1/2}\\) W_2 bound"}'

    parsed = parse_verdict(text)

    assert parsed["verdict"] == "correct" and "W_2" in parsed["reason"]


def test_empty_judge_output_is_retried_and_good_verdicts_are_reused():
    replies = iter(["", '{"verdict": "correct", "reason": "ok"}'])
    calls = []

    async def ask(prompt):
        calls.append(prompt)
        return next(replies)

    tasks = make_tasks()
    _, template = load_judge()
    fresh = record("D1", ["Chazal 2011", "Theorem 3.5"])
    previous = {
        ("D1", 0): {"method": "llm", "verdict": "correct", "reason": "fine"},
        ("D1", 1): {"method": "llm", "verdict": "incorrect", "reason": "unparsable judge output: "},
    }  # fmt: skip

    (judged,) = asyncio.run(
        judge_records([fresh], tasks, ask, template, previous=previous)
    )

    assert [t["judge"]["verdict"] for t in judged["turns"]] == ["correct", "correct"]
    assert len(calls) == 2  # only turn 1 re-graded, once retried after an empty reply


def test_score_ids_are_stable_so_a_second_push_overwrites():
    tasks = make_tasks()
    judged = [
        {**record("S1", ["12"]), "turns": [{**record("S1", ["12"])["turns"][0], "judge": {"correct": True, "score": 1.0, "reason": "ok"}}]},
    ]  # fmt: skip
    first, second = FakeClient(), FakeClient()

    push_scores(first, judged, tasks, experiment_id="exp-1")
    push_scores(second, judged, tasks, experiment_id="exp-1")

    ids = [s["score_id"] for s in first.scores]
    assert len(set(ids)) == len(ids)
    assert ids == [s["score_id"] for s in second.scores]
