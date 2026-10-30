import json

import pytest
import yaml

from source.agent.evaluation.dataset import (
    DEFAULT_TASKS_PATH,
    DatasetError,
    load_tasks,
    missing_facts,
    parse_tasks,
    summarise,
    to_langfuse_item,
    unknown_doc_keys,
)


def turn(user="question?", **extra) -> dict:
    return {"user": user, "expected_output": "reference", **extra}


def task(tid="T1", kind="single", turns=None, **extra) -> dict:
    return {
        "id": tid, "type": kind, "difficulty": "easy", "category": "fact_lookup",
        "source": "own", "language": "en", "turns": turns or [turn()], **extra,
    }  # fmt: skip


def dialog(tid="D1", n_turns=3, **extra) -> dict:
    turns = [turn(f"q{i}", depends_on_history=i > 0) for i in range(n_turns)]
    return task(tid, kind="dialog", turns=turns, **extra)


def test_real_task_set_is_valid_and_covers_the_requirements():
    tasks = load_tasks(DEFAULT_TASKS_PATH)
    stats = summarise(tasks)

    assert 15 <= stats["tasks"] <= 30
    assert stats["by_type"]["dialog"] >= 5
    assert stats["long_dialogs"] >= 2
    assert stats["history_turns"] > 0
    assert stats["memory_probes"] > 0
    assert stats["exact_checks"] > 0
    assert set(stats["by_difficulty"]) == {"easy", "medium", "hard"}
    assert len({t.id for t in tasks}) == len(tasks)


def test_parse_builds_immutable_tasks_with_defaults():
    (parsed,) = parse_tasks([task(turns=[turn(expected_tools=["search_papers"])])])

    assert parsed.is_single
    assert parsed.turns[0].expected_tools == ("search_papers",)
    assert parsed.turns[0].depends_on_history is False
    with pytest.raises(AttributeError):
        parsed.id = "other"  # frozen


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([task("A"), task("A")], "duplicate"),
        ([task(turns=[turn(), turn()])], "exactly one turn"),
        ([dialog(n_turns=1)], "at least two turns"),
        ([task(turns=[turn(memory_probe=True)])], "memory_probe"),
        ([task(turns=[turn(expected_tools=["web_search"])])], "unknown tool"),
        ([task(turns=[turn(user="  ")])], "user"),
        ([task(difficulty="extreme")], "difficulty"),
        ([task(turns=[turn(must_include=[[]])])], "must_include"),
        ([task(turns=[turn(check="exact")])], "must_include"),
        ([task(turns=[turn(check="fuzzy")])], "check"),
    ],
)
def test_invalid_tasks_are_rejected_with_a_clear_message(raw, message):
    with pytest.raises(DatasetError, match=message):
        parse_tasks(raw)


def test_first_turn_of_a_dialog_cannot_depend_on_history():
    turns = [turn(depends_on_history=True), turn()]

    with pytest.raises(DatasetError, match="first turn"):
        parse_tasks([task(kind="dialog", turns=turns)])


def test_missing_facts_reports_unmet_groups_case_insensitively():
    groups = (("Adam",), ("0.0002", "2e-4"), ("lambda = 10",))

    missing = missing_facts("We use ADAM with lr 2E-4.", groups)

    assert missing == [("lambda = 10",)]
    assert missing_facts("anything", ()) == []


def test_langfuse_item_keeps_turns_expectations_and_metadata():
    (parsed,) = parse_tasks([dialog("D7", n_turns=2, long_dialog=True)])

    item = to_langfuse_item(parsed)

    assert item["id"] == "D7"
    assert item["input"] == {"turns": ["q0", "q1"]}
    assert [e["turn"] for e in item["expected_output"]] == [0, 1]
    assert item["metadata"]["type"] == "dialog"
    assert item["metadata"]["long_dialog"] is True
    assert item["metadata"]["history_turns"] == [1]
    json.dumps(item)  # serialisable


def test_unknown_doc_keys_lists_references_missing_from_the_corpus():
    turns = [turn(reference_docs=["known", "ghost"])]
    tasks = parse_tasks([task(turns=turns)])

    assert unknown_doc_keys(tasks, {"known"}) == {"T1": ["ghost"]}


def test_load_reports_a_broken_file(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text(yaml.safe_dump({"items": []}), encoding="utf-8")

    with pytest.raises(DatasetError, match="tasks"):
        load_tasks(path)
