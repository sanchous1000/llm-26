import json
from pathlib import Path

import pytest

from source.rag.commands import mcp_config_text
from source.rag.engine import RagEngine
from source.rag.errors import RagError
from source.rag.evaluation import (
    covers_evidence,
    judge,
    run_evaluation,
    sample_questions,
    summarise,
)
from source.tests.test_rag import ScriptedLlm, chunk


def question(qid="q1", doc="doc0", start=0, end=100) -> dict:
    return {
        "id": qid, "doc_key": doc, "question": "anything", "reference_answer": "ref",
        "evidence": {"start": start, "end": end},
    }  # fmt: skip


def test_sampling_is_reproducible_and_bounded():
    pool = [question(f"q{i}") for i in range(20)]

    first = sample_questions(pool, 5, seed=3)

    assert first == sample_questions(pool, 5, seed=3) and len(first) == 5
    assert first != sample_questions(pool, 5, seed=4)
    assert len(sample_questions(pool, 100, seed=1)) == 20


def test_evidence_coverage_needs_the_right_document_and_half_the_span():
    q = question(doc="d", start=100, end=200)

    assert covers_evidence(chunk(doc_key="d", start_char=90, end_char=300), q)
    assert covers_evidence(chunk(doc_key="d", start_char=150, end_char=400), q)
    assert not covers_evidence(chunk(doc_key="d", start_char=180, end_char=400), q)
    assert not covers_evidence(chunk(doc_key="other", start_char=0, end_char=500), q)


@pytest.mark.parametrize(
    ("raw", "verdict"),
    [
        ('{"verdict": "correct", "reason": "ok"}', "correct"),
        ('Sure!\n```json\n{"verdict": "Partial", "reason": "x"}\n```', "partial"),
        ('{"verdict": "excellent"}', "unparsed"),
        ("no json at all", "unparsed"),
    ],
)
def test_judge_parses_verdicts_defensively(raw, verdict):
    assert judge(ScriptedLlm([raw]), "q", "ref", "cand")[0] == verdict


def test_run_evaluation_records_each_question_and_survives_failures(
    retriever,
):  # noqa: F811
    retr, cfg = retriever
    answers = ScriptedLlm(["Answer one [1].", "Answer two [1]."])

    class Failing:
        model = "failing"

        def complete(self, prompt):
            raise RagError("boom")

    judge_llm = ScriptedLlm(['{"verdict": "correct", "reason": "r"}'] * 2)
    questions = [question("a"), question("b")]

    rows = run_evaluation(
        RagEngine(retr, answers, cfg.rag), judge_llm, questions, top_k=2
    )
    failed = run_evaluation(
        RagEngine(retr, Failing(), cfg.rag), judge_llm, questions[:1]
    )

    assert [r.verdict for r in rows] == ["correct", "correct"] and all(
        r.citations == 1 for r in rows
    )
    assert failed[0].verdict == "error" and "boom" in failed[0].reason
    summary = summarise([*rows, *failed])
    assert summary["questions"] == 3 and summary["verdicts"]["correct"] == 2
    assert (
        summary["correct"] == pytest.approx(2 / 3) and summary["verdicts"]["error"] == 1
    )


def test_mcp_config_contains_absolute_paths_and_valid_json():
    text = mcp_config_text("/venv/bin/python", Path("/work/lab2"))

    block = text[text.index("{") : text.rindex("}") + 1]
    entry = json.loads(block)["mcpServers"]["papers-rag"]
    assert (
        entry["command"] == "/venv/bin/python"
        and entry["env"]["PYTHONPATH"] == "/work/lab2"
    )
    assert "claude mcp add papers-rag" in text and "docker compose up -d" in text
