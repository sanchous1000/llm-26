"""End-to-end check of the RAG pipeline on the evaluation questions.

For every sampled question the full pipeline runs (retrieve -> prompt -> LLM) and
the answer is compared with the reference answer by a second LLM call (judge).
Reported: whether the evidence passage reached the prompt / was cited, the
judge's verdict, citation quality and latency.
"""

import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from source.indexing.retrieval_check import MIN_COVERAGE
from source.rag.engine import RagEngine
from source.rag.errors import RagError
from source.rag.llm import Llm
from source.rag.models import Answer, RetrievedChunk

JUDGE_PROMPT = """\
You grade an answer to a question about a scientific paper against a reference answer.

Question: {question}

Reference answer: {reference}

Candidate answer: {candidate}

Verdict (choose one):
- "correct": the candidate states the key facts of the reference, no contradiction;
- "partial": on topic, but misses or gets wrong an important part of the reference;
- "incorrect": it contradicts the reference or answers something else;
- "no_answer": the candidate says that the answer is not available / it cannot answer.
Ignore wording, language and formatting. Reply with JSON only:
{{"verdict": "...", "reason": "<one short sentence>"}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
VERDICTS = ("correct", "partial", "incorrect", "no_answer")


@dataclass(frozen=True)
class RagEvalRow:
    question_id: str
    question: str
    verdict: str
    reason: str
    evidence_in_context: bool
    evidence_cited: bool
    citations: int
    verified_snippets: int
    unknown_citations: int
    retrieve_s: float
    llm_s: float
    answer: str

    def to_dict(self) -> dict:
        return asdict(self)


def sample_questions(questions: Sequence[dict], count: int, seed: int) -> list[dict]:
    """A reproducible random subset (all questions if ``count`` is large)."""
    pool = list(questions)
    random.Random(seed).shuffle(pool)
    return pool[:count]


def covers_evidence(chunk: RetrievedChunk, question: dict) -> bool:
    evidence = question["evidence"]
    if chunk.doc_key != question["doc_key"]:
        return False
    overlap = min(chunk.end_char, evidence["end"]) - max(
        chunk.start_char, evidence["start"]
    )
    return overlap >= MIN_COVERAGE * (evidence["end"] - evidence["start"])


def judge(llm: Llm, question: str, reference: str, candidate: str) -> tuple[str, str]:
    """``(verdict, reason)`` of the judge model; unparseable output is ``unparsed``."""
    raw = llm.complete(
        JUDGE_PROMPT.format(question=question, reference=reference, candidate=candidate)
    )
    match = _JSON_RE.search(raw)
    try:
        data = json.loads(match.group(0)) if match else {}
    except ValueError:
        data = {}
    verdict = str(data.get("verdict", "")).strip().lower()
    return (verdict if verdict in VERDICTS else "unparsed"), str(data.get("reason", ""))


def evaluate_question(
    engine: RagEngine, judge_llm: Llm, question: dict, *, top_k: int | None
) -> RagEvalRow:
    answer = engine.answer(question["question"], top_k=top_k)
    verdict, reason = judge(
        judge_llm, question["question"], question["reference_answer"], answer.text
    )
    return _row(question, answer, verdict, reason)


def _row(question: dict, answer: Answer, verdict: str, reason: str) -> RagEvalRow:
    return RagEvalRow(
        question_id=question["id"],
        question=question["question"],
        verdict=verdict,
        reason=reason,
        evidence_in_context=any(covers_evidence(c, question) for c in answer.contexts),
        evidence_cited=any(
            covers_evidence(c.chunk, question) for c in answer.citations
        ),
        citations=len(answer.citations),
        verified_snippets=sum(1 for c in answer.citations if c.verified),
        unknown_citations=len(answer.unknown_citations),
        retrieve_s=answer.timings.get("retrieve_s", 0.0),
        llm_s=answer.timings.get("llm_s", 0.0),
        answer=answer.text,
    )


def run_evaluation(
    engine: RagEngine,
    judge_llm: Llm,
    questions: Sequence[dict],
    *,
    top_k: int | None = None,
    on_row=None,
) -> list[RagEvalRow]:
    """Sequentially evaluate ``questions``; a failed question is recorded, not fatal."""
    rows = []
    for question in questions:
        try:
            row = evaluate_question(engine, judge_llm, question, top_k=top_k)
        except RagError as error:
            row = RagEvalRow(question["id"], question["question"], "error", str(error),
                             False, False, 0, 0, 0, 0.0, 0.0, "")  # fmt: skip
        rows.append(row)
        if on_row:
            on_row(row)
    return rows


def summarise(rows: Sequence[RagEvalRow]) -> dict:
    """Aggregate numbers for the report."""
    total = len(rows) or 1
    verdicts = Counter(r.verdict for r in rows)
    cited = sum(r.citations for r in rows)
    judged_ok = [r for r in rows if r.verdict in ("correct", "partial")]
    return {
        "questions": len(rows),
        "verdicts": {v: verdicts.get(v, 0) for v in (*VERDICTS, "unparsed", "error")},
        "correct": verdicts["correct"] / total,
        "correct_or_partial": (verdicts["correct"] + verdicts["partial"]) / total,
        "evidence_in_context": sum(r.evidence_in_context for r in rows) / total,
        "evidence_cited": sum(r.evidence_cited for r in rows) / total,
        "verified_snippet_share": (
            (sum(r.verified_snippets for r in rows) / cited) if cited else 0.0
        ),
        "unknown_citations": sum(r.unknown_citations for r in rows),
        "answered_with_citation": sum(1 for r in judged_ok if r.citations)
        / max(len(judged_ok), 1),
        "mean_retrieve_s": sum(r.retrieve_s for r in rows) / total,
        "mean_llm_s": sum(r.llm_s for r in rows) / total,
    }
