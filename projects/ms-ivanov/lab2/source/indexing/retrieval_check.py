"""Retrieval quality against the evaluation questions.

A retrieved chunk is *relevant* when it belongs to the question's document and
covers at least ``MIN_COVERAGE`` of the question's evidence span. Metrics:

* Recall@k   - a relevant chunk is among the first k results;
* Precision@k - share of the first k results that are relevant;
* MRR        - mean of 1 / rank of the first relevant result;
* Recall@T   - the first relevant result lies within the first T retrieved
  tokens, i.e. fits a context budget of T tokens (fair across chunk sizes).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from source.indexing.embedding import Embedder
from source.indexing.search import search

MIN_COVERAGE = 0.5
KS = (1, 5, 10)
BUDGETS = (1024, 2048, 4096, 8192)  # tokens of retrieved context an LLM could get


@dataclass(frozen=True)
class QueryResult:
    """One query's ranked hits reduced to what the metrics need."""

    ids: tuple[str, ...]
    relevant: tuple[bool, ...]
    tokens: tuple[int, ...]

    @property
    def first_rank(self) -> int | None:
        return next((i for i, r in enumerate(self.relevant, start=1) if r), None)

    @property
    def tokens_to_first(self) -> int | None:
        rank = self.first_rank
        return sum(self.tokens[:rank]) if rank else None


@dataclass(frozen=True)
class RetrievalScores:
    mode: str
    recall: dict[int, float]
    precision: dict[int, float]
    mrr: float
    questions: int
    budget_recall: dict[int, float]


def to_result(hits, doc_key: str, evidence: dict) -> QueryResult:
    """Mark which hits are relevant for a question."""
    span = evidence["end"] - evidence["start"]
    relevant = []
    for hit in hits:
        payload = hit.payload
        covered = min(payload["end_char"], evidence["end"]) - max(
            payload["start_char"], evidence["start"]
        )
        relevant.append(
            payload["doc_key"] == doc_key and covered >= MIN_COVERAGE * span
        )
    return QueryResult(
        tuple(str(hit.id) for hit in hits),
        tuple(relevant),
        tuple(hit.payload["token_count"] for hit in hits),
    )


def summarise(mode: str, results: Sequence[QueryResult]) -> RetrievalScores:
    total = len(results)
    recall = {
        k: sum(1 for r in results if r.first_rank and r.first_rank <= k) / total
        for k in KS
    }
    precision = {k: sum(sum(r.relevant[:k]) / k for r in results) / total for k in KS}
    mrr = sum(1 / r.first_rank for r in results if r.first_rank) / total
    budget = {
        b: sum(1 for r in results if r.tokens_to_first and r.tokens_to_first <= b)
        / total
        for b in BUDGETS
    }
    return RetrievalScores(mode, recall, precision, mrr, total, budget)


CHECK_HEADER = (
    "| variant | mode | R@1 | R@5 | R@10 | P@5 | P@10 | MRR "
    "| T1024 | T2048 | T4096 | T8192 |"
)
CHECK_RULE = "|---|---|---|---|---|---|---|---|---|---|---|---|"


def format_check_row(label: str, s: RetrievalScores) -> str:
    """One Markdown table row of a scores object."""
    cells = [
        *(f"{s.recall[k]:.3f}" for k in KS),
        *(f"{s.precision[k]:.3f}" for k in KS[1:]),
        f"{s.mrr:.3f}",
        *(f"{s.budget_recall[b]:.3f}" for b in BUDGETS),
    ]
    return f"| {label} | {s.mode} | " + " | ".join(cells) + " |"


def check_retrieval(
    client: QdrantClient,
    collection: str,
    embedder: Embedder,
    questions: list[dict],
    modes: tuple[str, ...],
    *,
    language_field: str = "question",
    params: models.SearchParams | None = None,
) -> list[RetrievalScores]:
    """Scores of each retrieval ``mode`` over all questions."""
    vectors = embedder.embed([q[language_field] for q in questions])
    scores = []
    for mode in modes:
        results = [
            to_result(
                search(client, collection, mode, vector, params=params),
                q["doc_key"],
                q["evidence"],
            )
            for q, vector in zip(questions, vectors, strict=True)
        ]
        scores.append(summarise(mode, results))
    return scores
