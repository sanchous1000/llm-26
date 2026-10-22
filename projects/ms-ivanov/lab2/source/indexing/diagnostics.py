"""Embedding-free quality checks of a chunking against the evaluation set.

Each question carries the exact span of its answer passage, so we can ask
how a chunking treats that passage before computing a single vector:

* **contained** - the whole passage lies inside one chunk (the answer is not
  cut in half);
* **dilution** - passage tokens / tokens of the chunk that holds most of it
  (1.0 = the chunk is exactly the answer, small = the answer is a needle in
  a haystack, which weakens a dense embedding of the chunk).
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass

from source.indexing.chunking import Chunk
from source.indexing.tokens import TokenCounter


@dataclass(frozen=True)
class ChunkingReport:
    chunks: int
    tokens_median: float
    tokens_p90: int
    contained: float
    dilution_median: float
    questions: int


def evaluate_chunking(
    chunks_by_doc: dict[str, list[Chunk]],
    questions: list[dict],
    body_by_doc: dict[str, str],
    counter: TokenCounter,
) -> ChunkingReport:
    """Summarise chunk sizes and how well the question passages survive."""
    sizes = sorted(c.token_count for chunks in chunks_by_doc.values() for c in chunks)
    contained, dilutions = 0, []
    for question in questions:
        doc, evidence = question["doc_key"], question["evidence"]
        best = _best_chunk(
            chunks_by_doc.get(doc, []), evidence["start"], evidence["end"]
        )
        if best is None:
            continue
        chunk, overlap = best
        contained += overlap >= evidence["end"] - evidence["start"]
        passage = body_by_doc[doc][evidence["start"] : evidence["end"]]
        dilutions.append(min(1.0, counter.count(passage) / max(1, chunk.token_count)))
    return ChunkingReport(
        chunks=len(sizes),
        tokens_median=statistics.median(sizes),
        tokens_p90=sizes[int(0.9 * (len(sizes) - 1))],
        contained=contained / len(questions),
        dilution_median=statistics.median(dilutions),
        questions=len(questions),
    )


def _best_chunk(chunks: list[Chunk], start: int, end: int) -> tuple[Chunk, int] | None:
    scored = [(min(c.end, end) - max(c.start, start), c) for c in chunks]
    scored = [item for item in scored if item[0] > 0]
    if not scored:
        return None
    overlap, chunk = max(scored, key=lambda item: item[0])
    return chunk, overlap


def group_by_doc(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.doc_key].append(chunk)
    return grouped
