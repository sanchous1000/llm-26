"""Tables and the selection rule of the HNSW study."""

from collections.abc import Sequence

from source.indexing.hnsw_study import StudyRow

# The approximate top-10 must match the exact one this well. 0.995 (not 0.99)
# leaves headroom: ANN recall drops as the corpus grows, so a setting that only
# just reaches the target today would miss it after scaling to the full corpus.
ANN_RECALL_TARGET = 0.995
QUALITY_TOLERANCE = 0.005  # allowed loss of Recall@10 and MRR vs exact search
MAX_SEARCH_EF = 64  # search effort we are willing to pay per query

_HEADER = (
    "| m | ef_construct | quant | ef | rescore | ANN R@10 | R@5 | R@10 | P@5 | MRR "
    "| p50 ms | p95 ms | build s | graph MB |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def format_row(row: StudyRow) -> str:
    ef = "exact" if row.ef is None else str(row.ef)
    build = "-" if row.ef is None else f"{row.build_s:.1f}"
    graph = "-" if row.ef is None else f"{row.graph_mb:.1f}"
    return (
        f"| {row.m} | {row.ef_construct} | {row.quantization} | {ef} | {row.rescore} "
        f"| {row.ann_recall:.3f} | {row.recall5:.3f} | {row.recall10:.3f} "
        f"| {row.precision5:.3f} | {row.mrr:.3f} | {row.p50_ms:.2f} | {row.p95_ms:.2f} "
        f"| {build} | {graph} |"
    )


def format_table(rows: Sequence[StudyRow]) -> str:
    return "\n".join([_HEADER, *(format_row(row) for row in rows)])


def exact_baseline(rows: Sequence[StudyRow]) -> StudyRow | None:
    return next((row for row in rows if row.ef is None), None)


def good_enough(
    rows: Sequence[StudyRow], *, max_ef: int = MAX_SEARCH_EF
) -> list[StudyRow]:
    """HNSW settings that lose (almost) nothing against exact search."""
    exact = exact_baseline(rows)
    if exact is None:
        return []
    return [
        row
        for row in rows
        if row.ef is not None
        and row.ef <= max_ef
        and row.quantization == "none"
        and row.ann_recall >= ANN_RECALL_TARGET
        and row.recall10 >= exact.recall10 - QUALITY_TOLERANCE
        and row.mrr >= exact.mrr - QUALITY_TOLERANCE
    ]


def recommend(
    rows: Sequence[StudyRow], *, max_ef: int = MAX_SEARCH_EF
) -> StudyRow | None:
    """Cheapest acceptable setting.

    Among the settings that meet the targets within the search-effort budget,
    take the smallest graph (``m``), then the cheapest build (``ef_construct``),
    then the smallest search ``ef``.
    """
    candidates = good_enough(rows, max_ef=max_ef)
    if not candidates:
        return None
    return min(candidates, key=lambda r: (r.m, r.ef_construct, r.ef or 0))
