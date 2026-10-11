"""Vector search over a collection: dense, sparse or hybrid (RRF)."""

from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from source.indexing.embedding import EmbeddedText
from source.indexing.vectorstore import DENSE, SPARSE

SEARCH_LIMIT = 20
_HYBRID_PREFETCH_FACTOR = 4  # candidates taken from each branch before fusion


def sparse_query(vector: EmbeddedText) -> models.SparseVector | None:
    if vector.sparse is None:
        return None
    items = sorted(vector.sparse.items())
    return models.SparseVector(
        indices=[i for i, _ in items], values=[w for _, w in items]
    )


def document_filter(doc_keys: Sequence[str] | None) -> models.Filter | None:
    """Restrict a search to some documents (``None`` = the whole collection)."""
    if not doc_keys:
        return None
    condition = models.FieldCondition(
        key="doc_key", match=models.MatchAny(any=list(doc_keys))
    )
    return models.Filter(must=[condition])


def search(
    client: QdrantClient,
    collection: str,
    mode: str,
    vector: EmbeddedText,
    *,
    limit: int = SEARCH_LIMIT,
    params: models.SearchParams | None = None,
    query_filter: models.Filter | None = None,
):
    """Top ``limit`` points of a query (``dense`` / ``sparse`` / ``hybrid`` mode)."""
    sparse = sparse_query(vector)
    if mode == "dense":
        response = client.query_points(
            collection, query=vector.dense, using=DENSE, limit=limit,
            search_params=params, query_filter=query_filter, with_payload=True,
        )  # fmt: skip
    elif mode == "sparse":
        response = client.query_points(
            collection, query=sparse, using=SPARSE, limit=limit,
            query_filter=query_filter, with_payload=True,
        )  # fmt: skip
    else:
        depth = limit * _HYBRID_PREFETCH_FACTOR
        response = client.query_points(
            collection,
            prefetch=[
                models.Prefetch(
                    query=vector.dense, using=DENSE, limit=depth,
                    params=params, filter=query_filter,
                ),
                models.Prefetch(
                    query=sparse, using=SPARSE, limit=depth, filter=query_filter
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )  # fmt: skip
    return response.points
