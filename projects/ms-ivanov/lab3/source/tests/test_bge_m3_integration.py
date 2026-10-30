"""Runs the real BGE-M3 model; skipped when the weights are not in the local cache."""

import math

import pytest
from huggingface_hub import try_to_load_from_cache

from source.indexing.config import EmbeddingConfig

WEIGHTS = try_to_load_from_cache("BAAI/bge-m3", "pytorch_model.bin")
pytestmark = pytest.mark.skipif(
    not isinstance(WEIGHTS, str), reason="bge-m3 not downloaded"
)


@pytest.fixture(scope="module")
def embedder():
    from source.indexing.embedding import build_embedder

    return build_embedder(
        EmbeddingConfig(device="cpu", dtype="float32", max_length=128)
    )


def test_vectors_are_normalised_multilingual_and_order_preserving(embedder):
    texts = [
        "Optimal transport moves one distribution onto another at minimal cost.",
        "Оптимальный транспорт переводит одно распределение в другое с минимальной стоимостью.",
        "Gradient boosting builds an ensemble of decision trees.",
    ]

    forward = embedder.embed(texts)
    reverse = embedder.embed(texts[::-1])

    assert embedder.dense_dim == 1024 and len(forward[0].dense) == 1024
    assert math.isclose(sum(x * x for x in forward[0].dense), 1.0, rel_tol=1e-4)
    cos = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    assert cos(forward[0].dense, forward[1].dense) > cos(
        forward[0].dense, forward[2].dense
    )
    assert cos(forward[0].dense, reverse[2].dense) > 0.999  # batching keeps order
    assert forward[0].sparse and all(w > 0 for w in forward[0].sparse.values())
    assert forward[0].n_tokens > 5


def test_dense_mode_returns_no_sparse_vector():
    from source.indexing.embedding import build_embedder

    dense_only = build_embedder(
        EmbeddingConfig(device="cpu", dtype="float32", mode="dense", max_length=64)
    )

    result = dense_only.embed(["short text for a dense-only embedding"])[0]

    assert result.dense is not None and result.sparse is None
