"""Embedding models behind one interface."""

from source.indexing.embedding.base import (
    EmbeddedText,
    Embedder,
    build_embedder,
    plan_batches,
)

__all__ = ["EmbeddedText", "Embedder", "build_embedder", "plan_batches"]
