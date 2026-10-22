"""Embedder protocol, batch planning and the model registry."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from source.indexing.config import EmbeddingConfig


@dataclass(frozen=True)
class EmbeddedText:
    """Vectors of one text; fields not requested by the mode are ``None``."""

    dense: list[float] | None
    sparse: dict[int, float] | None
    n_tokens: int


class Embedder(Protocol):
    """Turns texts into dense and/or sparse vectors."""

    dense_dim: int

    def embed(
        self, texts: Sequence[str], on_batch: Callable[[int, int], None] | None = None
    ) -> list[EmbeddedText]:
        """Embed ``texts`` keeping order; ``on_batch(done, total)`` reports progress."""
        ...


def plan_batches(lengths: Sequence[int], token_budget: int) -> list[list[int]]:
    """Group text indices so that ``len(batch) * longest <= token_budget``.

    Texts are sorted by length (longest first) to keep padding small; a text
    longer than the budget gets a batch of its own.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        longest = lengths[current[0]] if current else lengths[index]
        if current and (len(current) + 1) * longest > token_budget:
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Instantiate the embedder for ``config.model``."""
    if config.model.lower() in ("baai/bge-m3", "bge-m3"):
        from source.indexing.embedding.bge_m3 import BgeM3Embedder

        return BgeM3Embedder(config)
    raise ValueError(
        f"unsupported embedding model {config.model!r}; "
        "add an adapter to source/indexing/embedding/ and register it in build_embedder"
    )
