"""Splitting strategies: each returns the character spans of the chunks."""

from source.indexing.config import ChunkingConfig
from source.indexing.splitters.fixed import FixedSplitter
from source.indexing.splitters.recursive import RecursiveSplitter
from source.indexing.splitters.structured import StructuredSplitter
from source.indexing.splitters.types import Span, Splitter
from source.indexing.tokens import TokenCounter

_REGISTRY = {
    "fixed": FixedSplitter,
    "recursive": RecursiveSplitter,
    "structured": StructuredSplitter,
}


def build_splitter(config: ChunkingConfig, counter: TokenCounter) -> Splitter:
    """Create the splitter selected by ``config.splitter``."""
    return _REGISTRY[config.splitter](config, counter)


__all__ = ["Span", "Splitter", "build_splitter"]
