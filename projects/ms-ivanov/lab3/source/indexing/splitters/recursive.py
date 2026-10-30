"""Baseline: recursive splitting at paragraph/line/sentence/word boundaries."""

import semchunk

from source.indexing.config import ChunkingConfig
from source.indexing.splitters.types import Span
from source.indexing.tokens import TokenCounter


class RecursiveSplitter:
    def __init__(self, config: ChunkingConfig, counter: TokenCounter):
        self._size = config.chunk_size
        self._overlap = config.overlap or None
        self._counter = counter

    def split(self, text: str) -> list[Span]:
        _, offsets = semchunk.chunk(
            text, self._size, self._counter.count, offsets=True, overlap=self._overlap
        )
        return [tuple(pair) for pair in offsets]
