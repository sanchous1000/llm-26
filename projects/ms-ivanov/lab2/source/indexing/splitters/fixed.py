"""Baseline: sliding window of ``chunk_size`` tokens."""

from source.indexing.config import ChunkingConfig
from source.indexing.splitters.types import Span
from source.indexing.tokens import TokenCounter


class FixedSplitter:
    def __init__(self, config: ChunkingConfig, counter: TokenCounter):
        self._size = config.chunk_size
        self._step = config.chunk_size - config.overlap
        self._counter = counter

    def split(self, text: str) -> list[Span]:
        offsets = self._counter.offsets(text)
        spans: list[Span] = []
        for first in range(0, len(offsets), self._step):
            window = offsets[first : first + self._size]
            spans.append((window[0][0], window[-1][1]))
            if first + self._size >= len(offsets):
                break
        return spans
