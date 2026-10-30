"""Structure-aware splitter for scientific Markdown (the default).

Units (see :mod:`source.indexing.units`) are packed greedily:

* headings up to ``split_level`` always open a new chunk (unless the current
  chunk is still smaller than ``min_chunk_tokens``);
* when a unit does not fit, the cut moves back to the latest natural boundary
  (a heading or a theorem/definition/proof) if that keeps the chunk at least
  half full, so chunks end between ideas rather than inside one;
* ``overlap`` repeats whole trailing units of the previous chunk;
* a unit larger than ``chunk_size`` is cut by the recursive splitter.
"""

import semchunk

from source.indexing import blocks
from source.indexing.config import ChunkingConfig
from source.indexing.splitters.types import Span
from source.indexing.tokens import TokenCounter
from source.indexing.units import Unit, build_units

_LOOKBACK_FILL = 0.5


class StructuredSplitter:
    def __init__(self, config: ChunkingConfig, counter: TokenCounter):
        self._config = config
        self._counter = counter

    def split(self, text: str) -> list[Span]:
        units = build_units(text, blocks.parse_blocks(text), self._counter)
        packer = _Packer(text, self._config, self._counter)
        for unit in units:
            packer.add(unit)
        return packer.finish()


class _Packer:
    """Greedy packing state: closed spans plus the units of the open chunk."""

    def __init__(self, text: str, config: ChunkingConfig, counter: TokenCounter):
        self._text = text
        self._config = config
        self._counter = counter
        self._spans: list[Span] = []
        self._sizes: list[int] = []
        self._current: list[Unit] = []

    def add(self, unit: Unit) -> None:
        cfg = self._config
        size = _tokens(self._current)
        if unit.tokens > cfg.chunk_size:
            self._close(self._current)
            self._current = []
            self._split_oversize(unit)
            return
        opens_section = 0 < unit.heading_level <= cfg.split_level
        if self._current and opens_section and size >= cfg.min_chunk_tokens:
            self._close(self._current)
            self._current = [unit]
        elif self._current and size + unit.tokens > cfg.chunk_size:
            closed, carried = self._cut(self._current)
            self._close(closed)
            self._current = [*self._overlap_tail(closed), *carried, unit]
        else:
            self._current.append(unit)

    def finish(self) -> list[Span]:
        self._close(self._current)
        self._current = []
        self._merge_tiny_tail()
        return self._spans

    def _close(self, units: list[Unit]) -> None:
        if units:
            self._spans.append((units[0].start, units[-1].end))
            self._sizes.append(_tokens(units))

    def _cut(self, current: list[Unit]) -> tuple[list[Unit], list[Unit]]:
        """Cut at the latest natural boundary that keeps the chunk half full."""
        total, running, best = _tokens(current), 0, None
        for position, unit in enumerate(current):
            if position and unit.break_before and running >= _LOOKBACK_FILL * total:
                best = position
            running += unit.tokens
        return (current, []) if best is None else (current[:best], current[best:])

    def _overlap_tail(self, closed: list[Unit]) -> list[Unit]:
        budget, tail = self._config.overlap, []
        for unit in reversed(closed[1:]):  # never repeat the whole chunk
            if unit.tokens > budget:
                break
            budget -= unit.tokens
            tail.insert(0, unit)
        return tail

    def _split_oversize(self, unit: Unit) -> None:
        """A single block larger than a chunk: fall back to recursive splitting."""
        cfg = self._config
        piece_text = self._text[unit.start : unit.end]
        _, offsets = semchunk.chunk(
            piece_text,
            cfg.chunk_size,
            self._counter.count,
            offsets=True,
            overlap=cfg.overlap or None,
        )
        for start, end in offsets:
            self._spans.append((unit.start + start, unit.start + end))
            self._sizes.append(self._counter.count(piece_text[start:end]))

    def _merge_tiny_tail(self) -> None:
        """Glue a too-small last chunk onto its neighbour when that fits."""
        cfg = self._config
        if len(self._spans) < 2:
            return
        if (
            self._sizes[-1] < cfg.min_chunk_tokens
            and self._sizes[-2] + self._sizes[-1] <= cfg.chunk_size
        ):
            (start, _), (_, end) = self._spans[-2], self._spans[-1]
            self._spans[-2:] = [(start, end)]
            self._sizes[-2:] = [self._sizes[-2] + self._sizes[-1]]


def _tokens(units: list[Unit]) -> int:
    return sum(unit.tokens for unit in units)
