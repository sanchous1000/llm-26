"""Splitter protocol shared by all strategies."""

from typing import Protocol

Span = tuple[int, int]


class Splitter(Protocol):
    """Cuts one text segment into chunk spans (offsets relative to ``text``)."""

    def split(self, text: str) -> list[Span]: ...
