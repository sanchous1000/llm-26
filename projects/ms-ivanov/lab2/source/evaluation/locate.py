"""Locate a quoted passage inside a document (exact first, then fuzzy)."""

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 92.0
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Located:
    """Where the quote was found; ``text`` is the document's own wording."""

    start: int
    end: int
    text: str
    score: float


def _normalise(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace; return the new text and, per char, its original offset."""
    chars: list[str] = []
    index: list[int] = []
    previous_space = True
    for offset, char in enumerate(text):
        if char.isspace():
            if not previous_space:
                chars.append(" ")
                index.append(offset)
            previous_space = True
        else:
            chars.append(char)
            index.append(offset)
            previous_space = False
    return "".join(chars).strip(), index


def locate(body: str, quote: str) -> Located | None:
    """Find ``quote`` in ``body`` ignoring whitespace differences."""
    needle, _ = _normalise(quote)
    if len(needle) < 40:
        return None
    haystack, index = _normalise(body)
    position = haystack.find(needle)
    if position != -1:
        return _result(body, index, position, position + len(needle), 100.0)
    alignment = fuzz.partial_ratio_alignment(
        needle, haystack, score_cutoff=FUZZY_THRESHOLD
    )
    if alignment is None:
        return None
    return _result(
        body, index, alignment.dest_start, alignment.dest_end, alignment.score
    )


def _result(body: str, index: list[int], start: int, end: int, score: float) -> Located:
    original_start = index[start]
    original_end = index[end - 1] + 1
    return Located(
        original_start, original_end, body[original_start:original_end], score
    )
