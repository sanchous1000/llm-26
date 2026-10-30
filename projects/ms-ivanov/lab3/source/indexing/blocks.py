"""Markdown body -> typed blocks with character spans.

The converted documents follow a small, predictable dialect (ATX headings,
``$$`` display math, fenced code for algorithms, pipe tables, ``![..](..)``
figures, ``<!-- page: N -->`` markers), so a line scanner is enough and keeps
the spans exact.
"""

import re
from dataclasses import dataclass

HEADING = "heading"
PAGE = "page"
CODE = "code"
MATH = "math"
TABLE = "table"
LIST = "list"
IMAGE = "image"
QUOTE = "quote"
PARAGRAPH = "paragraph"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PAGE_RE = re.compile(r"^<!-- page: (\d+) -->$")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$")


@dataclass(frozen=True)
class Block:
    """A contiguous run of lines of one kind; ``level``/``page`` when relevant."""

    kind: str
    start: int
    end: int
    level: int = 0  # heading level
    title: str = ""  # heading text
    page: int = 0  # page marker number


def parse_blocks(body: str) -> list[Block]:
    """Split ``body`` into blocks; spans exclude the surrounding blank lines."""
    lines = body.splitlines(keepends=True)
    starts = _line_starts(lines)
    blocks: list[Block] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        kind, last, extra = _scan(lines, index)
        end = starts[last] + len(lines[last].rstrip("\r\n"))
        blocks.append(Block(kind, starts[index], end, **extra))
        index = last + 1
    return blocks


def _line_starts(lines: list[str]) -> list[int]:
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    return starts


def _scan(lines: list[str], index: int) -> tuple[str, int, dict]:  # noqa: PLR0911
    """Classify the block starting at ``index``; return (kind, last line, extras)."""
    text = lines[index].strip()
    if (fence := _FENCE_RE.match(text)) is not None:
        return CODE, _until_fence_end(lines, index, fence.group(1)), {}
    if text.startswith("$$"):
        return MATH, _until_math_end(lines, index), {}
    if (heading := _HEADING_RE.match(text)) is not None:
        return (
            HEADING,
            index,
            {"level": len(heading.group(1)), "title": heading.group(2)},
        )
    if (page := _PAGE_RE.match(text)) is not None:
        return PAGE, index, {"page": int(page.group(1))}
    if text.startswith("|"):
        return TABLE, _while(lines, index, lambda s: s.strip().startswith("|")), {}
    if _IMAGE_RE.match(text):
        return IMAGE, index, {}
    if _LIST_RE.match(lines[index]):
        return LIST, _list_end(lines, index), {}
    if text.startswith(">"):
        return QUOTE, _while(lines, index, lambda s: s.strip().startswith(">")), {}
    return PARAGRAPH, _paragraph_end(lines, index), {}


def _until_fence_end(lines: list[str], index: int, fence: str) -> int:
    for position in range(index + 1, len(lines)):
        if lines[position].strip().startswith(fence):
            return position
    return len(lines) - 1


def _until_math_end(lines: list[str], index: int) -> int:
    first = lines[index].strip()
    if len(first) > 2 and first.endswith("$$"):  # one-line "$$ ... $$"
        return index
    for position in range(index + 1, len(lines)):
        if lines[position].strip().endswith("$$"):
            return position
    return len(lines) - 1


def _while(lines: list[str], index: int, predicate) -> int:
    last = index
    while last + 1 < len(lines) and predicate(lines[last + 1]):
        last += 1
    return last


def _paragraph_end(lines: list[str], index: int) -> int:
    return _while(
        lines, index, lambda line: bool(line.strip()) and not _starts_block(line)
    )


def _starts_block(line: str) -> bool:
    text = line.strip()
    return bool(
        _FENCE_RE.match(text)
        or text.startswith("$$")
        or _HEADING_RE.match(text)
        or _PAGE_RE.match(text)
    )


def _list_end(lines: list[str], index: int) -> int:
    """A list continues over indented lines and blank lines before more items."""
    last = index
    position = index + 1
    while position < len(lines):
        line = lines[position]
        if line.strip():
            if not (_LIST_RE.match(line) or line.startswith((" ", "\t"))):
                break
            last = position
        elif not _blank_gap_continues(lines, position):
            break
        position += 1
    return last


def _blank_gap_continues(lines: list[str], position: int) -> bool:
    for following in lines[position + 1 :]:
        if following.strip():
            return bool(_LIST_RE.match(following) or following.startswith((" ", "\t")))
    return False
