"""Access to converted documents: body text, references stripping, splitting."""

import re
from dataclasses import dataclass
from pathlib import Path

_FRONT_MATTER_END = "\n---\n"
_HEADING_RE = re.compile(r"^(#{1,6}) (.+)$", re.MULTILINE)
_PAGE_RE = re.compile(r"^<!-- page: (\d+) -->$", re.MULTILINE)
_REFERENCES_RE = re.compile(
    r"^## (?:References|Bibliography|REFERENCES)\s*$", re.MULTILINE
)


@dataclass(frozen=True)
class Part:
    """A slice of the document text (without bibliography) sent in one request."""

    index: int
    total: int
    text: str


def read_body(path: Path) -> str:
    """Document text without the YAML front matter (offsets refer to this text)."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        end = text.find(_FRONT_MATTER_END, 4)
        if end != -1:
            return text[end + len(_FRONT_MATTER_END) :].lstrip("\n")
    return text


def references_span(body: str) -> tuple[int, int] | None:
    """Character span of the bibliography section, if the document has one."""
    start = _REFERENCES_RE.search(body)
    if start is None:
        return None
    following = re.compile(r"^## ", re.MULTILINE).search(body, start.end())
    return start.start(), following.start() if following else len(body)


def strip_references(body: str) -> str:
    """Body without the bibliography section."""
    span = references_span(body)
    return body if span is None else body[: span[0]] + body[span[1] :]


def split_into_parts(text: str, max_chars: int) -> list[Part]:
    """Split ``text`` into parts of <= ``max_chars``, cutting at ``##`` headings.

    A section longer than the limit is cut at paragraph breaks.
    """
    sections = _sections(text, max_chars)
    packed = [sections[0]]
    for left, right in sections[1:]:
        if right - packed[-1][0] <= max_chars:
            packed[-1] = (packed[-1][0], right)
        else:
            packed.append((left, right))
    return [
        Part(index + 1, len(packed), text[start:end])
        for index, (start, end) in enumerate(packed)
    ]


def _sections(text: str, max_chars: int) -> list[tuple[int, int]]:
    cuts = [m.start() for m in re.finditer(r"^## ", text, re.MULTILINE)]
    bounds = sorted({0, *cuts, len(text)})
    sections: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:], strict=False):
        cursor = left
        while right - cursor > max_chars:
            cut = text.rfind("\n\n", cursor, cursor + max_chars)
            cut = cut if cut > cursor else cursor + max_chars
            sections.append((cursor, cut))
            cursor = cut
        sections.append((cursor, right))
    return sections


def heading_path(body: str, offset: int) -> list[str]:
    """Headings (h1..h3) that enclose the character at ``offset``."""
    stack: dict[int, str] = {}
    for match in _HEADING_RE.finditer(body, 0, offset):
        level = len(match.group(1))
        stack = {k: v for k, v in stack.items() if k < level}
        stack[level] = match.group(2).strip()
    return [stack[level] for level in sorted(stack) if level <= 3]


def page_at(body: str, offset: int) -> int | None:
    """PDF page number (from page markers) containing ``offset``."""
    pages = [int(m.group(1)) for m in _PAGE_RE.finditer(body, 0, offset)]
    return pages[-1] if pages else None
