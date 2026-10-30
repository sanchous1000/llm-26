"""Text of a converted document: body without front matter, bibliography span."""

import re
from pathlib import Path

_FRONT_MATTER_END = "\n---\n"
_REFERENCES_RE = re.compile(
    r"^## (?:References|Bibliography|REFERENCES)\s*$", re.MULTILINE
)
_SECTION_RE = re.compile(r"^## ", re.MULTILINE)


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
    following = _SECTION_RE.search(body, start.end())
    return start.start(), following.start() if following else len(body)
