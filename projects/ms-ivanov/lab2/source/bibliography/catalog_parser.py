"""Extraction of curated metadata (priority, tags, section) from ``SOURCES.md``.

Each catalog entry looks like::

    - `key` ★★★ **[Title](url)**
      Authors — *Venue* — 2023-01-01 — `OT` `THEORY`
      One-paragraph note.
"""

import re
from dataclasses import dataclass
from pathlib import Path

_SECTION_RE = re.compile(r"^## (?P<title>.+?)\s*$")
_ENTRY_RE = re.compile(r"^- `(?P<key>[^`]+)`(?P<rest>.*)$")
_TAG_RE = re.compile(r"`([A-Z][A-Z0-9_]*)`")
_STAR = "★"


@dataclass(frozen=True)
class CatalogEntry:
    """Curated annotations of one bibliography key."""

    key: str
    priority: int | None
    tags: tuple[str, ...]
    section: str | None
    summary: str | None


def parse_catalog(path: Path) -> dict[str, CatalogEntry]:
    """Parse ``SOURCES.md`` into a ``key -> CatalogEntry`` mapping."""
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: dict[str, CatalogEntry] = {}
    section: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        section_match = _SECTION_RE.match(line)
        entry_match = _ENTRY_RE.match(line)
        index += 1
        if section_match:
            section = section_match.group("title")
        elif entry_match:
            body, index = _collect_body(lines, index)
            entry = _build_entry(entry_match, body, section)
            entries[entry.key] = entry
    return entries


def _collect_body(lines: list[str], start: int) -> tuple[list[str], int]:
    """Return the indented continuation lines of an entry and the next index."""
    index = start
    body: list[str] = []
    while index < len(lines) and lines[index].startswith("  "):
        body.append(lines[index].strip())
        index += 1
    return body, index


def _build_entry(match: re.Match, body: list[str], section: str | None) -> CatalogEntry:
    stars = match.group("rest").count(_STAR)
    meta_line = body[0] if body else ""
    summary = " ".join(body[1:]).strip() or None
    return CatalogEntry(
        key=match.group("key"),
        priority=stars or None,
        tags=tuple(_TAG_RE.findall(meta_line)),
        section=section,
        summary=summary,
    )
