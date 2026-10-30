"""Priorities and notes of extra sources from ``RAG_ADDITIONS.md``.

The file holds one markdown table::

    | Group | Priority | Key | Title | Year | Why it is needed | Open access |
    | B | ★★★ | `key` | Title | 2009 | Reason. | https://... |
"""

from pathlib import Path

from source.bibliography.catalog_parser import CatalogEntry

_MIN_COLUMNS = 7
_STAR = "★"
_GROUPS = {
    "A": "already in sources.bib, missing from the corpus",
    "B": "new foundational works",
}


def parse_additions_table(path: Path) -> dict[str, CatalogEntry]:
    """Parse the table into a ``key -> CatalogEntry`` mapping."""
    entries: dict[str, CatalogEntry] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = _parse_row(line)
        if entry is not None:
            entries[entry.key] = entry
    return entries


def _parse_row(line: str) -> CatalogEntry | None:
    if not line.startswith("|"):
        return None
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) < _MIN_COLUMNS or not cells[2].startswith("`"):
        return None
    group = cells[0]
    return CatalogEntry(
        key=cells[2].strip("`"),
        priority=cells[1].count(_STAR) or None,
        tags=(),
        section=f"RAG additions: group {group} ({_GROUPS.get(group, 'other')})",
        summary=cells[5] or None,
    )
