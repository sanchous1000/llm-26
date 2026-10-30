"""LaTeXML tables -> clean rectangular HTML tables (pipe tables in Markdown)."""

import copy

from bs4 import Tag

from source.parsing.latexml.dom import alive, collapse, new_tag, new_text

_BLOCK_TAGS = ("p", "div", "ul", "ol", "li", "figure", "section")
_NESTED_CELL_CLASSES = "span.ltx_tabular, span.ltx_tr, span.ltx_td"


def convert_tables(root: Tag) -> None:
    """Rebuild outermost tables; nested tabulars collapse into cell text."""
    for table in alive(root.find_all("table")):
        if table.parent is None or table.find_parent("table") is not None:
            continue
        grid = _grid(table)
        if not grid:
            table.decompose()
        elif len(grid) == 1 and len(grid[0]) == 1:
            table.replace_with(_cell_content(grid[0][0]))
        else:
            table.replace_with(_build_table(grid))


def _grid(table: Tag) -> list[list[Tag | None]]:
    """Expand colspan/rowspan so that every row has the same width.

    Slots covered by a span of another cell are ``None`` (rendered empty).
    """
    pending: dict[int, int] = {}  # column -> rows still covered by a rowspan
    grid: list[list[Tag | None]] = []
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        line: list[Tag | None] = []
        for cell in row.find_all(["td", "th"], recursive=False):
            _skip_covered(line, pending)
            span = _span(cell.get("colspan"))
            for offset in range(span):
                pending_rows = _span(cell.get("rowspan")) - 1
                if pending_rows:
                    pending[len(line) + offset] = pending_rows
            line.append(cell)
            line.extend([None] * (span - 1))
        _skip_covered(line, pending)
        if any(cell is not None for cell in line):
            grid.append(line)
    width = max((len(line) for line in grid), default=0)
    return [line + [None] * (width - len(line)) for line in grid]


def _skip_covered(line: list, pending: dict[int, int]) -> None:
    while pending.get(len(line), 0) > 0:
        pending[len(line)] -= 1
        line.append(None)


def _span(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


def _build_table(grid: list[list[Tag | None]]) -> Tag:
    table = new_tag("table")
    head, body = new_tag("thead"), new_tag("tbody")
    for index, line in enumerate(grid):
        row = new_tag("tr")
        for cell in line:
            out = new_tag("th" if index == 0 else "td")
            if cell is not None:
                out.append(_cell_content(cell))
            row.append(out)
        (head if index == 0 else body).append(row)
    table.append(head)
    if body.contents:
        table.append(body)
    return table


def _cell_content(cell: Tag) -> Tag:
    """Inline-only copy of a cell: blocks and nested tables are flattened."""
    holder = new_tag("span")
    for node in cell.contents:
        holder.append(copy.copy(node))
    for nested in alive(holder.find_all("table")):
        nested.replace_with(new_text(collapse(nested.get_text(" "))))
    for block in alive(holder.find_all(_BLOCK_TAGS)):
        block.insert_after(new_text(" "))
        block.unwrap()
    for nested in alive(holder.select(_NESTED_CELL_CLASSES)):
        nested.insert_after(new_text(" "))
        nested.unwrap()
    for line_break in alive(holder.find_all("br")):
        line_break.replace_with(new_text(" "))
    return holder
