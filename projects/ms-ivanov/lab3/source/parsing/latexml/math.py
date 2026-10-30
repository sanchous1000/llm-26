"""Math and equation handling: MathML/LaTeXML -> protected TeX tokens."""

import re

from bs4 import Tag

from source.parsing.latexml.context import Context
from source.parsing.latexml.dom import (
    alive,
    collapse,
    has_class,
    new_tag,
    new_text,
    text_of,
)
from source.parsing.placeholders import PlaceholderStore

_EQNO_RE = re.compile(r"^\((.+)\)$")
_TEX_WHITESPACE_RE = re.compile(r"\s+")
_TEX_LAYOUT_RE = re.compile(
    r"\\[vh]skip\s*-?[\d.]+\s*(?:pt|em|ex|mm|cm)?|\\(?:vspace|hspace)\*?\{[^}]*\}"
)


def tex_of(math: Tag) -> str:
    """TeX source of a ``<math>`` element (``alttext`` or the annotation)."""
    tex = math.get("alttext")
    if not tex:
        annotation = math.find("annotation", attrs={"encoding": "application/x-tex"})
        tex = annotation.get_text() if annotation else math.get_text()
    tex = _TEX_LAYOUT_RE.sub("", tex)
    return _TEX_WHITESPACE_RE.sub(" ", tex).strip()


def convert_equation_tables(root: Tag, context: Context) -> None:
    """Replace numbered/aligned equation tables by display-math paragraphs."""
    for table in alive(root.select("table.ltx_eqn_table")):
        if table.parent is None or table.find_parent("table", class_="ltx_eqn_table"):
            continue
        rows = [
            row
            for row in table.find_all("tr", class_="ltx_eqn_row")
            if row.find_parent("table", class_="ltx_eqn_table") is table
        ]
        container = new_tag("div")
        for row in rows:
            tex = _row_tex(row)
            if tex:
                paragraph = new_tag("p")
                paragraph.append(new_text(context.store.display_math(tex)))
                container.append(paragraph)
        table.replace_with(container)


def _row_tex(row: Tag) -> str:
    parts: list[str] = []
    number = ""
    for cell in row.find_all("td", recursive=False):
        if has_class(cell, "ltx_eqn_eqno"):
            number = _equation_number(cell)
            continue
        maths = [tex_of(math) for math in cell.find_all("math")]
        if maths:
            parts.append(" ".join(maths))
        elif text_of(cell):
            parts.append(f"\\text{{{text_of(cell)}}}")
    tex = " ".join(part for part in parts if part)
    return f"{tex} \\tag{{{number}}}" if tex and number else tex


def _equation_number(cell: Tag) -> str:
    match = _EQNO_RE.match(collapse(cell.get_text()))
    return match.group(1) if match else ""


def convert_math(root: Tag, context: Context) -> None:
    """Replace every remaining ``<math>`` by a token."""
    for math in alive(root.find_all("math")):
        tex = tex_of(math)
        if not tex:
            math.decompose()
        elif math.get("display") == "block":
            math.replace_with(_display_block(tex, context.store))
        else:
            in_table = math.find_parent(["td", "th"]) is not None
            math.replace_with(new_text(context.store.inline_math(tex, in_table)))


def _display_block(tex: str, store: PlaceholderStore) -> Tag:
    paragraph = new_tag("p")
    paragraph.append(new_text(store.display_math(tex)))
    return paragraph


def flatten_math_for_listing(root: Tag) -> None:
    """Inside code listings math stays readable: ``$tex$`` literal text."""
    for math in alive(root.find_all("math")):
        math.replace_with(new_text(f"${tex_of(math)}$"))
