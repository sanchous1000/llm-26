"""Inline formatting: fonts, cross references, citations, footnotes."""

from urllib.parse import urljoin

from bs4 import Tag

from source.parsing.latexml.dom import (
    alive,
    collapse,
    has_class,
    new_tag,
    new_text,
    text_of,
)

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def strip_theorem_italics(root: Tag) -> None:
    """Theorem bodies are italic by style, not by emphasis: drop that."""
    for theorem in root.select("div.ltx_theorem"):
        for span in alive(theorem.select("span.ltx_font_italic")):
            span.unwrap()


_RESETS = (
    ("ltx_font_bold", "ltx_font_medium"),
    ("ltx_font_italic", "ltx_font_upright"),
)


def _split_on_resets(root: Tag) -> None:
    """``bold( a medium(b) c )`` -> ``bold(a) b bold(c)`` (same for italics).

    LaTeXML expresses "normal weight inside bold" as a nested reset span,
    which would otherwise be flattened into one wrong ``**...**`` run.
    """
    for style, reset in _RESETS:
        for span in alive(root.select(f"span.{style}")):
            if span.find("span", class_=reset, recursive=False) is None:
                continue
            pieces: list = []
            run: list = []
            for child in list(span.contents):
                if isinstance(child, Tag) and has_class(child, reset):
                    _flush_run(span, style, run, pieces)
                    run = []
                    pieces.append(child.extract())
                else:
                    run.append(child.extract())
            _flush_run(span, style, run, pieces)
            span.replace_with(*pieces)


def _flush_run(span: Tag, style: str, run: list, pieces: list) -> None:
    if not run:
        return
    part = new_tag("span", **{"class": style})
    for node in run:
        part.append(node)
    pieces.append(part)


def convert_fonts(root: Tag) -> None:
    """Turn LaTeXML font classes into semantic strong/em/code tags."""
    _split_on_resets(root)
    for span in alive(root.find_all("span")):
        in_heading = span.find_parent(_HEADING_TAGS) is not None
        bold = has_class(span, "ltx_font_bold") and not in_heading
        italic = (
            has_class(span, "ltx_font_italic") or has_class(span, "ltx_font_oblique")
        ) and not in_heading
        code = has_class(span, "ltx_font_typewriter") and not in_heading
        if code:
            span.name = "code"
        elif bold and italic:
            _wrap_children(span, "em")
            span.name = "strong"
        elif bold:
            span.name = "strong"
        elif italic:
            span.name = "em"
    for element in alive(root.find_all(["strong", "em", "code"])):
        if not element.get_text().strip() and element.find("img") is None:
            element.unwrap()


def _wrap_children(element: Tag, name: str) -> None:
    wrapper = new_tag(name)
    for child in list(element.contents):
        wrapper.append(child.extract())
    element.append(wrapper)


def resolve_links(root: Tag, base_url: str) -> None:
    """Keep external links, flatten citations and internal cross references."""
    for cite in alive(root.find_all("cite")):
        cite.unwrap()
    for link in alive(root.find_all("a")):
        href = (link.get("href") or "").strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            link.unwrap()
        else:
            link["href"] = urljoin(base_url, href)
    for element in alive(root.select("span.ltx_ref, span.ltx_glossaryref")):
        element.unwrap()


def inline_footnotes(root: Tag) -> None:
    """Footnotes become ``[Footnote: ...]`` right where they are referenced."""
    for note in alive(root.select("span.ltx_note")):
        content = note.select_one("span.ltx_note_content")
        if content is None:
            note.decompose()
            continue
        for block in content.find_all(["p", "div"]):
            block.insert_after(new_text(" "))
            block.unwrap()
        text = collapse(content.get_text(" "))
        replacement = new_tag("span")
        replacement.append(new_text(f" [Footnote: {text}]" if text else ""))
        note.replace_with(replacement)


def replace_empty_titles(root: Tag) -> None:
    """Remove headings without any text (LaTeXML emits some for unnumbered parts)."""
    for heading in alive(root.find_all(_HEADING_TAGS)):
        if not text_of(heading):
            heading.decompose()
