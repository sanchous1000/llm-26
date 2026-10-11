"""Document title, abstract and section heading normalisation."""

import re

from bs4 import Tag

from source.parsing.latexml.dom import (
    alive,
    collapse,
    has_class,
    new_tag,
    new_text,
    text_of,
)

_RUNIN_TARGETS = ("p", "div")
_FRONT_MATTER_SECTIONS = ("abstract", "acknowledgements", "acknowledgments")


def extract_title(article: Tag) -> str:
    """Pop the document title (inner HTML, may contain math tokens)."""
    heading = article.find("h1", class_="ltx_title") or article.find("h1")
    if heading is None:
        return ""
    for note in heading.select("span.ltx_note, span.ltx_pubnotes"):
        note.decompose()
    title = collapse(heading.get_text())
    heading.decompose()
    return title


def promote_front_matter_titles(root: Tag) -> None:
    """Abstract / acknowledgements get proper level-2 headings."""
    for name in _FRONT_MATTER_SECTIONS:
        for block in alive(root.select(f"div.ltx_{name}")):
            title = block.find(["h6", "h5", "h4"], class_=f"ltx_title_{name}")
            label = text_of(title) if title else name.capitalize()
            if title:
                title.decompose()
            heading = new_tag("h2")
            heading.append(new_text(label))
            block.insert(0, heading)
    for keywords in alive(root.select("div.ltx_keywords")):
        label = new_tag("strong")
        label.append(new_text("Keywords:"))
        keywords.insert(0, new_text(" "))
        keywords.insert(0, label)


def merge_runin_titles(root: Tag) -> None:
    """``Proof.``/``Lemma 1.``/paragraph titles become a bold lead-in."""
    for heading in alive(root.find_all(class_="ltx_runin")):
        if not re.fullmatch(r"h[1-6]", heading.name):
            continue
        lead = _bold_lead(heading)
        target = _following_paragraph(heading)
        heading.decompose()
        if lead is None:
            continue
        if target is None:
            target = new_tag("p")
            heading_parent = heading.parent
            if heading_parent is not None:
                heading_parent.append(target)
        target.insert(0, lead)


def _bold_lead(heading: Tag) -> Tag | None:
    for inner in heading.find_all(["span", "strong", "em", "b", "i"]):
        inner.unwrap()
    if not heading.get_text().strip():
        return None
    lead = new_tag("strong")
    for node in list(heading.contents):
        lead.append(node.extract())
    lead.append(new_text(" "))
    return lead


def _following_paragraph(heading: Tag) -> Tag | None:
    sibling = heading.find_next_sibling(True)
    if sibling is None:
        return None
    if sibling.name == "p":
        return sibling
    if sibling.name in _RUNIN_TARGETS and has_class(sibling, "ltx_para"):
        return sibling.find("p")
    return None


def number_headings(root: Tag) -> None:
    """Make sure the section number is separated from the title text."""
    for tag in alive(
        root.select("h2 > span.ltx_tag, h3 > span.ltx_tag, h4 > span.ltx_tag")
    ):
        tag.insert_after(new_text(" "))
