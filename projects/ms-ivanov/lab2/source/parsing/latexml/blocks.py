"""Lists, code listings and verbatim blocks."""

import re

from bs4 import Tag

from source.parsing.latexml.context import Context
from source.parsing.latexml.dom import (
    alive,
    collapse,
    new_tag,
    new_text,
    text_of,
)
from source.parsing.latexml.math import flatten_math_for_listing

_ARABIC_LABEL_RE = re.compile(r"^\(?\d+[.)]?$")


def convert_listings(root: Tag, context: Context) -> None:
    """Algorithms and code listings become fenced code blocks."""
    for listing in alive(root.select("div.ltx_listing")):
        flatten_math_for_listing(listing)
        lines = [_listing_line(line) for line in listing.select("div.ltx_listingline")]
        if not lines:
            lines = listing.get_text().splitlines()
        code = "\n".join(line.rstrip() for line in lines).strip("\n")
        listing.replace_with(_code_paragraph(code, context))
    for block in alive(root.select("pre.ltx_verbatim, div.ltx_verbatim")):
        flatten_math_for_listing(block)
        block.replace_with(_code_paragraph(block.get_text().strip("\n"), context))
    for inline in alive(root.select("span.ltx_verbatim")):
        inline.name = "code"


def _listing_line(line: Tag) -> str:
    return line.get_text().replace(" ", " ").replace("\n", " ")


def _code_paragraph(code: str, context: Context) -> Tag:
    paragraph = new_tag("p")
    paragraph.append(new_text(context.store.code_block(code)))
    return paragraph


def convert_lists(root: Tag) -> None:
    """Normalise itemize/enumerate/description lists (innermost first)."""
    for dl in reversed(alive(root.select("dl.ltx_description"))):
        _description_to_list(dl)
    for lst in reversed(alive(root.select("ul, ol"))):
        _tidy_list(lst)


def _description_to_list(dl: Tag) -> None:
    new_list = new_tag("ul")
    term = None
    for child in dl.find_all(["dt", "dd"], recursive=False):
        if child.name == "dt":
            term = child
            continue
        item = new_tag("li")
        if term is not None:
            label = collapse(term.get_text(" "))
            if label:
                strong = new_tag("strong")
                strong.append(new_text(f"{label}:"))
                item.append(strong)
                item.append(new_text(" "))
        for node in list(child.contents):
            item.append(node.extract())
        new_list.append(item)
        term = None
    dl.replace_with(new_list)


def _tidy_list(lst: Tag) -> None:
    items = lst.find_all("li", recursive=False)
    labels = [_pop_label(item) for item in items]
    if lst.name == "ol" and not all(
        _ARABIC_LABEL_RE.match(label) for label in labels if label
    ):
        lst.name = "ul"
        for item, label in zip(items, labels, strict=True):
            if label:
                item.insert(0, new_text(f"{label} "))
    for item in items:
        _flatten_item(item)


def _pop_label(item: Tag) -> str:
    tag = item.find("span", class_="ltx_tag_item", recursive=False)
    if tag is None:
        return ""
    label = text_of(tag)
    tag.decompose()
    return label


def _flatten_item(item: Tag) -> None:
    for para in item.find_all("div", class_="ltx_para", recursive=False):
        para.unwrap()
    paragraphs = item.find_all("p", recursive=False)
    if len(paragraphs) == 1:
        paragraphs[0].unwrap()
