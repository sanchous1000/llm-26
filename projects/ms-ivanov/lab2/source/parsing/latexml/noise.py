"""Removal of page chrome and LaTeXML bookkeeping elements."""

from bs4 import Tag

from source.parsing.latexml.dom import alive

_NOISE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "button",
    "nav",
    "svg",
    "div.ltx_pagination",
    "div.ltx_authors",
    "div.ltx_dates",
    "span.ltx_ERROR",
    "span.ltx_rdf",
    "span.ltx_pubnotes",
    "a.ltx_anchor",
    "span.ltx_bib_cited",
    "sup.ltx_note_mark",
    "span.ltx_tag_note",
    "[style*='display:none']",
    "[style*='display: none']",
)


def remove_noise(root: Tag) -> None:
    """Drop elements that carry no document content."""
    for selector in _NOISE_SELECTORS:
        for element in alive(root.select(selector)):
            element.decompose()
