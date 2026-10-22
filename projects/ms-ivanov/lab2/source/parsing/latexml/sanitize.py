"""Last pass before Pandoc: keep only semantic tags and the few attributes we need."""

from bs4 import Tag

from source.parsing.latexml.dom import alive

_KEEP = {
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "li", "strong", "em", "code",
    "pre", "a", "img", "table", "thead", "tbody", "tr", "th", "td", "sup", "sub", "br",
    "blockquote", "hr",
}  # fmt: skip
_ATTRIBUTES = {"a": ("href",), "img": ("src", "alt"), "ol": ("start",)}


def sanitize(root: Tag) -> None:
    """Unwrap unknown tags (span, div, section, ...) and strip attributes."""
    for element in alive(root.find_all(True)):
        if element.name not in _KEEP:
            element.unwrap()
    for element in alive(root.find_all(True)):
        allowed = _ATTRIBUTES.get(element.name, ())
        element.attrs = {
            key: value for key, value in element.attrs.items() if key in allowed
        }
    for element in alive(root.find_all("table")):
        for stray in element.find_all(["strong", "em"]):
            if not stray.get_text().strip():
                stray.unwrap()
