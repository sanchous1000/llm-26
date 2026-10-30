"""Small helpers for manipulating a BeautifulSoup tree.

The converter works on a private copy of the document and rewrites it step
by step, so in-place mutation of the tree is intentional here.
"""

import re
from collections.abc import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

_FACTORY = BeautifulSoup("", "html.parser")
_WHITESPACE_RE = re.compile(r"\s+")


def new_tag(name: str, **attrs: str) -> Tag:
    return _FACTORY.new_tag(name, attrs=attrs or None)


def new_text(text: str) -> NavigableString:
    return NavigableString(text)


def has_class(element: Tag, name: str) -> bool:
    return name in (element.get("class") or ())


def collapse(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def text_of(element: Tag) -> str:
    return collapse(element.get_text(" "))


def alive(elements: Iterable[Tag]) -> list[Tag]:
    """Elements that are still attached to the tree."""
    return [element for element in elements if element.parent is not None]
