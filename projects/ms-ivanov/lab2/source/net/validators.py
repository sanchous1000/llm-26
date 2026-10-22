"""Content validators: reject error pages that were served with HTTP 200."""

from collections.abc import Callable
from pathlib import Path

from source.net.errors import FetchError

Validator = Callable[[Path, str], None]

_PDF_MAGIC = b"%PDF"
_HTML_SNIFF_BYTES = 200_000
_LATEXML_PARAGRAPH = b'class="ltx_p"'
_MIN_PARAGRAPHS = 10


def _invalid(message: str) -> FetchError:
    return FetchError(message, permanent=True)


def require_pdf(path: Path, content_type: str) -> None:
    """Accept only real PDF documents."""
    with path.open("rb") as handle:
        head = handle.read(len(_PDF_MAGIC))
    if head != _PDF_MAGIC:
        raise _invalid(f"not a PDF (content-type {content_type!r})")


def require_html(path: Path, content_type: str) -> None:
    """Accept only HTML documents."""
    head = path.read_bytes()[:_HTML_SNIFF_BYTES].lower()
    if b"<html" not in head and b"<!doctype html" not in head:
        raise _invalid(f"not an HTML page (content-type {content_type!r})")


def require_latexml_html(path: Path, content_type: str) -> None:
    """Accept only LaTeXML-rendered papers with real body text.

    ar5iv answers 200 with a stub ("No content available", "Fatal error")
    for papers it failed to render, so the paragraph count is checked too.
    """
    require_html(path, content_type)
    body = path.read_bytes()
    if body.count(_LATEXML_PARAGRAPH) < _MIN_PARAGRAPHS:
        raise _invalid("HTML page has no rendered paper content")


def require_text(path: Path, content_type: str) -> None:
    """Accept any non-empty textual body."""
    if path.stat().st_size == 0:
        raise _invalid("empty body")
