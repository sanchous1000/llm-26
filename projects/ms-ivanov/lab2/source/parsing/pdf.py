"""PDF -> Markdown through PyMuPDF4LLM, keeping page boundaries."""

import re
from pathlib import Path

import pymupdf
import pymupdf4llm

from source.parsing import postprocess
from source.parsing.errors import ConversionError
from source.parsing.models import ConversionResult

PAGE_MARKER = "<!-- page: {number} -->"
_BOLD_HEADING_RE = re.compile(r"^(#{1,6}) \*\*(.+?)\*\*\s*$", re.MULTILINE)
_MIN_TEXT_CHARS_PER_PAGE = 200


class PdfConverter:
    """Extracts structured text; formulas inside PDFs are not recoverable."""

    name = "pymupdf4llm@1"

    def convert(self, path: Path, base_url: str) -> ConversionResult:
        try:
            with pymupdf.open(path) as document:
                chunks = pymupdf4llm.to_markdown(
                    document,
                    page_chunks=True,
                    header=False,
                    footer=False,
                    ignore_images=True,
                    show_progress=False,
                )
        except (RuntimeError, ValueError, OSError) as error:
            raise ConversionError(f"cannot read PDF: {error}") from error
        pages = [
            f"{PAGE_MARKER.format(number=index)}\n\n{_bold_headings(chunk['text']).strip()}"
            for index, chunk in enumerate(chunks, start=1)
        ]
        markdown = postprocess.tidy("\n\n".join(pages))
        return ConversionResult(markdown=markdown, warnings=_warnings(chunks))


def _bold_headings(text: str) -> str:
    return _BOLD_HEADING_RE.sub(r"\1 \2", text)


def _warnings(chunks: list[dict]) -> tuple[str, ...]:
    empty = sum(
        1 for chunk in chunks if len(chunk["text"].strip()) < _MIN_TEXT_CHARS_PER_PAGE
    )
    if chunks and empty > len(chunks) / 2:
        return (f"{empty} of {len(chunks)} pages have almost no text (scanned PDF?)",)
    return ()
