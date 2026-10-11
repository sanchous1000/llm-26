"""Generic web resources: direct PDFs and landing pages (with their PDF)."""

from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from source.bibliography.models import Source
from source.downloaders.base import DownloadContext, acquire
from source.net.validators import require_html, require_pdf
from source.storage.records import EntryRecord

PAGE_FILENAME = "page.html"
PDF_FILENAME = "paper.pdf"


class WebDownloader:
    """Downloads a page and, if the page advertises one, the paper PDF."""

    def __init__(self, context: DownloadContext):
        self._context = context

    def download(
        self, source: Source, record: EntryRecord, entry_dir: Path
    ) -> EntryRecord:
        if urlparse(source.url).path.lower().endswith(".pdf"):
            return self._acquire_pdf(entry_dir, record, [source.url])
        record = acquire(
            self._context,
            entry_dir,
            record,
            name="page",
            filename=PAGE_FILENAME,
            urls=[source.url],
            validator=require_html,
        )
        page = entry_dir / PAGE_FILENAME
        if not page.exists():
            return record
        pdf_url = find_pdf_link(
            page.read_text(encoding="utf-8", errors="replace"), source.url
        )
        return self._acquire_pdf(entry_dir, record, [pdf_url]) if pdf_url else record

    def _acquire_pdf(
        self, entry_dir: Path, record: EntryRecord, urls: list[str]
    ) -> EntryRecord:
        return acquire(
            self._context,
            entry_dir,
            record,
            name="pdf",
            filename=PDF_FILENAME,
            urls=urls,
            validator=require_pdf,
        )


def find_pdf_link(html: str, base_url: str) -> str | None:
    """Return the absolute PDF URL declared by ``citation_pdf_url`` meta tags."""
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("meta", attrs={"name": "citation_pdf_url"})
    content = tag.get("content") if tag else None
    return urljoin(base_url, content.strip()) if content else None
