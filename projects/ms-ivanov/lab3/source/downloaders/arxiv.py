"""arXiv papers: HTML first (ar5iv, then arxiv.org), PDF as fallback."""

from pathlib import Path

from source import config
from source.bibliography.models import Source
from source.downloaders.base import DownloadContext, PdfPolicy, acquire
from source.net.validators import require_latexml_html, require_pdf
from source.storage.records import EntryRecord

HTML_FILENAME = "paper.html"
PDF_FILENAME = "paper.pdf"


class ArxivDownloader:
    """Downloads the rendered HTML of a paper, and its PDF when needed."""

    def __init__(self, context: DownloadContext):
        self._context = context

    def download(
        self, source: Source, record: EntryRecord, entry_dir: Path
    ) -> EntryRecord:
        if not source.arxiv_id:
            return record.with_missing("html", "no arXiv id")
        record = acquire(
            self._context,
            entry_dir,
            record,
            name="html",
            filename=HTML_FILENAME,
            urls=_format_urls(config.ARXIV_HTML_URLS, source.arxiv_id),
            validator=require_latexml_html,
        )
        if not self._wants_pdf(record):
            return record
        return acquire(
            self._context,
            entry_dir,
            record,
            name="pdf",
            filename=PDF_FILENAME,
            urls=_format_urls(config.ARXIV_PDF_URLS, source.arxiv_id),
            validator=require_pdf,
        )

    def _wants_pdf(self, record: EntryRecord) -> bool:
        policy = self._context.pdf_policy
        if policy is PdfPolicy.ALWAYS:
            return True
        return policy is PdfPolicy.FALLBACK and "html" not in record.files


def _format_urls(templates: tuple[str, ...], arxiv_id: str) -> list[str]:
    return [template.format(arxiv_id=arxiv_id) for template in templates]
