"""Journal / book / proceedings items: open-access PDFs found via OpenAlex
and Semantic Scholar.

Paywalled works are reported as ``unavailable``; their PDF can be placed by
hand as ``paper.pdf`` in the entry directory and is adopted on the next run.
"""

import re
from pathlib import Path
from urllib.parse import quote

from source import config
from source.bibliography.models import Source
from source.downloaders.base import DownloadContext, acquire
from source.net import FetchError
from source.net.validators import require_pdf
from source.storage.records import EntryRecord

PDF_FILENAME = "paper.pdf"
_HAL_LANDING_RE = re.compile(
    r"^(https?://[^/]*hal[^/]*/(?:hal|inria|tel|cel)-\d+)(?:v\d+)?/?$"
)


class DoiDownloader:
    """Fetches the legally available open-access PDF of a DOI, if any."""

    def __init__(self, context: DownloadContext):
        self._context = context

    def download(
        self, source: Source, record: EntryRecord, entry_dir: Path
    ) -> EntryRecord:
        if not source.doi:
            return record.with_missing("pdf", "no DOI")
        return acquire(
            self._context,
            entry_dir,
            record,
            name="pdf",
            filename=PDF_FILENAME,
            urls=lambda: self._open_access_pdf_urls(source.doi),
            validator=require_pdf,
        )

    def _open_access_pdf_urls(self, doi: str) -> list[str]:
        urls: list[str] = []
        errors: list[FetchError] = []
        for lookup in (self._openalex_urls, self._semantic_scholar_urls):
            try:
                urls.extend(lookup(doi))
            except FetchError as error:
                errors.append(error)
        candidates = unique_pdf_urls(urls)
        if candidates:
            return candidates
        raise FetchError(
            "no open-access PDF found" + "".join(f" ({error})" for error in errors),
            permanent=all(error.permanent for error in errors),
        )

    def _openalex_urls(self, doi: str) -> list[str]:
        url = config.OPENALEX_WORK_URL.format(doi=quote(doi, safe="/"))
        work = self._get_json_or_none(url)
        return pdf_candidates(work) if work else []

    def _semantic_scholar_urls(self, doi: str) -> list[str]:
        url = config.SEMANTIC_SCHOLAR_URL.format(doi=quote(doi, safe="/"))
        paper = self._get_json_or_none(url)
        return semantic_scholar_candidates(paper) if paper else []

    def _get_json_or_none(self, url: str) -> dict | None:
        """GET JSON; an unknown DOI (404) simply yields no candidates."""
        try:
            return self._context.client.get_json(url)
        except FetchError as error:
            if error.status == 404:
                return None
            raise


def pdf_candidates(work: dict) -> list[str]:
    """Open-access PDF URLs of an OpenAlex work, best location first."""
    locations = [work.get("best_oa_location"), *(work.get("locations") or [])]
    return [
        location["pdf_url"]
        for location in locations
        if location and location.get("is_oa") and location.get("pdf_url")
    ]


def semantic_scholar_candidates(paper: dict) -> list[str]:
    """Open-access PDF URLs of a Semantic Scholar paper (incl. its arXiv copy)."""
    urls = []
    open_access = (paper.get("openAccessPdf") or {}).get("url")
    if open_access:
        urls.append(open_access)
    arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
    if arxiv_id:
        urls.extend(
            template.format(arxiv_id=arxiv_id) for template in config.ARXIV_PDF_URLS
        )
    return urls


def unique_pdf_urls(urls: list[str]) -> list[str]:
    """Normalise HAL landing pages to their PDF, drop duplicates, cap the list."""
    normalised = (_HAL_LANDING_RE.sub(r"\1/document", url) for url in urls)
    return list(dict.fromkeys(normalised))[: config.MAX_DOI_PDF_CANDIDATES]
