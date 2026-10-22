"""Parsing of ``sources.bib`` into :class:`Source` objects."""

import re
from pathlib import Path
from urllib.parse import urlparse

import bibtexparser
from bibtexparser.bparser import BibTexParser

from source.bibliography.models import Source, SourceKind

_ARXIV_ABS_RE = re.compile(r"^/(?:abs|pdf|html)/(?P<id>[^?#]+?)(?:\.pdf)?$")
_ARXIV_DOI_PREFIX = "10.48550/arxiv."
_DOI_HOSTS = {"doi.org", "dx.doi.org"}


def parse_bib_file(path: Path) -> list[Source]:
    """Read a ``.bib`` file and return its entries in file order."""
    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    with path.open(encoding="utf-8") as handle:
        database = bibtexparser.load(handle, parser=parser)
    return [_to_source(entry) for entry in database.entries]


def _to_source(entry: dict) -> Source:
    url = entry.get("url", "").strip()
    arxiv_id = _extract_arxiv_id(entry, url)
    doi = _extract_doi(entry, url)
    return Source(
        key=entry["ID"],
        kind=_classify(url, arxiv_id, doi),
        title=_clean(entry.get("title", "")),
        url=url,
        authors=_split_authors(entry.get("author", "")),
        year=_parse_year(entry.get("year")),
        venue=_first_present(
            entry, ("journal", "booktitle", "howpublished", "publisher")
        ),
        arxiv_id=arxiv_id,
        doi=doi,
    )


def _classify(url: str, arxiv_id: str | None, doi: str | None) -> SourceKind:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if arxiv_id:
        return SourceKind.ARXIV
    if host == "github.com":
        return SourceKind.GITHUB
    if doi:
        return SourceKind.DOI
    return SourceKind.WEB


def _extract_arxiv_id(entry: dict, url: str) -> str | None:
    if entry.get("eprint"):
        return entry["eprint"].strip()
    parsed = urlparse(url)
    if parsed.netloc.lower().removeprefix("www.") == "arxiv.org":
        match = _ARXIV_ABS_RE.match(parsed.path)
        if match:
            return match.group("id")
    doi = entry.get("doi", "").strip().lower()
    if doi.startswith(_ARXIV_DOI_PREFIX):
        return doi.removeprefix(_ARXIV_DOI_PREFIX)
    return None


def _extract_doi(entry: dict, url: str) -> str | None:
    if entry.get("doi"):
        return entry["doi"].strip()
    parsed = urlparse(url)
    if parsed.netloc.lower() in _DOI_HOSTS:
        return parsed.path.lstrip("/") or None
    return None


def _split_authors(raw: str) -> tuple[str, ...]:
    return tuple(_clean(name) for name in raw.split(" and ") if name.strip())


def _parse_year(raw: str | None) -> int | None:
    return int(raw) if raw and raw.strip().isdigit() else None


def _first_present(entry: dict, fields: tuple[str, ...]) -> str | None:
    for field in fields:
        if entry.get(field):
            return _clean(entry[field])
    return None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
