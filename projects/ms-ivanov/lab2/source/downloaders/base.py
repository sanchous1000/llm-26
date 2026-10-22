"""Shared building blocks of all downloaders."""

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from source.bibliography.models import Source
from source.net import FetchError, HttpClient
from source.net.validators import Validator
from source.storage.records import MANUAL_URL, EntryRecord, FileRecord

logger = logging.getLogger(__name__)

UrlSource = Sequence[str] | Callable[[], Sequence[str]]


class PdfPolicy(StrEnum):
    """When to download the PDF of an arXiv paper next to its HTML."""

    NEVER = "never"
    FALLBACK = "fallback"  # only when no HTML version exists
    ALWAYS = "always"


@dataclass(frozen=True)
class DownloadContext:
    """Settings and services shared by all downloaders of one run."""

    client: HttpClient
    force: bool = False
    retry_missing: bool = False
    pdf_policy: PdfPolicy = PdfPolicy.FALLBACK


class Downloader(Protocol):
    """Fetches everything that is wanted for one source."""

    def download(
        self, source: Source, record: EntryRecord, entry_dir: Path
    ) -> EntryRecord:
        """Return ``record`` updated with the outcome of this attempt."""


def acquire(
    context: DownloadContext,
    entry_dir: Path,
    record: EntryRecord,
    *,
    name: str,
    filename: str,
    urls: UrlSource,
    validator: Validator,
    headers: Mapping[str, str] | None = None,
) -> EntryRecord:
    """Make sure the artifact ``name`` exists in ``entry_dir``.

    Existing files are kept (so reruns are cheap and files dropped in by hand
    are adopted); permanent absences are remembered; the first URL that
    yields a valid body wins.
    """
    target = entry_dir / filename
    if not context.force:
        if target.exists():
            return _adopt(record, name, target)
        if name in record.missing and not context.retry_missing:
            return record
    try:
        candidates = urls() if callable(urls) else urls
    except FetchError as error:
        return _failure(record, name, [str(error)], transient=not error.permanent)
    return _try_urls(
        context,
        record,
        name=name,
        target=target,
        candidates=candidates,
        validator=validator,
        headers=headers,
    )


def _try_urls(
    context: DownloadContext,
    record: EntryRecord,
    *,
    name: str,
    target: Path,
    candidates: Sequence[str],
    validator: Validator,
    headers: Mapping[str, str] | None,
) -> EntryRecord:
    failures: list[str] = []
    transient = False
    for url in candidates:
        try:
            fetched = context.client.fetch_file(url, target, validator, headers)
        except FetchError as error:
            logger.debug("%s: %s", name, error)
            failures.append(str(error))
            transient = transient or not error.permanent
            continue
        return record.with_file(FileRecord.from_fetched(name, fetched))
    if not candidates:
        failures.append("no candidate URL")
    return _failure(record, name, failures, transient)


def _failure(
    record: EntryRecord, name: str, failures: Sequence[str], transient: bool
) -> EntryRecord:
    reason = "; ".join(failures)
    return (
        record.with_error(name, reason)
        if transient
        else record.with_missing(name, reason)
    )


def _adopt(record: EntryRecord, name: str, target: Path) -> EntryRecord:
    """Register a file already on disk, keeping known provenance if any."""
    known = record.files.get(name)
    if (
        known is not None
        and known.path == target.name
        and known.size == target.stat().st_size
    ):
        return record
    data = target.read_bytes()
    return record.with_file(
        FileRecord(
            name=name,
            path=target.name,
            url=known.url if known else MANUAL_URL,
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            content_type="",
        )
    )
