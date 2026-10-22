"""Selection of sources and the resumable download loop."""

import logging
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime

from source.bibliography.models import Source, SourceKind
from source.downloaders.base import Downloader
from source.storage import EntryRecord, EntryStore

logger = logging.getLogger(__name__)

# Cheap, fast hosts first; the throttled arXiv queue goes last.
KIND_ORDER = (SourceKind.GITHUB, SourceKind.WEB, SourceKind.DOI, SourceKind.ARXIV)


def select_sources(
    sources: Sequence[Source],
    *,
    kinds: Collection[SourceKind] = (),
    keys: Collection[str] = (),
    limit: int | None = None,
) -> list[Source]:
    """Filter by kind/key and order by kind, then by priority (core first)."""
    selected = [
        source
        for source in sources
        if (not kinds or source.kind in kinds) and (not keys or source.key in keys)
    ]
    selected.sort(key=_sort_key)
    return selected[:limit] if limit is not None else selected


def _sort_key(source: Source) -> tuple[int, int, str]:
    return KIND_ORDER.index(source.kind), -(source.priority or 0), source.key


def run_pipeline(
    sources: Sequence[Source],
    store: EntryStore,
    downloaders: Mapping[SourceKind, Downloader],
) -> list[EntryRecord]:
    """Download every source, persisting the outcome after each one.

    A failure in one source is logged and recorded; it never stops the run.
    Interrupting with Ctrl-C is safe: finished entries are already saved.
    """
    records: list[EntryRecord] = []
    total = len(sources)
    for index, source in enumerate(sources, start=1):
        record = _process(source, store, downloaders[source.kind])
        records.append(record)
        logger.info(
            "[%d/%d] %-6s %-45s %s %s",
            index,
            total,
            source.kind.value,
            source.key,
            record.status,
            _describe(record),
        )
    return records


def _process(source: Source, store: EntryStore, downloader: Downloader) -> EntryRecord:
    previous = store.load(source) or EntryRecord(source=source)
    start = previous.without_errors()
    try:
        record = downloader.download(source, start, store.entry_dir(source))
    except Exception as error:  # noqa: BLE001 - recorded, run must continue
        logger.exception("Unexpected failure for %s", source.key)
        record = start.with_error("unexpected", f"{type(error).__name__}: {error}")
    if not record.same_outcome_as(previous) or previous.updated_at is None:
        record = record.with_timestamp(datetime.now(UTC).isoformat(timespec="seconds"))
    else:
        record = record.with_timestamp(previous.updated_at)
    store.save(record)
    return record


def _describe(record: EntryRecord) -> str:
    parts = [f"files={','.join(record.files) or '-'}"]
    parts += [
        f"{name}: {reason}"
        for name, reason in {**record.missing, **record.errors}.items()
    ]
    return " | ".join(parts)
