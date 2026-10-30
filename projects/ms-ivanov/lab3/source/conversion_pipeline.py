"""Resumable, parallel conversion of downloaded files into ``document.md``."""

import logging
import os
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from source.parsing.frontmatter import build_front_matter
from source.parsing.models import markdown_stats
from source.parsing.registry import build_converter
from source.storage import EntryRecord, EntryStore, FileRecord

logger = logging.getLogger(__name__)

DOCUMENT_FILENAME = "document.md"


@dataclass(frozen=True)
class Job:
    """Picklable description of one conversion."""

    key: str
    file_name: str
    path: str
    base_url: str
    include_references: bool


@dataclass(frozen=True)
class Outcome:
    """What a worker sends back to the main process."""

    key: str
    converter: str
    markdown: str | None
    warnings: tuple[str, ...]
    error: str | None


def run_conversion(
    records: Sequence[EntryRecord],
    store: EntryStore,
    *,
    force: bool = False,
    include_references: bool = True,
    workers: int | None = None,
    on_done: Callable[[EntryRecord], None] | None = None,
) -> list[EntryRecord]:
    """Convert every record that is not up to date; return the updated records."""
    by_key = {record.source.key: record for record in records}
    jobs = [
        _job_for(record, store, include_references)
        for record in records
        if _needs_conversion(record, store, force, include_references)
    ]
    logger.info("Converting %d of %d entries", len(jobs), len(records))
    for outcome in _execute(jobs, workers):
        record = _apply(by_key[outcome.key], outcome, store)
        by_key[outcome.key] = record
        logger.info("[%s] %s", record.source.key, _summary(record))
        if on_done:
            on_done(record)
    return [by_key[record.source.key] for record in records]


def _needs_conversion(
    record: EntryRecord, store: EntryStore, force: bool, include_references: bool
) -> bool:
    primary = record.primary
    if primary is None:
        return False
    if force:
        return True
    expected = build_converter(primary.name, include_references=include_references).name
    done = record.conversion
    document = store.entry_dir(record.source) / DOCUMENT_FILENAME
    return not (
        done.get("status") == "ok"
        and done.get("converter") == expected
        and done.get("source_sha256") == primary.sha256
        and document.exists()
    )


def _job_for(record: EntryRecord, store: EntryStore, include_references: bool) -> Job:
    primary = record.primary
    assert primary is not None  # guaranteed by _needs_conversion
    path = store.entry_dir(record.source) / primary.path
    return Job(
        record.source.key, primary.name, str(path), primary.url, include_references
    )


def _execute(jobs: Sequence[Job], workers: int | None):
    """Yield outcomes as they finish (inline for one job or one worker)."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1 or len(jobs) <= 1:
        yield from (convert_job(job) for job in jobs)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures: list[Future[Outcome]] = [pool.submit(convert_job, job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()


def convert_job(job: Job) -> Outcome:
    """Worker entry point: never raises, failures travel back as ``error``."""
    converter = build_converter(
        job.file_name, include_references=job.include_references
    )
    try:
        result = converter.convert(Path(job.path), job.base_url)
    except Exception as error:  # noqa: BLE001 - reported per document
        return Outcome(
            job.key, converter.name, None, (), f"{type(error).__name__}: {error}"
        )
    return Outcome(job.key, converter.name, result.markdown, result.warnings, None)


def _apply(record: EntryRecord, outcome: Outcome, store: EntryStore) -> EntryRecord:
    primary = record.primary
    assert primary is not None
    now = datetime.now(UTC)
    base = {
        "converter": outcome.converter,
        "source_file": primary.path,
        "source_sha256": primary.sha256,
        "converted_at": now.isoformat(timespec="seconds"),
    }
    if outcome.markdown is None:
        logger.error("Conversion of %s failed: %s", record.source.key, outcome.error)
        updated = record.with_conversion(
            {**base, "status": "error", "error": outcome.error}
        )
    else:
        stats = _write_document(record, primary, outcome, store, now)
        updated = record.with_conversion(
            {
                **base,
                "status": "ok",
                "path": DOCUMENT_FILENAME,
                "warnings": list(outcome.warnings),
                **stats,
            }
        )
    store.save(updated)
    return updated


def _write_document(
    record: EntryRecord,
    primary: FileRecord,
    outcome: Outcome,
    store: EntryStore,
    now: datetime,
) -> dict[str, int]:
    entry_dir = store.entry_dir(record.source)
    entry_path = entry_dir.relative_to(store.root).as_posix()
    header = build_front_matter(
        record.source, primary, entry_path, outcome.converter, now
    )
    target = entry_dir / DOCUMENT_FILENAME
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(header + (outcome.markdown or ""), encoding="utf-8")
    os.replace(temporary, target)
    return markdown_stats(outcome.markdown or "")


def _summary(record: EntryRecord) -> str:
    done = record.conversion
    if done.get("status") != "ok":
        return f"ERROR {done.get('error')}"
    warnings = f" warnings={done['warnings']}" if done.get("warnings") else ""
    return (
        f"ok {done['converter']} chars={done['chars']} headings={done['headings']} "
        f"tables={done['tables']} eq={done['display_equations']}{warnings}"
    )
