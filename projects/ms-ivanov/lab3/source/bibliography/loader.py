"""Combine the BibTeX data with the curated catalog annotations."""

import logging
from dataclasses import replace
from pathlib import Path

from source import config
from source.bibliography.additions_parser import parse_additions_table
from source.bibliography.bib_parser import parse_bib_file
from source.bibliography.catalog_parser import CatalogEntry, parse_catalog
from source.bibliography.models import Source

logger = logging.getLogger(__name__)


def load_sources(bib_path: Path, catalog_path: Path | None = None) -> list[Source]:
    """Load all sources; enrich them from the catalog when it is available.

    Extra sources placed next to the main files (``rag_additions<suffix>.bib``
    with ``RAG_ADDITIONS<suffix>.md``) are appended in file-name order; an extra
    entry with a key that already exists only contributes its open-access link.
    """
    merged = _enriched(parse_bib_file(bib_path), _read_catalog(catalog_path))
    for additions_bib in sorted(
        bib_path.parent.glob(f"{config.ADDITIONS_BIB_PREFIX}*.bib")
    ):
        table = additions_bib.with_name(
            config.ADDITIONS_CATALOG_PREFIX
            + additions_bib.stem.removeprefix(config.ADDITIONS_BIB_PREFIX)
            + ".md"
        )
        extra = _enriched(
            parse_bib_file(additions_bib),
            parse_additions_table(table) if table.exists() else {},
        )
        before = len(merged)
        merged = _merge(merged, extra)
        logger.info(
            "%s: %d new sources of %d entries",
            additions_bib.name,
            len(merged) - before,
            len(extra),
        )
    return merged


def _read_catalog(path: Path | None) -> dict[str, CatalogEntry]:
    return parse_catalog(path) if path is not None and path.exists() else {}


def _enriched(sources: list[Source], catalog: dict[str, CatalogEntry]) -> list[Source]:
    if not catalog:
        return sources
    missing = [source.key for source in sources if source.key not in catalog]
    if missing:
        logger.warning(
            "%d keys have no catalog entry, e.g. %s", len(missing), missing[:3]
        )
    return [_enrich(source, catalog.get(source.key)) for source in sources]


def _merge(base: list[Source], extra: list[Source]) -> list[Source]:
    by_key = {source.key: source for source in base}
    merged = list(base)
    for source in extra:
        known = by_key.get(source.key)
        if known is None:
            merged.append(source)
        elif known.open_access_url is None and source.open_access_url:
            merged[merged.index(known)] = replace(
                known, open_access_url=source.open_access_url
            )
    return merged


def _enrich(source: Source, entry: CatalogEntry | None) -> Source:
    if entry is None:
        return source
    return replace(
        source,
        priority=entry.priority,
        tags=entry.tags,
        section=entry.section,
        summary=entry.summary,
    )
