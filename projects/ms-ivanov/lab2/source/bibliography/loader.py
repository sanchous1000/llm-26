"""Combine the BibTeX data with the curated catalog annotations."""

import logging
from dataclasses import replace
from pathlib import Path

from source.bibliography.bib_parser import parse_bib_file
from source.bibliography.catalog_parser import CatalogEntry, parse_catalog
from source.bibliography.models import Source

logger = logging.getLogger(__name__)


def load_sources(bib_path: Path, catalog_path: Path | None = None) -> list[Source]:
    """Load all sources; enrich them from the catalog when it is available."""
    sources = parse_bib_file(bib_path)
    if catalog_path is None or not catalog_path.exists():
        return sources
    catalog = parse_catalog(catalog_path)
    missing = [source.key for source in sources if source.key not in catalog]
    if missing:
        logger.warning(
            "%d keys have no catalog entry, e.g. %s", len(missing), missing[:3]
        )
    return [_enrich(source, catalog.get(source.key)) for source in sources]


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
