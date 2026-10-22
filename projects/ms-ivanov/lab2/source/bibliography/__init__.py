"""Reading the bibliography (``sources.bib`` + ``SOURCES.md``)."""

from source.bibliography.loader import load_sources
from source.bibliography.models import Source, SourceKind

__all__ = ["Source", "SourceKind", "load_sources"]
