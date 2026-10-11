"""Domain model of a bibliography entry."""

from dataclasses import dataclass
from enum import StrEnum


class SourceKind(StrEnum):
    """How a source has to be acquired."""

    ARXIV = "arxiv"
    DOI = "doi"
    GITHUB = "github"
    WEB = "web"


@dataclass(frozen=True)
class Source:
    """One bibliography entry, enriched with the curated catalog data."""

    key: str
    kind: SourceKind
    title: str
    url: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    priority: int | None = None
    tags: tuple[str, ...] = ()
    section: str | None = None
    summary: str | None = None

    def to_metadata(self) -> dict:
        """Plain JSON-serialisable view used in ``meta.json``."""
        return {
            "key": self.key,
            "kind": self.kind.value,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "priority": self.priority,
            "tags": list(self.tags),
            "section": self.section,
            "summary": self.summary,
        }
