"""Immutable description of what has been downloaded for one source."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from source.bibliography.models import Source
from source.net.models import FetchedFile

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"

# Which downloaded file is the best input for the later parsing step.
PRIMARY_PREFERENCE = ("html", "readme", "pdf", "page")
MANUAL_URL = "manual"


def _frozen(data: Mapping | None = None) -> Mapping:
    return MappingProxyType(dict(data or {}))


@dataclass(frozen=True)
class FileRecord:
    """One stored file (``name`` is the logical role: html, pdf, readme, page)."""

    name: str
    path: str
    url: str
    size: int
    sha256: str
    content_type: str

    @classmethod
    def from_fetched(cls, name: str, fetched: FetchedFile) -> "FileRecord":
        return cls(
            name=name,
            path=fetched.path.name,
            url=fetched.url,
            size=fetched.size,
            sha256=fetched.sha256,
            content_type=fetched.content_type,
        )

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "url": self.url,
            "size": self.size,
            "sha256": self.sha256,
            "content_type": self.content_type,
        }

    @classmethod
    def from_dict(cls, name: str, data: Mapping) -> "FileRecord":
        return cls(
            name=name,
            **{
                key: data[key]
                for key in ("path", "url", "size", "sha256", "content_type")
            },
        )


@dataclass(frozen=True)
class EntryRecord:
    """State of one source: stored files plus why other files are absent.

    ``missing`` holds permanent absences (404, no open-access copy) that are
    not retried by default; ``errors`` holds transient failures that are.
    """

    source: Source
    files: Mapping[str, FileRecord] = field(default_factory=_frozen)
    missing: Mapping[str, str] = field(default_factory=_frozen)
    errors: Mapping[str, str] = field(default_factory=_frozen)
    updated_at: str | None = None
    conversion: Mapping[str, Any] = field(default_factory=_frozen)

    @property
    def primary(self) -> FileRecord | None:
        for name in PRIMARY_PREFERENCE:
            if name in self.files:
                return self.files[name]
        return None

    @property
    def status(self) -> str:
        if self.files:
            return STATUS_OK
        return STATUS_ERROR if self.errors else STATUS_UNAVAILABLE

    def with_file(self, record: FileRecord) -> "EntryRecord":
        return replace(
            self,
            files=_frozen({**self.files, record.name: record}),
            missing=_without(self.missing, record.name),
            errors=_without(self.errors, record.name),
        )

    def with_missing(self, name: str, reason: str) -> "EntryRecord":
        return replace(
            self,
            files=_without(self.files, name),
            missing=_frozen({**self.missing, name: reason}),
            errors=_without(self.errors, name),
        )

    def with_error(self, name: str, reason: str) -> "EntryRecord":
        return replace(
            self,
            files=_without(self.files, name),
            missing=_without(self.missing, name),
            errors=_frozen({**self.errors, name: reason}),
        )

    def without_errors(self) -> "EntryRecord":
        return replace(self, errors=_frozen())

    def with_conversion(self, conversion: Mapping[str, Any]) -> "EntryRecord":
        return replace(self, conversion=_frozen(conversion))

    def with_timestamp(self, timestamp: str) -> "EntryRecord":
        return replace(self, updated_at=timestamp)

    def same_outcome_as(self, other: "EntryRecord") -> bool:
        """True when files and failure reasons are identical (ignores timestamps)."""
        return (self.files, self.missing, self.errors) == (
            other.files,
            other.missing,
            other.errors,
        )

    def to_dict(self) -> dict:
        primary = self.primary
        return {
            **self.source.to_metadata(),
            "status": self.status,
            "primary": primary.name if primary else None,
            "files": {name: file.to_dict() for name, file in self.files.items()},
            "missing": dict(self.missing),
            "errors": dict(self.errors),
            "updated_at": self.updated_at,
            "conversion": dict(self.conversion),
        }

    @classmethod
    def from_dict(cls, source: Source, data: Mapping) -> "EntryRecord":
        files = {
            name: FileRecord.from_dict(name, item)
            for name, item in data.get("files", {}).items()
        }
        return cls(
            source=source,
            files=_frozen(files),
            missing=_frozen(data.get("missing")),
            errors=_frozen(data.get("errors")),
            updated_at=data.get("updated_at"),
            conversion=_frozen(data.get("conversion")),
        )


def _without(mapping: Mapping, name: str) -> Mapping:
    return _frozen({key: value for key, value in mapping.items() if key != name})
