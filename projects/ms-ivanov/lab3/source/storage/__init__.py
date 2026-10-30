"""On-disk layout of the downloaded corpus."""

from source.storage.records import EntryRecord, FileRecord
from source.storage.store import EntryStore

__all__ = ["EntryRecord", "EntryStore", "FileRecord"]
