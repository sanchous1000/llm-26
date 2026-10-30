"""Reading and writing per-entry metadata and the corpus manifest.

Layout::

    data/papers/
      manifest.json                 # index of every entry
      <kind>/<key>/meta.json        # source metadata + download outcome
      <kind>/<key>/<downloaded files>
"""

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path

from source import config
from source.bibliography.models import Source
from source.storage.records import EntryRecord

logger = logging.getLogger(__name__)


class EntryStore:
    """Filesystem persistence for :class:`EntryRecord` objects."""

    def __init__(self, root: Path):
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def entry_dir(self, source: Source) -> Path:
        return self._root / source.kind.value / source.key

    def load(self, source: Source) -> EntryRecord | None:
        """Return the stored record of ``source`` or ``None`` if absent/corrupt."""
        meta_path = self.entry_dir(source) / config.META_FILENAME
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            return EntryRecord.from_dict(source, data)
        except (ValueError, KeyError, TypeError):
            logger.exception("Ignoring corrupt metadata file %s", meta_path)
            return None

    def save(self, record: EntryRecord) -> None:
        meta_path = self.entry_dir(record.source) / config.META_FILENAME
        self._write_json(meta_path, record.to_dict())

    def write_manifest(self, records: Iterable[EntryRecord]) -> Path:
        """Write ``manifest.json`` with paths relative to the corpus root."""
        entries = [self._manifest_item(record) for record in records]
        entries.sort(key=lambda item: (item["kind"], item["key"]))
        manifest_path = self._root / config.MANIFEST_FILENAME
        self._write_json(manifest_path, {"count": len(entries), "entries": entries})
        return manifest_path

    def _manifest_item(self, record: EntryRecord) -> dict:
        entry_dir = self.entry_dir(record.source).relative_to(self._root)
        data = record.to_dict()
        data["dir"] = entry_dir.as_posix()
        if data["conversion"].get("status") == "ok":
            data["markdown"] = (entry_dir / data["conversion"]["path"]).as_posix()
        data["files"] = {
            name: {**file, "path": (entry_dir / file["path"]).as_posix()}
            for name, file in data["files"].items()
        }
        return data

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
