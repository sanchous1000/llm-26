"""YAML front matter with the metadata every chunk will inherit."""

from datetime import datetime

import yaml

from source.bibliography.models import Source
from source.storage.records import FileRecord


def build_front_matter(
    source: Source,
    original: FileRecord,
    entry_path: str,
    converter: str,
    converted_at: datetime,
) -> str:
    """Render the ``---`` delimited metadata block."""
    data = {
        "key": source.key,
        "title": source.title,
        "authors": list(source.authors),
        "year": source.year,
        "venue": source.venue,
        "kind": source.kind.value,
        "source_url": source.url,
        "arxiv_id": source.arxiv_id,
        "doi": source.doi,
        "priority": source.priority,
        "tags": list(source.tags),
        "catalog_section": source.section,
        "original_file": f"{entry_path}/{original.path}",
        "original_url": original.url,
        "converter": converter,
        "converted_at": converted_at.isoformat(timespec="seconds"),
    }
    data = {key: value for key, value in data.items() if value not in (None, [], "")}
    dumped = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100)
    return f"---\n{dumped}---\n\n"
