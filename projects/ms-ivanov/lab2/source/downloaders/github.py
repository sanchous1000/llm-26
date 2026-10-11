"""GitHub repositories: the README is the documentation worth indexing."""

import os
from pathlib import Path
from urllib.parse import urlparse

from source import config
from source.bibliography.models import Source
from source.downloaders.base import DownloadContext, acquire
from source.net.validators import require_text
from source.storage.records import EntryRecord

README_FILENAME = "README.md"
_RAW_ACCEPT = "application/vnd.github.raw+json"


class GithubDownloader:
    """Downloads the raw README of a repository through the GitHub API."""

    def __init__(self, context: DownloadContext):
        self._context = context

    def download(
        self, source: Source, record: EntryRecord, entry_dir: Path
    ) -> EntryRecord:
        repo = parse_repository(source.url)
        if repo is None:
            return record.with_missing("readme", f"unsupported GitHub URL {source.url}")
        owner, name = repo
        return acquire(
            self._context,
            entry_dir,
            record,
            name="readme",
            filename=README_FILENAME,
            urls=[config.GITHUB_README_URL.format(owner=owner, repo=name)],
            validator=require_text,
            headers=_api_headers(),
        )


def parse_repository(url: str) -> tuple[str, str] | None:
    """Return ``(owner, repo)`` for a github.com URL or ``None``."""
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc.lower().removeprefix("www.") != "github.com" or len(parts) < 2:
        return None
    return parts[0], parts[1].removesuffix(".git")


def _api_headers() -> dict[str, str]:
    headers = {"Accept": _RAW_ACCEPT}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers
