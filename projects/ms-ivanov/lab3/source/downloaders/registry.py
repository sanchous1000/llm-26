"""Mapping from source kind to the downloader that handles it."""

from source.bibliography.models import SourceKind
from source.downloaders.arxiv import ArxivDownloader
from source.downloaders.base import DownloadContext, Downloader
from source.downloaders.doi import DoiDownloader
from source.downloaders.github import GithubDownloader
from source.downloaders.web import WebDownloader


def build_downloaders(context: DownloadContext) -> dict[SourceKind, Downloader]:
    """Create one downloader per source kind, sharing ``context``."""
    return {
        SourceKind.ARXIV: ArxivDownloader(context),
        SourceKind.DOI: DoiDownloader(context),
        SourceKind.GITHUB: GithubDownloader(context),
        SourceKind.WEB: WebDownloader(context),
    }
