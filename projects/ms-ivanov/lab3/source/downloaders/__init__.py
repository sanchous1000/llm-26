"""Per-kind downloaders that turn bibliography entries into local files."""

from source.downloaders.base import DownloadContext, Downloader, PdfPolicy
from source.downloaders.registry import build_downloaders

__all__ = ["DownloadContext", "Downloader", "PdfPolicy", "build_downloaders"]
