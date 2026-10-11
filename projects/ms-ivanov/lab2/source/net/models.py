"""Value objects shared by the network layer."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FetchedFile:
    """A file that has been downloaded and stored on disk."""

    path: Path
    url: str
    size: int
    sha256: str
    content_type: str
