"""Value objects of the conversion layer."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

_HEADING_RE = re.compile(r"^#{1,6} ", re.MULTILINE)
_TABLE_SEPARATOR_RE = re.compile(r"^\|[-:| ]+\|$", re.MULTILINE)
_DISPLAY_MATH_RE = re.compile(r"^\$\$$", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


@dataclass(frozen=True)
class ConversionResult:
    """Markdown body (without front matter) produced by a converter."""

    markdown: str
    warnings: tuple[str, ...] = ()
    extra: Mapping[str, str] = field(default_factory=dict)


class Converter(Protocol):
    """Turns one downloaded file into Markdown."""

    name: str

    def convert(self, path: Path, base_url: str) -> ConversionResult:
        """Read ``path`` (downloaded from ``base_url``) and return Markdown."""


def markdown_stats(markdown: str) -> dict[str, int]:
    """Cheap structural statistics, identical for every converter."""
    return {
        "chars": len(markdown),
        "headings": len(_HEADING_RE.findall(markdown)),
        "tables": len(_TABLE_SEPARATOR_RE.findall(markdown)),
        "display_equations": len(_DISPLAY_MATH_RE.findall(markdown)) // 2,
        "images": len(_IMAGE_RE.findall(markdown)),
    }
