"""Repository README files: already Markdown (sometimes reStructuredText)."""

import re
from pathlib import Path

from source.parsing import pandoc, postprocess
from source.parsing.models import ConversionResult

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BADGE_RE = re.compile(
    r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!\[[^\]]*\]\(https?://(?:img\.shields\.io|badge)[^)]*\)"
)
_RST_UNDERLINE_RE = re.compile(r"^[=\-~^\"]{4,}\s*$", re.MULTILINE)
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6} ", re.MULTILINE)


class ReadmeConverter:
    """Keeps README text, drops badges and HTML comments."""

    name = "readme@1"

    def convert(self, path: Path, base_url: str) -> ConversionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        if _looks_like_rst(text):
            text = pandoc.convert(text, "rst")
        text = _HTML_COMMENT_RE.sub("", text)
        text = _BADGE_RE.sub("", text)
        return ConversionResult(markdown=postprocess.tidy(text))


def _looks_like_rst(text: str) -> bool:
    return not _MARKDOWN_HEADING_RE.search(text) and bool(
        _RST_UNDERLINE_RE.search(text)
    )
