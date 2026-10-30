"""Generic web pages -> Markdown (main content only) through trafilatura."""

from pathlib import Path

import trafilatura

from source.parsing import pandoc, postprocess
from source.parsing.models import ConversionResult


class WebPageConverter:
    """Strips navigation and boilerplate, keeps headings, lists and tables."""

    name = "trafilatura@1"

    def convert(self, path: Path, base_url: str) -> ConversionResult:
        html = path.read_text(encoding="utf-8", errors="replace")
        extracted = trafilatura.extract(
            html,
            url=base_url,
            output_format="markdown",
            include_tables=True,
            include_links=True,
            include_formatting=True,
            include_images=False,
            favor_recall=True,
        )
        warnings: tuple[str, ...] = ()
        if not extracted or len(extracted.strip()) < 200:
            warnings = ("main-content extraction failed; used the whole page",)
            extracted = pandoc.convert(html, "html")
        return ConversionResult(markdown=postprocess.tidy(extracted), warnings=warnings)
