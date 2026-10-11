"""LaTeXML (arXiv / ar5iv) HTML -> Markdown.

The document is cleaned with BeautifulSoup into plain semantic HTML (math and
code are swapped for tokens) and then converted by Pandoc.
"""

from pathlib import Path

from bs4 import BeautifulSoup, Tag

from source.parsing import pandoc, postprocess
from source.parsing.errors import ConversionError
from source.parsing.latexml import blocks, headings, inline
from source.parsing.latexml.bibliography import convert_bibliography
from source.parsing.latexml.context import Context
from source.parsing.latexml.figures import convert_figures, convert_images
from source.parsing.latexml.math import convert_equation_tables, convert_math
from source.parsing.latexml.noise import remove_noise
from source.parsing.latexml.sanitize import sanitize
from source.parsing.latexml.tables import convert_tables
from source.parsing.models import ConversionResult

CONVERTER_NAME = "latexml-html@1"


class LatexmlConverter:
    """Converts a downloaded arXiv HTML page into Markdown."""

    def __init__(self, include_references: bool = True):
        self._include_references = include_references
        self.name = CONVERTER_NAME + ("" if include_references else "+norefs")

    def convert(self, path: Path, base_url: str) -> ConversionResult:
        html = path.read_text(encoding="utf-8", errors="replace")
        return self.convert_html(html, base_url)

    def convert_html(self, html: str, base_url: str) -> ConversionResult:
        article = _find_article(BeautifulSoup(html, "lxml"))
        context = Context(base_url=base_url)
        title = self._prepare(article, context)
        cleaned = (
            f"<h1>{title}</h1>\n{article.decode_contents()}"
            if title
            else article.decode_contents()
        )
        markdown = pandoc.convert(cleaned, "html")
        markdown = postprocess.tidy(context.store.restore(markdown))
        if len(markdown) < 200:
            context.warnings.append("converted document is suspiciously short")
        return ConversionResult(markdown=markdown, warnings=tuple(context.warnings))

    def _prepare(self, article: Tag, context: Context) -> str:
        """Run all DOM passes in dependency order; returns the title HTML."""
        remove_noise(article)
        blocks.convert_listings(article, context)
        convert_equation_tables(article, context)
        convert_math(article, context)
        title = headings.extract_title(article)
        inline.strip_theorem_italics(article)
        inline.convert_fonts(article)
        inline.resolve_links(article, context.base_url)
        inline.inline_footnotes(article)
        headings.promote_front_matter_titles(article)
        headings.merge_runin_titles(article)
        headings.number_headings(article)
        inline.replace_empty_titles(article)
        convert_images(article, context.base_url)
        convert_figures(article)
        convert_bibliography(article, self._include_references)
        blocks.convert_lists(article)
        convert_tables(article)
        sanitize(article)
        return title


def _find_article(soup: BeautifulSoup) -> Tag:
    article = soup.find("article", class_="ltx_document") or soup.find(
        "div", class_="ltx_page_content"
    )
    if article is None:
        raise ConversionError("no LaTeXML article found (not a rendered arXiv paper?)")
    return article
