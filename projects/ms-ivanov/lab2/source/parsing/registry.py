"""Which converter handles which downloaded file."""

from source.parsing.latexml import LatexmlConverter
from source.parsing.models import Converter
from source.parsing.pdf import PdfConverter
from source.parsing.readme import ReadmeConverter
from source.parsing.webpage import WebPageConverter

# Logical file names as recorded by the downloaders (FileRecord.name).
_CONVERTER_FACTORIES = {
    "html": LatexmlConverter,
    "pdf": PdfConverter,
    "readme": ReadmeConverter,
    "page": WebPageConverter,
}


def build_converter(file_name: str, *, include_references: bool = True) -> Converter:
    """Return the converter for a downloaded file role (html, pdf, ...)."""
    factory = _CONVERTER_FACTORIES[file_name]
    if factory is LatexmlConverter:
        return LatexmlConverter(include_references=include_references)
    return factory()
