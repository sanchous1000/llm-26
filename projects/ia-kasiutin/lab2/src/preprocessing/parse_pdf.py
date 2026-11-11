from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

LAB2_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = LAB2_ROOT / "docs" / "diploma.pdf"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "diploma.md"

_LATEX_MTTF = (
    "$$\n"
    r"\mathrm{MTTF} = \frac{1}{n} \sum_{i=1}^{n}"
    r" (t_{\text{сбой},i} - t_{\text{нач.работы},i})"
    "\n$$"
)
# Правила восстановления формул, где текстовый слой PDF / OCR искажает глифы.
FORMULA_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"MTTF\s*=\s*1?\s*𝑛+\s*[×x]\s*∑\s*\(\s*𝑡+\s*сбой\s*,\s*𝑖+\s*[−–-]\s*𝑡+\s*нач\.работы\s*,\s*𝑖+\s*\)\s*𝑛+\s*𝑖+\s*=\s*1",
            re.IGNORECASE,
        ),
        _LATEX_MTTF,
    ),
    (
        re.compile(r"\$\$\s*M\s*T\s*T\s*F\s*=\s*\\frac\s*\{\s*1\s*\}\s*\{\s*n\s*\}[^$]*\$\$"),
        _LATEX_MTTF,
    ),
)


def pdf_modified_date(path: Path) -> str:
    match = re.search(br"/ModDate\s*\(D:(\d{8})", path.read_bytes())
    if match is None:
        return ""
    raw = match.group(1).decode()
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def front_matter(pdf_path: Path) -> str:
    relative = pdf_path.relative_to(LAB2_ROOT).as_posix()
    return (
        "---\n"
        f"source: {pdf_path.name}\n"
        f"path: {relative}\n"
        f"modified: {pdf_modified_date(pdf_path)}\n"
        "---\n"
    )


def pipeline_options() -> PdfPipelineOptions:
    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_formula_enrichment = True
    options.generate_picture_images = False
    options.generate_page_images = False
    return options


def convert(pdf_path: Path) -> DoclingDocument:
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options())
        }
    )
    return converter.convert(pdf_path).document


def page_count(document: DoclingDocument) -> int:
    return max((int(page) for page in document.pages), default=0)


def page_marker(source_name: str, page: int) -> str:
    return f"<!-- source: {source_name} page: {page} -->"


def normalize_page_markdown(markdown: str) -> str:
    for pattern, replacement in FORMULA_FIXES:
        markdown = pattern.sub(lambda _match, r=replacement: r, markdown)
    return markdown


def iter_pages(document: DoclingDocument) -> Iterator[tuple[int, str]]:
    """Возвращает (страница, markdown страницы); тело может быть пустым."""
    for page in range(1, page_count(document) + 1):
        body = document.export_to_markdown(page_no=page, image_placeholder="").strip()
        yield page, normalize_page_markdown(body)


def render(document: DoclingDocument, pdf_path: Path) -> str:
    parts = [front_matter(pdf_path).rstrip()]
    for page, body in iter_pages(document):
        parts.append(page_marker(pdf_path.name, page))
        if body:
            parts.append(body)
    return "\n\n".join(parts) + "\n"


def main() -> None:
    if not PDF_PATH.is_file():
        raise SystemExit(f"Не найден PDF: {PDF_PATH}")
    document = convert(PDF_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render(document, PDF_PATH), encoding="utf-8")
    print(f"Записано {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
