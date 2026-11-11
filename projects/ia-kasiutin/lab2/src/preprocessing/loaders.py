from __future__ import annotations

import argparse
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from . import parse_pdf

OUTPUT_DIR = parse_pdf.OUTPUT_PATH.parent
SOURCES_DIR = parse_pdf.LAB2_ROOT / "docs"
_PAGE_SPLIT = re.compile(r"<!--\s*source:\s*\S+\s+page:\s*(\d+)[^>]*-->")


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(parse_pdf.LAB2_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _modified_date(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()


def _parse_markdown_pages(markdown: str, base_meta: dict) -> list[Document]:
    parts = _PAGE_SPLIT.split(markdown)
    docs: list[Document] = []
    for idx in range(1, len(parts), 2):
        page = int(parts[idx])
        body = parts[idx + 1].strip()
        if body:
            docs.append(Document(page_content=body, metadata={**base_meta, "page": page}))
    return docs


class DoclingPageLoader(BaseLoader):
    """PDF -> по одному Document на страницу из кэшированного `cache_dir/<stem>.md`."""

    def __init__(
        self,
        file_path: str | Path,
        cache_dir: str | Path = OUTPUT_DIR,
        reparse: bool = False,
    ) -> None:
        self.file_path = Path(file_path)
        self.cache_dir = Path(cache_dir)
        self.reparse = reparse

    @property
    def _markdown_path(self) -> Path:
        return self.cache_dir / f"{self.file_path.stem}.md"

    def lazy_load(self) -> Iterator[Document]:
        if self.reparse or not self._markdown_path.is_file():
            document = parse_pdf.convert(self.file_path)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._markdown_path.write_text(
                parse_pdf.render(document, self.file_path), encoding="utf-8"
            )
        base = {
            "source": self.file_path.name,
            "path": _relative_path(self.file_path),
            "modified": parse_pdf.pdf_modified_date(self.file_path),
            "format": "pdf",
            "parser": "docling",
        }
        yield from _parse_markdown_pages(
            self._markdown_path.read_text(encoding="utf-8"), base
        )


class MarkdownFileLoader(BaseLoader):
    """MD -> один Document на файл."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def lazy_load(self) -> Iterator[Document]:
        yield Document(
            page_content=self.file_path.read_text(encoding="utf-8"),
            metadata={
                "source": self.file_path.name,
                "path": _relative_path(self.file_path),
                "modified": _modified_date(self.file_path),
                "format": "md",
                "parser": "plain",
            },
        )


class DoclingGenericLoader(BaseLoader):
    """DOCX/PPTX/HTML -> один Document на файл через Docling (без страниц)."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def lazy_load(self) -> Iterator[Document]:
        from docling.document_converter import DocumentConverter

        document = DocumentConverter().convert(self.file_path).document
        yield Document(
            page_content=document.export_to_markdown(image_placeholder=""),
            metadata={
                "source": self.file_path.name,
                "path": _relative_path(self.file_path),
                "modified": _modified_date(self.file_path),
                "format": self.file_path.suffix.lstrip(".").lower(),
                "parser": "docling",
            },
        )


def loader_for(path: Path, reparse: bool = False) -> BaseLoader:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return DoclingPageLoader(path, reparse=reparse)
    if suffix in {".md", ".markdown"}:
        return MarkdownFileLoader(path)
    if suffix in {".docx", ".pptx", ".html", ".htm"}:
        return DoclingGenericLoader(path)
    raise ValueError(f"Неподдерживаемый формат: {path}")


def load_documents(
    sources_dir: str | Path = SOURCES_DIR, reparse: bool = False
) -> list[Document]:
    supported = {".pdf", ".md", ".markdown"}
    documents: list[Document] = []
    for path in sorted(Path(sources_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            documents.extend(loader_for(path, reparse=reparse).load())
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Парсинг источников в LangChain Documents")
    parser.add_argument("--sources", default=str(SOURCES_DIR))
    parser.add_argument("--reparse", action="store_true", help="игнорировать кэш")
    args = parser.parse_args()

    docs = load_documents(args.sources, reparse=args.reparse)
    print(f"Documents: {len(docs)}")
    if docs:
        print("Пример metadata:", docs[0].metadata)
    print("Формулы MTTF (LaTeX):", sum(r"\mathrm{MTTF}" in d.page_content for d in docs))


if __name__ == "__main__":
    main()
