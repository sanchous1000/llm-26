import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import pdfplumber
from tabulate import tabulate


@dataclass
class Page:
    number: int
    markdown: str


@dataclass
class Document:
    source: str  # путь относительно raw_dir
    date: str
    pages: list[Page] = field(default_factory=list)


class ParserService:

    def __init__(self):
        pass

    def parse_pdf_pages(self, file_bytes: bytes) -> list[Page]:
        pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for number, page in enumerate(pdf.pages, 1):
                tables = page.find_tables()
                if tables:
                    text = self.__extract_with_tables(page, tables)
                else:
                    text = page.extract_text(x_tolerance=1.5) or ""  # иначе слова слипаются в плотном наборе
                if text.strip():
                    pages.append(Page(number, text))
        return pages

    def __extract_with_tables(self, page, tables):
        page_tables = []
        for i, table in enumerate(tables):
            page_tables.append({
                "id": i,
                "text": self.__table_to_md(table.extract()),
                "bbox": {
                    "top": table.bbox[1],
                    "x0": table.bbox[0],
                    "x1": table.bbox[2],
                    "bottom": table.bbox[3],
                },
            })

        words = page.extract_words(x_tolerance=1.5)  # иначе слова слипаются в плотном наборе
        full_text = ""
        prev_level = -1
        prev_word = ""
        placed_tables = set()
        table_queue = []
        place_table = False
        for i, word in enumerate(words):
            word_bbox = {"top": word["top"], "x0": word["x0"], "x1": word["x1"], "bottom": word["bottom"]}
            in_table = self.__find_in_tables(word_bbox, page_tables)
            if in_table is None:
                if i > 0:
                    full_text += prev_word
                    full_text += " " if word_bbox["bottom"] == prev_level else "\n"
                    if place_table:
                        full_text += "".join(table_queue)
                        table_queue = []
                        place_table = False
                prev_level = word_bbox["bottom"]
                prev_word = word["text"]
            elif in_table not in placed_tables:
                table = page_tables[in_table]
                table_queue.append("\n" + table["text"] + "\n")
                place_table = True
                placed_tables.add(in_table)
        full_text += prev_word
        if place_table:
            full_text += "".join(table_queue)
        return full_text

    @staticmethod
    def __touch_bbox(word_bbox, bbox):
        return (word_bbox["bottom"] >= bbox["top"] and word_bbox["top"] <= bbox["bottom"]
                and word_bbox["x1"] >= bbox["x0"] and word_bbox["x0"] <= bbox["x1"])

    def __find_in_tables(self, word_bbox, tables):
        for table in tables:
            if self.__touch_bbox(word_bbox, table["bbox"]):
                return table["id"]
        return None

    @staticmethod
    def __table_to_md(table):
        df = pd.DataFrame(table[1:], columns=table[0])
        return tabulate(df, headers="firstrow", tablefmt="pipe")  # pipe, не plain — чтобы в .md была настоящая таблица

    @staticmethod
    def __strip_duplicate_title(body: str, title: str) -> str:
        """arXiv-статьи часто сами начинаются с названия статьи обычным текстом — убирает этот
        повтор перед тем, как добавить название как настоящий заголовок `#`."""
        pattern = r"\s*".join(re.escape(w) for w in title.split())
        m = re.match(pattern, body, flags=re.I)
        return body[m.end():].lstrip("\n ") if m else body

    def parse_file(self, path: Path, raw_dir: Path) -> Document:
        pages = self.parse_pdf_pages(path.read_bytes())
        meta = path.with_suffix(".json")  # название и дата публикации, если файл скачан download_corpus.py
        if pages and meta.exists():
            info = json.loads(meta.read_text(encoding="utf-8"))
            body = self.__strip_duplicate_title(pages[0].markdown, info["title"])
            pages[0] = Page(pages[0].number, f"# {info['title']}\n\n" + body)
            date = info.get("published", "")[:10]
        else:
            date = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
        return Document(source=path.relative_to(raw_dir).as_posix(), date=date,
                        pages=[p for p in pages if p.markdown.strip()])

    @staticmethod
    def list_files(raw_dir: Path) -> list[Path]:
        return sorted(raw_dir.rglob("*.pdf"))

    def parse_corpus(self, raw_dir: Path) -> list[Document]:
        docs = []
        for p in self.list_files(raw_dir):
            try:
                doc = self.parse_file(p, raw_dir)
            except Exception as e:  # битый файл не должен останавливать всю сборку
                print(f"Пропущен {p.name}: {type(e).__name__}: {e}")
                continue
            if doc.pages:
                docs.append(doc)
            else:
                print(f"Пропущен {p.name}: текст не найден (возможно, скан без текстового слоя)")
        return docs


def to_markdown(doc: Document) -> str:
    """Один Markdown-файл на документ: метаданные в шапке, номер страницы перед каждой страницей."""
    head = f"---\nsource: {doc.source}\ndate: {doc.date}\npages: {len(doc.pages)}\n---\n\n"
    parts = [f"<!-- page: {p.number} -->\n\n{p.markdown}" for p in doc.pages]
    return head + "\n\n".join(parts) + "\n"


def write_markdown(docs: list[Document], out_dir: Path) -> None:
    for doc in docs:
        target = out_dir / (Path(doc.source).with_suffix(".md"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(to_markdown(doc), encoding="utf-8")
