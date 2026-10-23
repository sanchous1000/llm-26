"""Нарезка Markdown на чанки: сначала по заголовкам h1-h3, затем по токенам.

Токены считаются токенизатором модели эмбеддингов. Номера страниц пробрасываются через
служебные маркеры в тексте: после нарезки по ним определяются page_start/page_end чанка.
"""
import bisect
import re
from dataclasses import dataclass
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from doc_rag.config import SplitterConfig
from doc_rag.services.parser import PageMarkdown

# маркер страницы занимает отдельную строку: <START>номер<END>
_PAGE_START, _PAGE_END = "", ""
_MARKER_RE = re.compile(f"{_PAGE_START}(\\d+){_PAGE_END}\n?")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")

LONG_HEADING_CHARS = 60  # заголовок длиннее - считаем содержательным текстом
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


@dataclass
class Chunk:
    text: str  # текст для эмбеддинга (с путём заголовков, если включено)
    chunk_index: int
    page_start: int
    page_end: int
    h1: str | None = None
    h2: str | None = None
    h3: str | None = None


@dataclass
class _Section:
    text: str  # содержит маркеры страниц
    headers: dict[int, str]
    enter_page: int
    heading_only: bool = False  # тело секции - сам длинный «заголовок» (абзац, набранный жирным)


class SplitterService:
    def __init__(
        self,
        config: SplitterConfig,
        count_tokens: Callable[[str], int],
        max_seq_length: int | None = None,
    ):
        self.config = config
        self._count = count_tokens
        self.max_seq_length = max_seq_length

    # ------------------------------------------------------------------ публичный API

    def split(self, pages: list[PageMarkdown]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in self._split_by_headers(pages):
            chunks.extend(self._split_section(section, start_index=len(chunks)))

        if self.max_seq_length:
            too_long = sum(1 for c in chunks if self._count(c.text) > self.max_seq_length)
            if too_long:
                logger.warning(
                    f"{too_long} из {len(chunks)} чанков длиннее max_seq_length={self.max_seq_length} "
                    f"токенов модели - хвост будет обрезан при векторизации"
                )
        return chunks

    # ------------------------------------------------------------------ шаг 1: заголовки

    def _split_by_headers(self, pages: list[PageMarkdown]) -> list[_Section]:
        levels = set(self.config.headers_to_split_on)
        sections: list[_Section] = []
        headers: dict[int, str] = {}
        lines: list[str] = []
        enter_page = pages[0].page if pages else 1

        def flush():
            text = "\n".join(lines).strip("\n")
            if _MARKER_RE.sub("", text).strip():
                sections.append(_Section(text, dict(headers), enter_page))
            elif headers and len(headers[max(headers)]) >= LONG_HEADING_CHARS:
                # у секции нет текста, а «заголовок» длинный - это абзац, набранный жирным; без этого
                # он потерялся бы (в чанки попадает только тело секций)
                body = f"{_PAGE_START}{enter_page}{_PAGE_END}\n{headers[max(headers)]}"
                sections.append(_Section(body, dict(headers), enter_page, heading_only=True))
            lines.clear()

        for page in pages:
            lines.append(f"{_PAGE_START}{page.page}{_PAGE_END}")
            for line in page.markdown.split("\n"):
                m = _HEADING_RE.match(line)
                if m and len(m.group(1)) in levels:
                    flush()
                    level = len(m.group(1))
                    headers = {k: v for k, v in headers.items() if k < level}
                    headers[level] = m.group(2)
                    enter_page = page.page
                else:
                    lines.append(line)
        flush()
        return sections

    # ------------------------------------------------------------------ шаг 2: токены

    def _visible_tokens(self, text: str) -> int:
        return self._count(_MARKER_RE.sub("", text))

    def _header_prefix(self, headers: dict[int, str], skip_last: bool = False) -> str:
        if not self.config.include_headers_in_text or not headers:
            return ""
        levels = sorted(headers)[:-1] if skip_last else sorted(headers)  # последний уровень уже в тексте чанка
        return " > ".join(headers[k] for k in levels) + "\n\n" if levels else ""

    def _split_section(self, section: _Section, start_index: int) -> list[Chunk]:
        prefix = self._header_prefix(section.headers, skip_last=section.heading_only)
        chunk_size = self.config.chunk_size
        # путь заголовков входит в бюджет токенов (но не уменьшает его больше чем вдвое)
        budget = max(chunk_size - self._count(prefix), chunk_size // 2) if prefix else chunk_size
        overlap = min(self.config.chunk_overlap, budget // 2)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=overlap,
            length_function=self._visible_tokens,
            separators=SEPARATORS,
            keep_separator="end",
            add_start_index=True,
        )
        markers = list(_MARKER_RE.finditer(section.text))
        marker_pos = [m.start() for m in markers]
        marker_page = [int(m.group(1)) for m in markers]

        def page_at(pos: int) -> int:
            i = bisect.bisect_right(marker_pos, pos) - 1
            return marker_page[i] if i >= 0 else section.enter_page

        result = []
        for doc in splitter.create_documents([section.text]):
            raw = doc.page_content
            text = _MARKER_RE.sub("", raw).strip()
            if not text:
                continue

            start = doc.metadata["start_index"]
            result.append(
                Chunk(
                    text=prefix + text,
                    chunk_index=start_index + len(result),
                    page_start=page_at(start + self._first_visible(raw)),
                    page_end=page_at(start + self._last_visible(raw)),
                    h1=section.headers.get(1),
                    h2=section.headers.get(2),
                    h3=section.headers.get(3),
                )
            )
        return result

    @staticmethod
    def _first_visible(raw: str) -> int:
        """Индекс первого символа вне маркеров и пробелов."""
        pos = 0
        for m in _MARKER_RE.finditer(raw):
            if raw[pos:m.start()].strip():
                break
            pos = m.end()
        return pos + (len(raw[pos:]) - len(raw[pos:].lstrip()))

    @staticmethod
    def _last_visible(raw: str) -> int:
        """Индекс последнего символа вне маркеров и пробелов."""
        end = len(raw)
        for m in reversed(list(_MARKER_RE.finditer(raw))):
            if raw[m.end():end].strip():
                break
            end = m.start()
        return max(len(raw[:end].rstrip()) - 1, 0)
