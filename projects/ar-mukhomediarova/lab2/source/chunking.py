"""Шаг 3а. Разбиение нормализованного Markdown на чанки библиотечными сплиттерами LangChain
(пакет langchain-text-splitters).

Сплиттеры:
  recursive — RecursiveCharacterTextSplitter по всему документу (абзац → строка → предложение
              → слово), без учёта разделов; chunk_size/overlap в токенах bge-m3;
  markdown  — MarkdownHeaderTextSplitter по заголовкам h1–h{header_levels}: раздел = чанк;
              раздел длиннее chunk_size дорезается RecursiveCharacterTextSplitter без перекрытия;
  hybrid    — MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter с перекрытием overlap
              (заголовки + окно/overlap).
Для кода из репозитория (.py) вместо обычного рекурсивного сплиттера используется
RecursiveCharacterTextSplitter.from_language(Language.PYTHON).

Собственная логика здесь только служебная: подсчёт токенов, номера страниц по меткам
<!-- page: N --> и строка «Документ > Раздел» для эмбеддинга.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_text_splitters import Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import settings

PAGE_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
FENCE_RE = re.compile(r"^```", re.M)


# ------------------------------------------------------------ токенизатор
@functools.lru_cache(maxsize=4)
def _load_tokenizer(name: str | None):
    if not name:
        return None
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_pretrained(name)
    except Exception as e:  # нет сети / пакета — грубая оценка
        print(f"  [warn] токенизатор {name} недоступен ({e.__class__.__name__}), считаю токены по словам")
        return None


class TokenCounter:
    """length_function для сплиттеров LangChain: длина в токенах bge-m3."""

    def __init__(self, name: str | None):
        self.tok = _load_tokenizer(name)
        self.exact = self.tok is not None

    @functools.lru_cache(maxsize=200_000)
    def __call__(self, text: str) -> int:
        if not text:
            return 0
        if self.tok is not None:
            return len(self.tok.encode(text, add_special_tokens=False).ids)
        # bge-m3 (XLM-R): в среднем ~1.7 токена на русское слово, плюс пунктуация
        return int(len(re.findall(r"\w+", text)) * 1.7 + len(re.findall(r"[^\w\s]", text)) * 0.5) + 1

    def __hash__(self):
        return id(self)


# ------------------------------------------------------------ чтение Markdown
@dataclass
class Doc:
    meta: dict
    body: str


def read_markdown(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    if raw.startswith("---\n"):
        end = raw.index("\n---", 4)
        meta = yaml.safe_load(raw[4:end]) or {}
        body = raw[end + 4:]
    return Doc(meta, body.strip() + "\n")


def load_corpus(cfg: dict) -> list[Doc]:
    md_dir = settings.lab_path(cfg, "markdown_dir")
    return [read_markdown(p) for p in sorted(md_dir.glob("*.md"))]


# ------------------------------------------------------------ чанк
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    embed_text: str
    heading_path: list[str]
    sections: list[str]
    page_start: int | None
    page_end: int | None
    n_tokens: int
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.update(d.pop("meta"))
        return d


def _same_title(heading: str, title: str) -> bool:
    """h1 документа повторяет его название («REALM: …*» ≈ «REALM: …», «src/x.py» ≈ «repo/src/x.py»)."""
    a, b = (re.sub(r"[^\w]+", "", s.lower()) for s in (heading, title))
    return bool(a) and (a in b or b in a)


def make_head(title: str, path) -> str:
    """Строка контекста для эмбеддинга: «Документ > Раздел > Подраздел» (ограниченной длины)."""
    path = list(path)
    if path and _same_title(path[0], str(title)):
        path = path[1:]
    t = str(title)
    t = t if len(t) <= 70 else t[:70].rsplit(" ", 1)[0] + "…"
    parts = [p if len(p) <= 80 else p[:80].rsplit(" ", 1)[0] + "…" for p in path[-2:]]
    return " > ".join([t] + parts)


def _clean_heading(h: str) -> str:
    return h.replace("\\", "").strip()


def _strip_pages(text: str) -> str:
    text = PAGE_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)  # «  \n» — markdown-переносы от MarkdownHeaderTextSplitter
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _pages_of(text: str, start_page: int | None) -> tuple[int | None, int | None, int | None]:
    """Страницы фрагмента: страница в начале + метки внутри; третье значение — страница в конце."""
    marks = [(m.start(), int(m.group(1))) for m in PAGE_RE.finditer(text)]
    first_text = re.search(r"\S", PAGE_RE.sub(lambda m: " " * len(m.group(0)), text))
    pages = []
    if start_page is not None and (not marks or (first_text and first_text.start() < marks[0][0])):
        pages.append(start_page)
    pages += [p for _, p in marks]
    end_page = marks[-1][1] if marks else start_page
    return (min(pages) if pages else None), (max(pages) if pages else None), end_page


def _page_at(text: str, pos: int, start_page: int | None) -> int | None:
    marks = [int(m.group(1)) for m in PAGE_RE.finditer(text[:pos])]
    return marks[-1] if marks else start_page


def _heading_paths(body: str) -> list[tuple[int, tuple[str, ...]]]:
    """Позиции заголовков вне блоков кода и путь заголовков после каждого (для recursive)."""
    fences = [m.start() for m in FENCE_RE.finditer(body)]
    in_code = lambda pos: sum(1 for f in fences if f < pos) % 2 == 1
    stack: list[tuple[int, str]] = []
    out = []
    for m in HEADING_RE.finditer(body):
        if in_code(m.start()):
            continue
        level = len(m.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, _clean_heading(m.group(2))))
        out.append((m.start(), tuple(t for _, t in stack)))
    return out


def _locate(text: str, piece: str, cursor: int) -> int:
    """Позиция фрагмента в исходном тексте. start_index из LangChain бывает −1: при склейке
    частей сплиттер может заменить разделитель, и фрагмент перестаёт быть точной подстрокой."""
    for probe in (piece, piece[:120], piece[:40]):
        pos = text.find(probe, cursor)
        if pos >= 0:
            return pos
    words = re.findall(r"\S+", piece)[:8]
    if words:
        m = re.compile(r"\s+".join(map(re.escape, words))).search(text, cursor)
        if m:
            return m.start()
    return cursor


def _split(splitter: RecursiveCharacterTextSplitter, text: str) -> list[tuple[int, str]]:
    out, cursor = [], 0
    for piece in splitter.split_text(text):
        pos = _locate(text, piece, cursor)
        out.append((pos, piece))
        cursor = pos + 1
    return out


# ------------------------------------------------------------ сплиттеры
def _recursive(count: TokenCounter, size: int, overlap: int, fmt: str | None) -> RecursiveCharacterTextSplitter:
    kw = dict(chunk_size=size, chunk_overlap=overlap, length_function=count)
    if fmt == "py":
        return RecursiveCharacterTextSplitter.from_language(Language.PYTHON, **kw)
    # абзац → строка (пункт списка, строка таблицы) → предложение → слово
    return RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", "; ", " ", ""], **kw)


def chunk_document(doc: Doc, cfg: dict, count: TokenCounter) -> list[Chunk]:
    c = cfg["chunking"]
    size, overlap = int(c["chunk_size"]), int(c["overlap"])
    splitter, include_headers = c["splitter"], bool(c["include_headers"])
    levels = int(c.get("header_levels", 3))
    title = str(doc.meta.get("title", doc.meta.get("doc_id")))
    fmt = doc.meta.get("format")

    # запас токенов под строку «Документ > Раздел» (самая длинная в документе)
    header_budget = 0
    if include_headers:
        paths = [p for _, p in _heading_paths(doc.body)] or [()]
        header_budget = max(count(make_head(title, p)) for p in paths) + 2
    limit = max(32, size - header_budget)

    pieces: list[dict] = []  # {text, path, sections, page_start, page_end}
    if splitter == "recursive":
        heads = _heading_paths(doc.body)
        for start, content in _split(_recursive(count, limit, overlap, fmt), doc.body):
            end = start + len(content)
            path = next((p for pos, p in reversed(heads) if pos <= start), ())
            inside = [p for pos, p in heads if start < pos < end]
            ps, pe, _ = _pages_of(content, _page_at(doc.body, start, None))
            pieces.append({"text": content, "path": path,
                           "sections": [path] + inside, "page_start": ps, "page_end": pe})
    elif splitter in ("markdown", "hybrid"):
        md = MarkdownHeaderTextSplitter([("#" * i, f"h{i}") for i in range(1, levels + 1)],
                                        strip_headers=True)
        inner = _recursive(count, limit, overlap if splitter == "hybrid" else 0, fmt)
        page = None
        for sec in md.split_text(doc.body):
            path = tuple(_clean_heading(sec.metadata[f"h{i}"]) for i in range(1, levels + 1)
                         if f"h{i}" in sec.metadata)
            text = sec.page_content
            sec_start_page = page
            _, _, page = _pages_of(text, page)  # страница на конце раздела — для следующего
            if not _strip_pages(text):
                continue
            for start, content in _split(inner, text):
                ps, pe, _ = _pages_of(content, _page_at(text, start, sec_start_page))
                pieces.append({"text": content, "path": path, "sections": [path],
                               "page_start": ps, "page_end": pe})
    else:
        raise ValueError(f"неизвестный сплиттер: {splitter}")

    chunks = []
    for p in pieces:
        body = _strip_pages(p["text"])
        if not body:
            continue
        path = list(p["path"])
        embed_text = f"{make_head(title, path)}\n\n{body}" if include_headers else body
        chunks.append(Chunk(
            chunk_id=f"{doc.meta['doc_id']}::{len(chunks):04d}",
            doc_id=doc.meta["doc_id"],
            title=title,
            text=body,
            embed_text=embed_text,
            heading_path=path,
            sections=list(dict.fromkeys(" > ".join(s) for s in p["sections"] if s)) or [""],
            page_start=p["page_start"],
            page_end=p["page_end"],
            n_tokens=count(embed_text),
            meta={k: doc.meta.get(k) for k in
                  ("source_type", "format", "path", "url", "date", "semester", "page_mode")},
        ))
    return chunks


def chunk_corpus(cfg: dict) -> tuple[list[Chunk], dict]:
    count = TokenCounter(cfg["chunking"].get("tokenizer"))
    docs = load_corpus(cfg)
    chunks = [ch for d in docs for ch in chunk_document(d, cfg, count)]
    toks = sorted(ch.n_tokens for ch in chunks) or [0]
    stats = {
        "documents": len(docs),
        "chunks": len(chunks),
        "tokens_total": sum(toks),
        "tokens_mean": round(sum(toks) / len(toks), 1),
        "tokens_median": toks[len(toks) // 2],
        "tokens_max": toks[-1],
        "tokenizer_exact": count.exact,
    }
    return chunks, stats
