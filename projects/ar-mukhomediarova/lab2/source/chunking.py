"""Шаг 3а. Разбиение нормализованного Markdown на чанки.

Сплиттеры:
  recursive — без учёта структуры: документ режется на окна chunk_size с перекрытием overlap,
              границы ищутся по абзацам → предложениям → словам;
  markdown  — по разделам h1–h6: каждый раздел — отдельный чанк, длинные режутся на окна;
  hybrid    — как markdown, но мелкие соседние разделы склеиваются (до min_chunk_tokens),
              а длинные режутся окнами с перекрытием (заголовки + окно/overlap).
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

import settings

PAGE_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SENT_RE = re.compile(r"(?<=[.!?…;])\s+(?=[«\"(\[A-ZА-ЯЁ0-9])")


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


# ------------------------------------------------------------ разбор Markdown
@dataclass
class Block:
    kind: str                 # para | list | table | code
    text: str
    path: tuple[str, ...]     # цепочка заголовков
    page: int | None
    header: str = ""          # шапка таблицы (повторяется при разрезании)


@dataclass
class Doc:
    meta: dict
    blocks: list[Block]


def read_markdown(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    if raw.startswith("---\n"):
        end = raw.index("\n---", 4)
        meta = yaml.safe_load(raw[4:end]) or {}
        body = raw[end + 4:]
    lines = body.splitlines()
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []
    page = None
    buf: list[str] = []
    buf_kind = None

    def path_now():
        return tuple(t for _, t in stack)

    def flush():
        nonlocal buf, buf_kind
        if buf and "".join(buf).strip():
            text = "\n".join(buf).strip()
            header = ""
            if buf_kind == "table":
                rows = text.splitlines()
                header = "\n".join(rows[:2]) if len(rows) > 2 and set(rows[1]) <= set("|-: ") else ""
            blocks.append(Block(buf_kind or "para", text, path_now(), page, header))
        buf, buf_kind = [], None

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):  # блок кода целиком, «#» внутри — не заголовки
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            blocks.append(Block("code", "\n".join(lines[i:j + 1]), path_now(), page))
            i = j + 1
            continue
        m = PAGE_RE.fullmatch(line.strip())
        if m:
            flush()
            page = int(m.group(1))
            i += 1
            continue
        h = HEADING_RE.match(line)
        if h and h.group(2).strip():
            flush()
            level = len(h.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, h.group(2).strip().replace("\\", "")))
            i += 1
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        kind = "table" if line.lstrip().startswith("|") else (
            "list" if re.match(r"^\s*([-*+]|\d+[.)])\s+", line) else "para")
        if buf_kind and kind != buf_kind and not (buf_kind == "list" and kind == "para"):
            flush()
        buf_kind = buf_kind or kind
        buf.append(line)
        i += 1
    flush()
    return Doc(meta, blocks)


def load_corpus(cfg: dict) -> list[Doc]:
    md_dir = settings.lab_path(cfg, "markdown_dir")
    return [read_markdown(p) for p in sorted(md_dir.glob("*.md"))]


# ------------------------------------------------------------ единицы и окна
@dataclass
class Unit:
    text: str
    tokens: int
    block: int                # индекс блока (для склейки через \n\n)
    path: tuple[str, ...]
    page: int | None
    header: str = ""          # шапка таблицы
    sep: str = " "            # разделитель внутри блока
    header_tokens: int = 0    # стоимость шапки, если окно начинается с середины таблицы


def _hard_split(text: str, limit: int, count) -> list[str]:
    words, parts, cur = text.split(" "), [], []
    for w in words:
        cur.append(w)
        if count(" ".join(cur)) > limit and len(cur) > 1:
            cur.pop()
            parts.append(" ".join(cur))
            cur = [w]
    if cur:
        parts.append(" ".join(cur))
    return parts


def block_units(block: Block, idx: int, limit: int, count) -> list[Unit]:
    """Разбить блок на атомарные единицы не длиннее limit токенов."""
    if block.kind == "table":
        rows = block.text.splitlines()
        body = rows[2:] if block.header else rows
        pieces, sep = body, "\n"
    elif block.kind in ("code", "list"):
        pieces, sep = block.text.splitlines(), "\n"
        if block.kind == "code" and count(block.text) <= limit:
            pieces = [block.text]
    else:
        pieces = SENT_RE.split(block.text) if count(block.text) > limit else [block.text]
        sep = " "
    if block.header:  # строке таблицы нужен запас под повторяемую шапку
        limit = max(16, limit - count(block.header) - 4)
    units = []
    for piece in pieces:
        if not piece.strip():
            continue
        parts = [piece] if count(piece) <= limit else _hard_split(piece, limit, count)
        for p in parts:
            units.append(Unit(p, count(p), idx, block.path, block.page, block.header, sep,
                              count(block.header) if block.header else 0))
    if block.kind == "code" and len(units) > 1:  # каждой части возвращаем ограду ```
        fence = block.text.splitlines()[0]
        for u in units:
            if not u.text.startswith("```"):
                u.text = fence + "\n" + u.text
            if not u.text.rstrip().endswith("```"):
                u.text = u.text + "\n```"
            u.tokens = count(u.text)
            u.sep = "\n\n"
    return units


def render(units: list[Unit]) -> str:
    out, prev_block, headers_done = [], None, set()
    for u in units:
        if u.header and (u.block, u.header) not in headers_done:
            if prev_block is not None:
                out.append("\n\n")
            out.append(u.header + "\n")
            headers_done.add((u.block, u.header))
            out.append(u.text)
        else:
            if prev_block is not None:
                out.append(u.sep if u.block == prev_block else "\n\n")
            out.append(u.text)
        prev_block = u.block
    return "".join(out).strip()


def pack(units: list[Unit], size: int, overlap: int) -> list[list[Unit]]:
    """Жадно собрать единицы в окна ≤ size с перекрытием ≥ overlap токенов.
    Шапка таблицы учитывается один раз на окно (render повторяет её при разрезании таблицы)."""

    def cost(window: list[Unit]) -> int:
        heads = {(u.block, u.header): u.header_tokens for u in window if u.header}
        return sum(u.tokens for u in window) + sum(heads.values()) + 2 * len(window)

    windows, cur = [], []
    for u in units:
        if cur and cost(cur + [u]) > size:
            windows.append(cur)
            tail = []
            if overlap > 0:
                for prev in reversed(cur):
                    if sum(t.tokens for t in tail) >= overlap or cost([prev] + tail) > size // 2:
                        break
                    tail.insert(0, prev)
            cur = tail
            while cur and cost(cur + [u]) > size:  # перекрытие не должно раздувать окно
                cur.pop(0)
        cur.append(u)
    if cur:
        windows.append(cur)
    return windows


# ------------------------------------------------------------ чанкер
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


def _sections_of(units: list[Unit]) -> list[str]:
    seen = []
    for u in units:
        s = " > ".join(u.path)
        if s not in seen:
            seen.append(s)
    return seen


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


def chunk_document(doc: Doc, cfg: dict, count: TokenCounter) -> list[Chunk]:
    c = cfg["chunking"]
    size, overlap = int(c["chunk_size"]), int(c["overlap"])
    splitter, include_headers = c["splitter"], bool(c["include_headers"])
    title = doc.meta.get("title", doc.meta.get("doc_id"))
    # запас токенов под строку «Документ > Раздел» (берём самую длинную в документе)
    header_budget = 0
    if include_headers:
        paths = {b.path for b in doc.blocks} or {()}
        header_budget = max(count(make_head(title, p)) for p in paths) + 2
    limit = max(32, size - header_budget)

    units_by_block = [block_units(b, i, limit, count) for i, b in enumerate(doc.blocks)]

    # группы, внутри которых режем окнами
    if splitter == "recursive":
        groups = [[u for us in units_by_block for u in us]]
    else:
        groups, cur, cur_path = [], [], None
        for b, us in zip(doc.blocks, units_by_block):
            if us and b.path != cur_path and cur:
                groups.append(cur)
                cur = []
            cur_path = b.path
            cur += us
        if cur:
            groups.append(cur)
        if splitter == "hybrid":
            # склеиваем мелкие соседние разделы одного верхнего раздела (h1)
            merged, min_tok = [], int(c.get("min_chunk_tokens", 0))
            for g in groups:
                g_tok = sum(u.tokens for u in g)
                if merged:
                    prev = merged[-1]
                    p_tok = sum(u.tokens for u in prev)
                    same_top = prev[0].path[:1] == g[0].path[:1]
                    if same_top and (p_tok < min_tok or g_tok < min_tok) and p_tok + g_tok <= limit:
                        prev.extend(g)
                        continue
                merged.append(list(g))
            groups = merged
        elif splitter != "markdown":
            raise ValueError(f"неизвестный сплиттер: {splitter}")

    chunks = []
    for g in groups:
        for window in pack(g, limit, overlap):
            body = render(window)
            if not body:
                continue
            path = list(window[0].path)
            head = make_head(title, path)
            embed_text = f"{head}\n\n{body}" if include_headers else body
            pages = [u.page for u in window if u.page is not None]
            chunks.append(Chunk(
                chunk_id=f"{doc.meta['doc_id']}::{len(chunks):04d}",
                doc_id=doc.meta["doc_id"],
                title=title,
                text=body,
                embed_text=embed_text,
                heading_path=path,
                sections=_sections_of(window),
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
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
