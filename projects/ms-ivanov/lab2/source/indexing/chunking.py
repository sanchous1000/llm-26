"""Document body -> list of :class:`Chunk` with metadata."""

import bisect
import re
import uuid
from dataclasses import dataclass

from source.evaluation.documents import references_span
from source.indexing import blocks as b
from source.indexing.config import ChunkingConfig
from source.indexing.splitters import build_splitter
from source.indexing.tokens import TokenCounter

_NAMESPACE = uuid.UUID("6f3c1b1e-5d0a-4a52-9c35-2f6f0d3b7a10")
_PAGE_LINE_RE = re.compile(r"^<!-- page: \d+ -->[ \t]*\n?", re.MULTILINE)
_IMAGE_LINE_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)[ \t]*\n?", re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_MIN_CONTENT_CHARS = 20


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice ``body[start:end]`` of one document."""

    chunk_id: str
    doc_key: str
    index: int
    start: int
    end: int
    text: str
    token_count: int
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    embed_text: str


def chunk_document(
    doc_key: str,
    title: str,
    body: str,
    config: ChunkingConfig,
    counter: TokenCounter,
) -> list[Chunk]:
    """Split ``body`` according to ``config``; offsets refer to ``body``."""
    spans = _split_segments(body, config, counter)
    headings = [x for x in b.parse_blocks(body) if x.kind == b.HEADING]
    pages = [x for x in b.parse_blocks(body) if x.kind == b.PAGE]
    digest = config.digest()
    chunks: list[Chunk] = []
    for start, end in spans:
        text = body[start:end]
        # Leading headings are already part of the section path, so with
        # headings enabled they are moved out of the embedded body.
        content = _leading_headings_end(text)
        lead = content if config.include_headings else 0
        embed_body = _clean(text[lead:], config)
        if len(embed_body.strip()) < _MIN_CONTENT_CHARS:
            continue
        path = _section_path(body, headings, start + lead)
        chunks.append(
            Chunk(
                chunk_id=str(
                    uuid.uuid5(_NAMESPACE, f"{doc_key}|{digest}|{start}|{end}")
                ),
                doc_key=doc_key,
                index=len(chunks),
                start=start,
                end=end,
                text=text,
                token_count=counter.count(text),
                section_path=path,
                page_start=_page_at(pages, start + content) or _page_at(pages, start),
                page_end=_page_at(pages, end),
                embed_text=_embed_text(title, path, embed_body, config),
            )
        )
    return chunks


def _leading_headings_end(text: str) -> int:
    """Offset of the first block that is not a heading or a page marker."""
    for block in b.parse_blocks(text):
        if block.kind not in (b.HEADING, b.PAGE):
            return block.start
    return len(text)


def _split_segments(
    body: str, config: ChunkingConfig, counter: TokenCounter
) -> list[tuple[int, int]]:
    """Run the splitter on the indexable parts of the body (bibliography excluded)."""
    splitter = build_splitter(config, counter)
    segments = [(0, len(body))]
    refs = None if config.include_references else references_span(body)
    if refs is not None:
        segments = [s for s in ((0, refs[0]), (refs[1], len(body))) if s[1] > s[0]]
    spans: list[tuple[int, int]] = []
    for seg_start, seg_end in segments:
        text = body[seg_start:seg_end]
        spans.extend((seg_start + s, seg_start + e) for s, e in splitter.split(text))
    return sorted(spans)


def _section_path(body: str, headings: list[b.Block], position: int) -> tuple[str, ...]:
    """Headings (below the title) that enclose ``position``."""
    stack: dict[int, str] = {}
    for heading in headings:
        if heading.start > position:
            break
        stack = {level: t for level, t in stack.items() if level < heading.level}
        stack[heading.level] = heading.title
    return tuple(stack[level] for level in sorted(stack) if level >= 2)


def _page_at(pages: list[b.Block], position: int) -> int | None:
    starts = [p.start for p in pages]
    index = bisect.bisect_right(starts, position) - 1
    return pages[index].page if index >= 0 else None


def clean_chunk_text(text: str, *, strip_images: bool = True) -> str:
    """Remove page markers (and image links) and collapse blank-line runs."""
    cleaned = _PAGE_LINE_RE.sub("", text)
    if strip_images:
        cleaned = _IMAGE_LINE_RE.sub("", cleaned)
    return _BLANK_RUN_RE.sub("\n\n", cleaned).strip()


def _clean(text: str, config: ChunkingConfig) -> str:
    return clean_chunk_text(text, strip_images=config.strip_images)


def _embed_text(
    title: str, path: tuple[str, ...], body: str, config: ChunkingConfig
) -> str:
    if not config.include_headings:
        return body
    header = "\n".join(part for part in (title, " > ".join(path)) if part)
    return f"{header}\n\n{body}" if header else body
