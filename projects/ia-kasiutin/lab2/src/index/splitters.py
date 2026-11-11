from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from config import Config

HEADER_KEYS = ["h1", "h2", "h3", "h4", "h5", "h6"]
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


@lru_cache(maxsize=4)
def load_tokenizer(model: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer


def token_counter(model: str, add_special_tokens: bool = False) -> Callable[[str], int]:
    tokenizer = load_tokenizer(model)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))

    return count


def embedder_token_counter(config: Config) -> Callable[[str], int]:
    """Считает точное число токенов на входе SentenceTransformer (префикс + текст + спецтокены)."""
    count_with_special = token_counter(config.embedding.model, add_special_tokens=True)
    prefix = config.embedding.document_prefix
    return lambda text: count_with_special(prefix + text)


def _body_splitter(
    chunk_size: int, chunk_overlap: int, length: Callable[[str], int]
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length,
        separators=SEPARATORS,
    )


def _update_headers(active_headers: dict[str, str], new_headers: dict[str, str]) -> None:
    if not new_headers:
        return
    top_idx = min(HEADER_KEYS.index(k) for k in new_headers)
    for k in HEADER_KEYS[top_idx:]:
        active_headers.pop(k, None)
    active_headers.update(new_headers)


def _header_sections(
    page: Document, levels: int, active_headers: dict[str, str]
) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#" * (i + 1), HEADER_KEYS[i]) for i in range(levels)],
        strip_headers=True,
    )
    sections: list[Document] = []
    for sec in splitter.split_text(page.page_content):
        if not sec.page_content.strip():
            continue
        _update_headers(
            active_headers, {k: sec.metadata[k] for k in HEADER_KEYS if k in sec.metadata}
        )
        sections.append(
            Document(page_content=sec.page_content.strip(), metadata={**page.metadata, **active_headers})
        )

    if not sections and page.page_content.strip():
        for line in page.page_content.splitlines():
            match = re.match(r"^(#{1,6})\s+(.*)", line.strip())
            if match and len(match.group(1)) <= levels and match.group(2).strip():
                _update_headers(active_headers, {HEADER_KEYS[len(match.group(1)) - 1]: match.group(2).strip()})
        cleaned = re.sub(r"(?m)^#{1,6}\s+", "", page.page_content).strip()
        if cleaned:
            return [Document(page_content=cleaned, metadata={**page.metadata, **active_headers})]
    return sections


def _heading_path(metadata: dict) -> list[str]:
    path = [metadata[key] for key in HEADER_KEYS if metadata.get(key)]
    if not path and metadata.get("section"):
        path = [metadata["section"]]
    return path


def _split_body(
    sec_body: str,
    heading_prefix: str,
    max_chunk_tokens: int,
    max_seq_length: int,
    chunk_overlap: int,
    length: Callable[[str], int],
    count_embed: Callable[[str], int],
) -> list[str]:
    full = f"{heading_prefix}{sec_body}"
    if length(full) <= max_chunk_tokens and count_embed(full) <= max_seq_length:
        return [sec_body]

    prefix_tokens = length(heading_prefix) if heading_prefix else 0
    budget = max(1, max_chunk_tokens - prefix_tokens)
    overlap = min(chunk_overlap, budget - 1)
    raw_pieces = [p.strip() for p in _body_splitter(budget, overlap, length).split_text(sec_body) if p.strip()]

    pieces: list[str] = []
    for piece in raw_pieces:
        text = f"{heading_prefix}{piece}"
        over = max(length(text) - max_chunk_tokens, count_embed(text) - max_seq_length)
        if over > 0 and budget > over:
            sub_budget = max(1, budget - over)
            sub_overlap = min(chunk_overlap, sub_budget - 1)
            pieces.extend(
                p.strip() for p in _body_splitter(sub_budget, sub_overlap, length).split_text(piece) if p.strip()
            )
        else:
            pieces.append(piece)
    return pieces


def split_documents(
    documents: list[Document],
    config: Config,
) -> list[Document]:
    s = config.splitter
    emb = config.embedding
    length = token_counter(emb.model, add_special_tokens=False)
    count_embed = embedder_token_counter(config)

    embed_overhead = count_embed("")
    max_chunk_tokens = min(s.chunk_size, emb.max_seq_length - embed_overhead)
    if max_chunk_tokens < 1:
        raise ValueError(
            f"Недостаточно места в max_seq_length={emb.max_seq_length} после вычета префикса ({embed_overhead})"
        )

    sections: list[Document] = []
    active_headers: dict[str, str] = {}
    current_source: str | None = None

    for page in documents:
        source = page.metadata.get("source")
        if source != current_source:
            active_headers.clear()
            current_source = source
        sections.extend(_header_sections(page, s.header_levels, active_headers))

    chunks: list[Document] = []
    for sec in sections:
        sec_body = sec.page_content.strip()
        if not sec_body:
            continue

        base_meta = dict(sec.metadata)
        path = _heading_path(base_meta)
        heading_prefix = ""
        if path:
            base_meta["section"] = " > ".join(path)
            heading_prefix = f"{base_meta['section']}\n\n"

        pieces = _split_body(
            sec_body=sec_body,
            heading_prefix=heading_prefix,
            max_chunk_tokens=max_chunk_tokens,
            max_seq_length=emb.max_seq_length,
            chunk_overlap=s.chunk_overlap,
            length=length,
            count_embed=count_embed,
        )

        for piece_body in pieces:
            text = f"{heading_prefix}{piece_body}"
            metadata = dict(base_meta)
            metadata["chunk_index"] = len(chunks)
            metadata["n_tokens"] = length(text)
            metadata["embed_tokens"] = count_embed(text)
            key = f"{metadata.get('source')}|{metadata.get('page')}|{len(chunks)}|{text}"
            metadata["chunk_id"] = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            chunks.append(Document(page_content=text, metadata=metadata))
    return chunks
