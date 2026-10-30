"""Plain-text rendering of answers and passages for the terminal."""

import re
from collections.abc import Sequence

from source.rag.models import Answer, RetrievedChunk

_CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
_LABELS = {
    "ru": {
        "sources": "Источники",
        "page": "стр.",
        "unverified": "(фрагмент подобран)",
        "none": "Источники в ответе не указаны.",
        "unknown": "Модель сослалась на несуществующие номера",
    },
    "en": {
        "sources": "Sources",
        "page": "p.",
        "unverified": "(snippet selected)",
        "none": "The answer cites no sources.",
        "unknown": "The model cited non-existent numbers",
    },
}


def language_of(text: str) -> str:
    return "ru" if _CYRILLIC_RE.search(text) else "en"


def format_answer(answer: Answer, *, show_timings: bool = True) -> str:
    """Answer text followed by the numbered sources with snippets."""
    labels = _LABELS[language_of(answer.question)]
    lines = [answer.text, ""]
    if answer.citations:
        lines.append(f"{labels['sources']}:")
        for citation in answer.citations:
            lines += _citation_lines(citation, labels)
    else:
        lines.append(labels["none"])
    if answer.unknown_citations:
        numbers = ", ".join(f"[{n}]" for n in answer.unknown_citations)
        lines.append(f"! {labels['unknown']}: {numbers}")
    if show_timings and answer.timings:
        lines.append("")
        lines.append(_timings(answer))
    return "\n".join(lines)


def _citation_lines(citation, labels: dict) -> list[str]:
    chunk = citation.chunk
    place = [chunk.section] if chunk.section else []
    if chunk.page:
        place.append(f"{labels['page']} {chunk.page}")
    head = f"[{citation.number}] {chunk.title}"
    if chunk.short_citation:
        head += f" ({chunk.short_citation})"
    if place:
        head += " — " + ", ".join(place)
    lines = [head]
    note = "" if citation.verified else f" {labels['unverified']}"
    lines.append(f"    «{citation.snippet}»{note}")
    if chunk.source_url:
        lines.append(f"    {chunk.source_url}")
    return lines


def _timings(answer: Answer) -> str:
    t = answer.timings
    parts = [f"search {t.get('retrieve_s', 0):.1f}s ({answer.mode})"]
    if "llm_s" in t:
        parts.append(f"LLM {t['llm_s']:.1f}s")
    if answer.model:
        parts.append(answer.model)
    return "[" + ", ".join(parts) + "]"


def format_chunks(chunks: Sequence[RetrievedChunk], *, width: int = 600) -> str:
    """Retrieval-only output: ranked passages with metadata."""
    blocks = []
    for chunk in chunks:
        meta = " | ".join(
            part
            for part in (
                f"{chunk.title} ({chunk.short_citation})",
                chunk.section,
                f"p.{chunk.page}" if chunk.page else "",
                f"score {chunk.score:.3f}",
                chunk.doc_key,
            )
            if part
        )
        text = " ".join(chunk.text.split())
        shown = text if len(text) <= width else text[:width].rstrip() + " […]"
        blocks.append(f"#{chunk.rank} {meta}\n    {shown}")
    return "\n\n".join(blocks) if blocks else "(nothing found)"
