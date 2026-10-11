"""Prompt assembly and parsing of the model's cited answer."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from source.evaluation.locate import locate
from source.indexing.chunking import clean_chunk_text
from source.rag.models import Citation, RetrievedChunk

INSTRUCTIONS = """\
You are a research assistant answering questions about a collection of scientific
papers (optimal transport, topological data analysis, generative models,
representation alignment).

Rules:
1. Use ONLY the numbered passages below. If they do not contain the answer, say so
   plainly, do not guess or use outside knowledge, cite nothing and omit SOURCES.
2. After every statement that relies on a passage, cite it as [n] (several: [1][3]).
   Never cite a number that is not listed.
3. Answer in the same language as the question. Keep technical terms, method names
   and formulas (LaTeX) in their original form.
4. Be precise and concise; define the symbols you use.
5. Do not use any tools.

Output format (the SOURCES line is mandatory and must be on its own line):
<the answer with [n] citations>

SOURCES:
[n] "<verbatim quote of at most 200 characters copied exactly from passage n that
supports the answer; do not translate it>"
(one line per cited passage, only passages you actually cited)
"""

_SOURCES_RE = re.compile(r"^\s*\**SOURCES:?\**\s*$", re.IGNORECASE | re.MULTILINE)
_CITATION_RE = re.compile(r"\[(\d{1,3})\]")
_QUOTE_LINE_RE = re.compile(r'^\s*\[(\d{1,3})\]\s*["“«](.+?)["”»]?\s*$')
_WORD_RE = re.compile(r"[\w]{3,}", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_QUOTE_CHARS = 25


@dataclass(frozen=True)
class PromptBundle:
    """The final prompt and the passages that made it into it (numbered 1..n)."""

    text: str
    contexts: tuple[RetrievedChunk, ...]


def build_prompt(
    question: str,
    contexts: Sequence[RetrievedChunk],
    *,
    history: Sequence[tuple[str, str]] = (),
    max_context_tokens: int,
) -> PromptBundle:
    """Instructions + conversation + as many passages as fit the token budget."""
    chosen: list[RetrievedChunk] = []
    used = 0
    for chunk in contexts:
        if chosen and used + chunk.token_count > max_context_tokens:
            break
        chosen.append(chunk)
        used += chunk.token_count
    passages = "\n\n".join(
        _passage(number, c) for number, c in enumerate(chosen, start=1)
    )
    parts = [INSTRUCTIONS, "Passages:\n\n" + passages]
    if history:
        parts.append("Conversation so far:\n" + _format_history(history))
    parts.append(f"Question: {question}\n\nAnswer:")
    return PromptBundle("\n\n".join(parts), tuple(chosen))


def _passage(number: int, chunk: RetrievedChunk) -> str:
    where = " | ".join(
        part
        for part in (
            (
                f"{chunk.title} ({chunk.short_citation})"
                if chunk.short_citation
                else chunk.title
            ),
            chunk.section,
            f"page {chunk.page}" if chunk.page else "",
        )
        if part
    )
    return f"[{number}] {where}\n{clean_chunk_text(chunk.text)}"


def _format_history(
    history: Sequence[tuple[str, str]], max_answer_chars: int = 1200
) -> str:
    lines = []
    for question, answer in history:
        shortened = (
            answer
            if len(answer) <= max_answer_chars
            else answer[:max_answer_chars] + "..."
        )
        lines.append(f"User: {question}\nAssistant: {shortened}")
    return "\n".join(lines)


def split_sources_block(text: str) -> tuple[str, str]:
    """Split the answer from its trailing quote block (second part may be empty).

    The block normally starts with a ``SOURCES:`` line; when the model forgets
    it, a tail made of ``[n] "quote"`` lines is recognised instead.
    """
    match = _SOURCES_RE.search(text)
    if match is not None:
        return text[: match.start()].strip(), text[match.end() :].strip()
    lines = text.rstrip().splitlines()
    first = len(lines)
    while first > 0 and (
        not lines[first - 1].strip() or _QUOTE_LINE_RE.match(lines[first - 1])
    ):
        first -= 1
    tail = lines[first:]
    if not any(_QUOTE_LINE_RE.match(line) for line in tail):
        return text.strip(), ""
    return "\n".join(lines[:first]).strip(), "\n".join(tail).strip()


def parse_citation_numbers(answer: str, available: int) -> tuple[list[int], list[int]]:
    """``(valid numbers in first-use order, numbers outside 1..available)``."""
    valid: list[int] = []
    unknown: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        number = int(match.group(1))
        target = valid if 1 <= number <= available else unknown
        if number not in target:
            target.append(number)
    return valid, unknown


def parse_quotes(sources_block: str) -> dict[int, list[str]]:
    """``{n: [quotes]}`` from the lines of the SOURCES block."""
    quotes: dict[int, list[str]] = {}
    for line in sources_block.splitlines():
        match = _QUOTE_LINE_RE.match(line)
        if match:
            quotes.setdefault(int(match.group(1)), []).append(match.group(2).strip())
    return quotes


def make_citations(
    answer: str,
    sources_block: str,
    contexts: Sequence[RetrievedChunk],
    question: str,
    snippet_chars: int,
) -> tuple[list[Citation], list[int]]:
    """Citations for each ``[n]`` used in ``answer``, with a snippet per citation."""
    numbers, unknown = parse_citation_numbers(answer, len(contexts))
    quotes = parse_quotes(sources_block)
    citations = []
    for number in numbers:
        chunk = contexts[number - 1]
        snippet, verified = _snippet(
            chunk, quotes.get(number, []), question, answer, snippet_chars
        )
        citations.append(Citation(number, chunk, snippet, verified))
    return sorted(citations, key=lambda c: c.number), unknown


def _snippet(
    chunk: RetrievedChunk, quotes: Sequence[str], question: str, answer: str, limit: int
) -> tuple[str, bool]:
    """A verified quote from the model if possible, else the best-matching sentences."""
    text = clean_chunk_text(chunk.text)
    for quote in quotes:
        if len(quote) < _MIN_QUOTE_CHARS:
            continue
        found = locate(text, quote)
        if found is not None:
            return _shorten(" ".join(found.text.split()), limit), True
    return best_snippet(text, f"{question} {answer}", limit), False


def best_snippet(text: str, query: str, limit: int) -> str:
    """The passage sentence window that shares most words with ``query``."""
    sentences = [s for s in _SENTENCE_RE.split(" ".join(text.split())) if s]
    if not sentences:
        return ""
    wanted = {w.lower() for w in _WORD_RE.findall(query)}
    scores = [len(wanted & {w.lower() for w in _WORD_RE.findall(s)}) for s in sentences]
    best = max(range(len(sentences)), key=lambda i: (scores[i], -i))
    low = high = best
    window = sentences[best]
    while True:  # grow the window around the best sentence while it fits
        grew = False
        if (
            high + 1 < len(sentences)
            and len(window) + len(sentences[high + 1]) + 1 <= limit
        ):
            high += 1
            window += " " + sentences[high]
            grew = True
        if low > 0 and len(window) + len(sentences[low - 1]) + 1 <= limit:
            low -= 1
            window = sentences[low] + " " + window
            grew = True
        if not grew:
            break
    return _shorten(window, limit)


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
