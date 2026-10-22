"""Value objects of the RAG layer."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved passage with everything needed to cite it."""

    rank: int
    score: float
    chunk_id: str
    doc_key: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    venue: str | None
    source_url: str | None
    ord: int
    start_char: int
    end_char: int
    token_count: int
    section_path: tuple[str, ...]
    page: int | None
    text: str

    @property
    def section(self) -> str:
        return " > ".join(self.section_path)

    @property
    def short_citation(self) -> str:
        """``Korotin et al., 2022``-style label."""
        if not self.authors:
            author = ""
        else:
            surname = self.authors[0].split()[-1]
            author = surname if len(self.authors) == 1 else f"{surname} et al."
        return ", ".join(part for part in (author, str(self.year or "")) if part)

    def to_dict(self, max_chars: int | None = None) -> dict:
        data = asdict(self)
        data["url"] = data.pop("source_url")
        data["section"] = self.section
        data["citation"] = self.short_citation
        data["authors"] = list(self.authors)
        data["section_path"] = list(self.section_path)
        if max_chars is not None and len(self.text) > max_chars:
            data["text"] = self.text[:max_chars].rstrip() + " [...]"
            data["truncated"] = True
        return data


@dataclass(frozen=True)
class Citation:
    """A numbered source the answer refers to."""

    number: int
    chunk: RetrievedChunk
    snippet: str
    verified: bool  # the snippet is a verbatim quote found in the passage


@dataclass(frozen=True)
class Answer:
    """Result of one question."""

    question: str
    standalone_question: str
    text: str
    citations: tuple[Citation, ...]
    contexts: tuple[RetrievedChunk, ...]
    unknown_citations: tuple[int, ...] = ()
    model: str = ""
    mode: str = ""
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "standalone_question": self.standalone_question,
            "answer": self.text,
            "model": self.model,
            "mode": self.mode,
            "timings": self.timings,
            "citations": [
                {
                    "number": c.number,
                    "doc_key": c.chunk.doc_key,
                    "title": c.chunk.title,
                    "citation": c.chunk.short_citation,
                    "section": c.chunk.section,
                    "page": c.chunk.page,
                    "url": c.chunk.source_url,
                    "chunk_id": c.chunk.chunk_id,
                    "snippet": c.snippet,
                    "snippet_verified": c.verified,
                }
                for c in self.citations
            ],
            "unknown_citations": list(self.unknown_citations),
        }
