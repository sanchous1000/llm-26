"""Query -> ranked, citable passages (Qdrant for vectors, PostgreSQL for texts)."""

import logging
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from sqlalchemy.orm import Session, sessionmaker

from source.indexing import vectorstore
from source.indexing.config import PipelineConfig
from source.indexing.db import models as m
from source.indexing.db import repository as repo
from source.indexing.embedding import Embedder, build_embedder
from source.indexing.search import document_filter, search
from source.rag.errors import RagError
from source.rag.models import RetrievedChunk

logger = logging.getLogger(__name__)

_CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
_LETTER_RE = re.compile(r"[^\W\d_]")
_CYRILLIC_SHARE = 0.3
_AVAILABLE_MODES = {
    "hybrid": ("hybrid", "dense", "sparse"),
    "dense": ("dense",),
    "sparse": ("sparse",),
}


@dataclass(frozen=True)
class DocumentInfo:
    key: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    venue: str | None
    kind: str
    priority: int | None
    tags: tuple[str, ...]
    url: str | None
    chunks: int

    def to_dict(self) -> dict:
        return {
            "doc_key": self.key,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "kind": self.kind,
            "priority": self.priority,
            "tags": list(self.tags),
            "url": self.url,
            "chunks": self.chunks,
        }


def resolve_mode(query: str, requested: str, built_mode: str) -> str:
    """Pick the search mode: ``auto`` avoids sparse search for non-English queries.

    The sparse (lexical) vectors only match words that occur in the English
    papers, so a Russian query is served by the dense vectors alone.
    """
    available = _AVAILABLE_MODES[built_mode]
    if requested != "auto":
        if requested not in available:
            raise ValueError(
                f"mode {requested!r} is not available (index built as {built_mode})"
            )
        return requested
    letters = _LETTER_RE.findall(query)
    share = len(_CYRILLIC_RE.findall(query)) / len(letters) if letters else 0.0
    preferred = "dense" if share >= _CYRILLIC_SHARE else "hybrid"
    return preferred if preferred in available else available[0]


class Retriever:
    """Thread-safe search facade; the embedder loads lazily and stays resident."""

    def __init__(
        self,
        config: PipelineConfig,
        client: QdrantClient,
        sessions: sessionmaker[Session],
        embedder_factory: Callable[[], Embedder] | None = None,
    ):
        self._config = config
        self._client = client
        self._sessions = sessions
        self._embedder_factory = embedder_factory or (
            lambda: build_embedder(config.embedding)
        )
        self._embedder: Embedder | None = None
        self._lock = threading.Lock()

    @property
    def built_mode(self) -> str:
        """Vectors stored in the collection: ``dense``, ``sparse`` or ``hybrid``."""
        return self._config.embedding.mode

    @property
    def collection(self) -> str:
        return self._config.collection_name

    @property
    def chunk_set_id(self) -> str:
        return self._config.chunking.digest()

    def check_ready(self) -> None:
        """Fail early, with advice, when the index has not been built."""
        if not self._client.collection_exists(self.collection):
            raise RagError(
                f"collection {self.collection} does not exist: run "
                "`docker compose up -d` and `python -m source index` first"
            )

    def warm_up(self) -> None:
        """Load the embedding model now instead of during the first query."""
        self._embed("warm-up")

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: str | None = None,
        doc_keys: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """The ``top_k`` most relevant passages for ``query``."""
        rag = self._config.rag
        limit = top_k or rag.top_k
        resolved = resolve_mode(query, mode or rag.mode, self._config.embedding.mode)
        self.check_ready()
        hits = search(
            self._client,
            self.collection,
            resolved,
            self._embed(query),
            limit=limit,
            params=vectorstore.search_params(self._config),
            query_filter=document_filter(doc_keys),
        )
        return self._materialise(hits)

    def neighbors(
        self, chunk_id: str, before: int = 1, after: int = 1
    ) -> list[RetrievedChunk]:
        """A chunk with ``before``/``after`` neighbouring chunks of its document."""
        with self._sessions() as session:
            found = repo.load_chunks(session, self.chunk_set_id, [chunk_id])
            if chunk_id not in found:
                raise RagError(f"unknown chunk id {chunk_id}")
            chunk, _ = found[chunk_id]
            rows = repo.chunk_window(
                session,
                self.chunk_set_id,
                chunk.doc_key,
                max(0, chunk.ord - before),
                chunk.ord + after,
            )
            return [
                _to_chunk(row, doc, rank=i, score=0.0)
                for i, (row, doc) in enumerate(rows, 1)
            ]

    def documents(
        self, contains: str | None = None, limit: int = 100
    ) -> list[DocumentInfo]:
        with self._sessions() as session:
            rows = repo.list_documents(session, self.chunk_set_id, contains, limit)
            return [
                DocumentInfo(
                    d.key,
                    d.title,
                    tuple(d.authors or ()),
                    d.year,
                    d.venue,
                    d.kind,
                    d.priority,
                    tuple(d.tags or ()),
                    d.source_url,
                    n_chunks,
                )
                for d, n_chunks in rows
            ]

    def outline(self, doc_key: str) -> list[dict]:
        """Sections of a document in reading order with the chunks that start them."""
        with self._sessions() as session:
            chunks = repo.outline(session, self.chunk_set_id, doc_key)
        if not chunks:
            raise RagError(f"unknown document {doc_key!r}")
        sections: list[dict] = []
        for chunk in chunks:
            path = " > ".join(chunk.section_path)
            if sections and sections[-1]["section"] == path:
                sections[-1]["chunks"].append(chunk.chunk_id)
                sections[-1]["tokens"] += chunk.token_count
            else:
                sections.append(
                    {
                        "section": path,
                        "page": chunk.page_start,
                        "chunks": [chunk.chunk_id],
                        "tokens": chunk.token_count,
                    }
                )
        return sections

    def _embed(self, text: str):
        with self._lock:
            if self._embedder is None:
                self._embedder = self._embedder_factory()
            return self._embedder.embed([text])[0]

    def _materialise(self, hits) -> list[RetrievedChunk]:
        ids = [str(hit.id) for hit in hits]
        with self._sessions() as session:
            rows = repo.load_chunks(session, self.chunk_set_id, ids)
            result = []
            for rank, hit in enumerate(hits, start=1):
                if str(hit.id) not in rows:
                    logger.warning(
                        "chunk %s is in Qdrant but not in PostgreSQL", hit.id
                    )
                    continue
                chunk, document = rows[str(hit.id)]
                result.append(
                    _to_chunk(chunk, document, rank=rank, score=float(hit.score))
                )
            return result


def _to_chunk(
    chunk: m.Chunk, document: m.Document, *, rank: int, score: float
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        score=score,
        chunk_id=chunk.chunk_id,
        doc_key=chunk.doc_key,
        title=document.title,
        authors=tuple(document.authors or ()),
        year=document.year,
        venue=document.venue,
        source_url=document.source_url,
        ord=chunk.ord,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        token_count=chunk.token_count,
        section_path=tuple(chunk.section_path or ()),
        page=chunk.page_start,
        text=chunk.text,
    )
