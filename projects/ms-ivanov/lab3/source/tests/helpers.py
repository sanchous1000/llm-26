"""Shared test helpers: a deterministic embedder and tiny indexable documents."""

import hashlib
import re

from source.indexing.config import PipelineConfig
from source.indexing.embedding import EmbeddedText
from source.indexing.pipeline import DocInfo


class WordCounter:
    """Whitespace "tokens": deterministic and dependency-free."""

    _WORD = re.compile(r"\S+")

    def count(self, text: str) -> int:
        return len(self._WORD.findall(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [match.span() for match in self._WORD.finditer(text)]


COUNTER = WordCounter()


class FakeEmbedder:
    """Deterministic 8-d vectors derived from the text hash."""

    dense_dim = 8

    def __init__(self):
        self.calls: list[int] = []

    def embed(self, texts, on_batch=None):
        self.calls.append(len(texts))
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            dense = [b / 255 for b in digest[:8]]
            sparse = {int(b): 0.5 for b in digest[8:12]}
            out.append(EmbeddedText(dense, sparse, len(text.split())))
        return out


def config(**chunking) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "chunking": {
                "tokenizer": "words",
                "chunk_size": 60,
                "min_chunk_tokens": 5,
                **chunking,
            }
        }
    )


def make_docs(tmp_path, count=2) -> list[DocInfo]:
    docs = []
    for number in range(count):
        path = tmp_path / f"doc{number}.md"
        sections = "\n\n".join(
            f"## Section {s}\n\n" + " ".join(f"d{number}s{s}w{i}" for i in range(50))
            for s in range(3)
        )
        path.write_text(f"# Doc {number}\n\n{sections}\n", encoding="utf-8")
        docs.append(
            DocInfo(f"doc{number}", f"Doc {number}", "arxiv", 2020, "V", ("A",), 3, ("OT",),
                    "http://x", f"doc{number}.md", path)
        )  # fmt: skip
    return docs
