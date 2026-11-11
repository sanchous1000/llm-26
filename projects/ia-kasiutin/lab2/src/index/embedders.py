from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from config import Config, EmbeddingConfig


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ChunkEmbeddings(Embeddings):
    """Sentence-transformers с префиксами search_document / search_query (по карточке FRIDA).

    Префикс подаётся только на вход энкодера и не попадает в сохранённый текст чанка.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.device = resolve_device(config.device)
        self.model = SentenceTransformer(config.model, device=self.device)
        self.model.max_seq_length = config.max_seq_length

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        return self.model.encode(
            [prefix + text for text in texts],
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1,
        ).astype(np.float32)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, self.config.document_prefix).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self.config.query_prefix)[0].tolist()

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.config.query_prefix)

    def embed_documents_array(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.config.document_prefix)


class EmbeddingCache:
    """Кэш векторов по sha256(model, параметры, префикс, текст) в одном файле на модель."""

    def __init__(self, directory: Path, config: EmbeddingConfig) -> None:
        slug = re.sub(r"[^\w.-]+", "_", config.model)
        self.directory = directory
        self.index_path = directory / f"{slug}.index.json"
        self.vectors_path = directory / f"{slug}.vectors.npy"
        self.salt = f"{config.model}|{config.document_prefix}|{config.normalize}|{config.max_seq_length}"
        self.keys: dict[str, int] = {}
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        if self.index_path.is_file() and self.vectors_path.is_file():
            self.keys = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.vectors = np.load(self.vectors_path)

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self.salt}|{text}".encode("utf-8")).hexdigest()

    def embed(self, texts: list[str], embedder: ChunkEmbeddings) -> tuple[np.ndarray, int]:
        """Возвращает (матрица векторов в порядке texts, число реально посчитанных)."""
        keys = [self.key(t) for t in texts]
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts):
            if key not in self.keys and key not in missing:
                missing[key] = text
        if missing:
            new = embedder.embed_documents_array(list(missing.values()))
            base = self.vectors if self.vectors.size else np.zeros((0, new.shape[1]), np.float32)
            self.vectors = np.vstack([base, new])
            for offset, key in enumerate(missing):
                self.keys[key] = len(base) + offset
            self._save()
        return self.vectors[[self.keys[k] for k in keys]], len(missing)

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp_vectors = self.vectors_path.with_suffix(".tmp.npy")
        np.save(tmp_vectors, self.vectors)
        os.replace(tmp_vectors, self.vectors_path)
        tmp_index = self.index_path.with_suffix(".tmp")
        tmp_index.write_text(json.dumps(self.keys), encoding="utf-8")
        os.replace(tmp_index, self.index_path)


_WORD = re.compile(r"\w+", re.UNICODE)


def bm25_tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("ё", "е"))


def build_bm25(texts: list[str], config: Config):
    from rank_bm25 import BM25Okapi

    sparse = config.embedding.sparse
    return BM25Okapi([bm25_tokenize(t) for t in texts], k1=sparse.k1, b=sparse.b)
