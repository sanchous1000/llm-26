"""Загрузка и валидация общего config.yaml."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

LAB2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = LAB2_ROOT / "config.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourcesConfig(_Section):
    dir: str = "docs"
    reparse: bool = False


class SplitterConfig(_Section):
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    header_levels: int = Field(default=3, ge=1, le=6)

    @model_validator(mode="after")
    def _check_overlap(self) -> SplitterConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Нужно 0 <= chunk_overlap < chunk_size")
        return self


class SparseConfig(_Section):
    k1: float = 1.5
    b: float = 0.75


class EmbeddingConfig(_Section):
    vectorization: Literal["dense", "hybrid"] = "dense"
    model: str = "ai-forever/FRIDA"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    batch_size: int = Field(default=16, gt=0)
    max_seq_length: int = Field(default=512, gt=0)
    normalize: bool = True
    query_prefix: str = "search_query: "
    document_prefix: str = "search_document: "
    sparse: SparseConfig = Field(default_factory=SparseConfig)


class HnswConfig(_Section):
    M: int = Field(default=32, ge=4)
    ef_construction: int = Field(default=200, ge=4)
    ef_search: int = Field(default=64, ge=1)

    @model_validator(mode="after")
    def _check_efc(self) -> HnswConfig:
        if self.ef_construction < self.M:
            raise ValueError("Нужно ef_construction >= M")
        return self


class IndexConfig(_Section):
    type: Literal["hnsw", "flat"] = "hnsw"
    hnsw: HnswConfig = Field(default_factory=HnswConfig)
    quantization: Literal["none", "fp16", "sq8"] = "none"
    dir: str = "faiss"

    @model_validator(mode="after")
    def _check_quantization(self) -> IndexConfig:
        if self.type == "flat" and self.quantization != "none":
            raise ValueError("index.quantization поддерживается только для type=hnsw")
        return self


class RagConfig(_Section):
    collection: str | None = None
    top_n: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    max_context_tokens: int = Field(default=3000, ge=1)
    snippet_chars: int = Field(default=240, ge=1)


class LlmConfig(_Section):
    base_url: str | None = None
    model: str = "auto"
    temperature: float = Field(default=0.1, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    max_tokens: int = Field(default=700, ge=1)
    repetition_penalty: float = 1.15
    timeout_s: float = Field(default=120, gt=0)


class ArtifactsConfig(_Section):
    dir: str = "artifacts"


class Config(_Section):
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    root: Path = Field(default=LAB2_ROOT, exclude=True)

    @model_validator(mode="after")
    def _check_seq_len(self) -> Config:
        if self.splitter.chunk_size > self.embedding.max_seq_length:
            raise ValueError(
                f"chunk_size={self.splitter.chunk_size} больше "
                f"embedding.max_seq_length={self.embedding.max_seq_length}: модель обрежет вход"
            )
        return self

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else self.root / path

    @property
    def sources_dir(self) -> Path:
        return self.resolve(self.sources.dir)

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve(self.artifacts.dir)

    @property
    def faiss_dir(self) -> Path:
        return self.artifacts_dir / self.index.dir

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def _set_override(data: dict[str, Any], dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Переопределение должно быть вида a.b=value: {item!r}")
        dotted, raw = item.split("=", 1)
        _set_override(data, dotted.strip(), raw)
    config = Config.model_validate(data)
    config.root = path.resolve().parent
    return config


def sources_fingerprint(directory: Path) -> list[list[Any]]:
    items = []
    parsed_dir = LAB2_ROOT / "src" / "preprocessing" / "output"
    roots = [directory] + ([parsed_dir] if parsed_dir.is_dir() else [])
    for root in roots:
        for item in sorted(root.rglob("*")):
            if item.is_file() and item.suffix != ".jsonl":
                stat = item.stat()
                items.append([item.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns])
    return items


def config_hash(config: Config) -> str:
    """Хэш того, что влияет на содержимое артефактов (device и batch_size не входят)."""
    dumped = config.to_dict()
    embedding = dumped["embedding"]
    embedding.pop("device")
    embedding.pop("batch_size")
    if config.embedding.vectorization == "dense":
        embedding.pop("sparse")
    payload = {
        "splitter": dumped["splitter"],
        "embedding": embedding,
        "sources": sources_fingerprint(config.sources_dir),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
