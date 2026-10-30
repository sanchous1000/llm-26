"""Validated, hashable configuration of the indexing pipeline.

Every parameter that changes the produced chunks or vectors is part of a
config, and the configs are hashed: artifacts of different configurations
live side by side (Postgres ``chunk_set``, one Qdrant collection per
chunking x embedding pair), so experiments never overwrite each other.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def digest(self) -> str:
        """Short stable hash of the field values."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]  # noqa: S324


class ChunkingConfig(_Frozen):
    """How documents are cut into chunks."""

    splitter: Literal["structured", "recursive", "fixed"] = "structured"
    tokenizer: str = "BAAI/bge-m3"  # chunk sizes are counted in these tokens
    chunk_size: int = Field(1024, ge=16, le=32768)
    overlap: int = Field(0, ge=0)
    # structured splitter
    split_level: int = Field(2, ge=1, le=6)  # headings up to this level start a chunk
    min_chunk_tokens: int = Field(128, ge=0)  # smaller chunks are merged forward
    # embedded text
    include_headings: bool = True  # prepend title + section path to embedded text
    strip_images: bool = True  # drop ![..](..) lines from embedded text
    include_references: bool = False  # index the bibliography section too

    @model_validator(mode="after")
    def _check(self) -> "ChunkingConfig":
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class EmbeddingConfig(_Frozen):
    """Which model produces the vectors and how."""

    model: str = "BAAI/bge-m3"
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    max_length: int = Field(8192, ge=16, le=32768)
    batch_tokens: int = Field(16384, ge=64)  # token budget of one forward pass


class VectorIndexConfig(_Frozen):
    """Qdrant collection parameters."""

    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    hnsw_m: int = Field(16, ge=4)
    hnsw_ef_construct: int = Field(100, ge=4)
    hnsw_ef: int = Field(
        64, ge=1
    )  # search-time beam width (not part of the stored index)
    on_disk: bool = False


class RagConfig(_Frozen):
    """Question answering on top of the index (step 6)."""

    top_k: int = Field(5, ge=1, le=50)  # contexts retrieved per question
    mode: Literal["auto", "dense", "sparse", "hybrid"] = (
        "auto"  # auto: by query language
    )
    max_context_tokens: int = Field(6000, ge=256)  # budget for passages in the prompt
    model: str = "opencode-go/deepseek-v4.1-flash"  # served through `opencode run`
    llm_timeout_s: int = Field(300, ge=10)
    history_turns: int = Field(3, ge=0)  # previous Q/A pairs kept in a chat prompt
    snippet_chars: int = Field(320, ge=80)  # length of the quoted snippet per citation


class PipelineConfig(_Frozen):
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    index: VectorIndexConfig = VectorIndexConfig()
    rag: RagConfig = RagConfig()
    min_priority: int = Field(3, ge=1, le=3)  # catalog stars required to be indexed

    @property
    def collection_name(self) -> str:
        """One Qdrant collection per (chunking, embedding, index) configuration."""
        return f"chunks_{self.chunking.digest()}_{self.embedding.digest()}"


def load_config(
    path: Path | None, overrides: list[str] | None = None
) -> PipelineConfig:
    """Read a YAML file (optional) and apply ``section.key=value`` overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        _apply_override(data, item)
    return PipelineConfig.model_validate(data)


def _apply_override(data: dict[str, Any], item: str) -> None:
    key, separator, raw = item.partition("=")
    if not separator:
        raise ValueError(f"override must look like section.key=value, got {item!r}")
    *parents, leaf = key.strip().split(".")
    node = data
    for parent in parents:
        node = node.setdefault(parent, {})
    node[leaf] = yaml.safe_load(raw)  # "4096" -> int, "true" -> bool, "x" -> str
