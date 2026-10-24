import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class Corpus(BaseModel):
    raw_dir: str = "data/raw"  # download_corpus.py кладёт статьи в data/raw/papers/*.pdf
    markdown_dir: str = "data/markdown"
    artifacts_dir: str = "artifacts"


class Splitter(BaseModel):
    chunk_size: int = 300       # в токенах
    chunk_overlap: int = 50
    include_headings: bool = True  # приклеивать ближайший заголовок раздела к тексту чанка


class Embeddings(BaseModel):
    vectorizer: str = "hybrid"  # dense | sparse | hybrid
    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 32


class Qdrant(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str = ""
    collection_name: str = "rag_chunks"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef_search: int = 64


class Model(BaseModel):
    url: str = "http://localhost:8000"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    temperature: float = 0.2
    max_tokens: int = 700
    max_context_chars: int = 6000


class Retrieval(BaseModel):
    top_k: int = 10


class Config(BaseModel):
    corpus: Corpus = Corpus()
    splitter: Splitter = Splitter()
    embeddings: Embeddings = Embeddings()
    qdrant: Qdrant = Qdrant()
    model: Model = Model()
    retrieval: Retrieval = Retrieval()


def path_of(cfg: Config, section: str, field: str) -> Path:
    return ROOT / getattr(getattr(cfg, section), field)


def load_config(path: Path | None = None) -> Config:
    path = path or (ROOT / "config.yml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg = Config(**(data or {}))
    if os.getenv("QDRANT_URL"):
        cfg.qdrant.url = os.getenv("QDRANT_URL")
    if os.getenv("QDRANT_API_KEY"):
        cfg.qdrant.api_key = os.getenv("QDRANT_API_KEY")
    if os.getenv("LLM_URL"):
        cfg.model.url = os.getenv("LLM_URL")
    if os.getenv("LLM_API_KEY"):
        cfg.model.api_key = os.getenv("LLM_API_KEY")
    return cfg


# Только параметры, которые реально нужно варьировать на шаге 5 (сравнение конфигураций).
FLAGS = {
    "--chunk-size": ("splitter", "chunk_size", int),
    "--overlap": ("splitter", "chunk_overlap", int),
    "--include-headings": ("splitter", "include_headings", lambda v: str(v).lower() in ("1", "true", "yes")),
    "--vectorizer": ("embeddings", "vectorizer", str),
    "--dense-model": ("embeddings", "dense_model", str),
    "--hnsw-m": ("qdrant", "hnsw_m", int),
    "--ef-construct": ("qdrant", "hnsw_ef_construct", int),
    "--ef-search": ("qdrant", "hnsw_ef_search", int),
}


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", help="путь к своему config.yml")
    for flag in FLAGS:
        ap.add_argument(flag)


def cfg_from_args(args) -> Config:
    cfg = load_config(Path(args.config) if getattr(args, "config", None) else None)
    for flag, (section, field, cast) in FLAGS.items():
        val = getattr(args, flag.lstrip("-").replace("-", "_"), None)
        if val is not None:
            setattr(getattr(cfg, section), field, cast(val))
    return cfg


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Для run_experiments.py: {"splitter": {"chunk_size": 500}, ...} поверх базового конфига."""
    data = cfg.model_dump()
    for section, fields in overrides.items():
        data[section].update(fields)
    return Config(**data)