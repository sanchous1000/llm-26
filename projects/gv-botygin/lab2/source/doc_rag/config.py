import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

SOURCE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = SOURCE_DIR / "config.yml"

load_dotenv(SOURCE_DIR / ".env")


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection_prefix: str = "docs"


class EmbeddingsConfig(BaseModel):
    model_name: str = "bge-m3"  # идентификатор модели - входит в имя коллекции
    model_path: str = "models/bge-m3"  # локальная папка модели; относительный путь - от папки lab2
    device: str = "auto"  # auto | cuda | cpu
    fp16: bool = True  # половинная точность на GPU (меньше памяти и быстрее)
    batch_size: int = Field(16, gt=0)
    max_seq_length: int = Field(512, gt=0)  # усечение входа при векторизации (bge-m3 умеет до 8192)

    def resolved_model_path(self) -> Path:
        p = Path(self.model_path)
        return p if p.is_absolute() else SOURCE_DIR.parent / p


class SplitterConfig(BaseModel):
    chunk_size: int = Field(300, ge=1)
    chunk_overlap: int = Field(50, ge=0)
    headers_to_split_on: list[int] = [1, 2, 3]
    include_headers_in_text: bool = True

    @model_validator(mode="after")
    def _check(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("splitter.chunk_overlap должен быть меньше chunk_size")
        if not all(1 <= lvl <= 3 for lvl in self.headers_to_split_on):
            raise ValueError("splitter.headers_to_split_on: допустимы уровни 1-3")
        return self


class SearchConfig(BaseModel):
    top_k: int = Field(5, ge=1)
    score_threshold: float = 0.0


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen3.5:4b"
    # qwen3.5 - thinking-модель: без отключения рассуждения съедают max_tokens (None - не передавать)
    reasoning_effort: str | None = "none"
    temperature: float = 0.0
    max_tokens: int = 500
    timeout: int = 300


class Config(BaseModel):
    qdrant: QdrantConfig = QdrantConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    splitter: SplitterConfig = SplitterConfig()
    search: SearchConfig = SearchConfig()
    llm: LLMConfig = LLMConfig()

    def index_params(self) -> dict:
        """Параметры, влияющие на содержимое индекса (и, значит, на имя коллекции)."""
        return {
            "embeddings": {
                "model_name": self.embeddings.model_name,
                "max_seq_length": self.embeddings.max_seq_length,
            },
            "splitter": self.splitter.model_dump(),
        }

    def collection_name(self) -> str:
        return collection_name(self)


def collection_name(config: Config) -> str:
    """{prefix}_{sha1(splitter + embeddings)[:8]}.

    Смена splitter/embeddings даёт новую коллекцию, смена search/llm/batch_size — нет.
    """
    raw = json.dumps(config.index_params(), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{config.qdrant.collection_prefix}_{digest}"


def resolve_config_path(path: str | Path | None = None) -> Path:
    p = Path(path or os.getenv("DOC_RAG_CONFIG") or DEFAULT_CONFIG_PATH)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return SOURCE_DIR / p  # относительный путь — от папки source


@lru_cache()
def load_config(path: str | Path | None = None) -> Config:
    p = resolve_config_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # секреты — только из окружения
    if os.getenv("QDRANT_API_KEY"):
        data.setdefault("qdrant", {})["api_key"] = os.environ["QDRANT_API_KEY"]
    if os.getenv("LLM_API_KEY"):
        data.setdefault("llm", {})["api_key"] = os.environ["LLM_API_KEY"]

    return Config(**data)


def get_config() -> Config:
    return load_config()
