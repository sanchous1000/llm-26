"""Wiring: configuration + databases -> retriever / engine."""

from pydantic import ValidationError
from sqlalchemy.orm import sessionmaker

from source.indexing import vectorstore
from source.indexing.config import PipelineConfig
from source.indexing.db import repository as repo
from source.indexing.settings import InfraSettings
from source.rag.engine import RagEngine
from source.rag.errors import RagError
from source.rag.llm import Llm, OpencodeLlm
from source.rag.retriever import Retriever


def build_retriever(config: PipelineConfig) -> Retriever:
    """Connect to Qdrant and PostgreSQL (settings from ``.env``)."""
    try:
        settings = InfraSettings()
    except ValidationError as error:
        raise RagError(
            "database settings are missing: copy .env.example to .env "
            "and run `docker compose up -d`"
        ) from error
    client = vectorstore.make_client(settings)
    sessions = sessionmaker(repo.make_engine(settings))
    return Retriever(config, client, sessions)


def build_engine(config: PipelineConfig, llm: Llm | None = None) -> RagEngine:
    llm = llm or OpencodeLlm(config.rag.model, config.rag.llm_timeout_s)
    return RagEngine(build_retriever(config), llm, config.rag)
