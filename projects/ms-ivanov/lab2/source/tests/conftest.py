"""Shared fixtures: sample data and a scripted fake HTTP client."""

import hashlib
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from source.bibliography.models import Source, SourceKind
from source.downloaders.base import DownloadContext
from source.indexing.db import repository as repo
from source.indexing.pipeline import run_chunking, run_embedding
from source.net import FetchedFile, FetchError
from source.rag.retriever import Retriever
from source.tests.helpers import COUNTER, FakeEmbedder, config, make_docs

PDF_BYTES = b"%PDF-1.7 fake pdf"
HTML_BYTES = (
    b"<!DOCTYPE html><html><body>"
    + b'<p class="ltx_p">text</p>' * 12
    + b"</body></html>"
)

SAMPLE_BIB = """\
% comment line
@misc{peyre2018computational,
  title = {Computational Optimal Transport},
  author = {Gabriel Peyré and Marco Cuturi},
  year = {2019},
  eprint = {1803.00567},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/1803.00567},
  howpublished = {FnT ML}
}

@article{carlsson2009topology,
  title = {Topology and data},
  author = {Gunnar Carlsson},
  year = {2009},
  url = {https://doi.org/10.1090/S0273-0979-09-01249-X},
  doi = {10.1090/S0273-0979-09-01249-X},
  journal = {Bulletin of the AMS 46}
}

@misc{argintum2025rtdlite,
  title = {RTD-Lite official code},
  author = {ArGintum},
  year = {2025},
  url = {https://github.com/ArGintum/RTD-Lite},
  howpublished = {GitHub}
}

@misc{optimaltransport,
  title = {Computational OT course},
  author = {Someone},
  year = {2020},
  url = {https://optimaltransport.github.io/}
}

@article{rakotomamonjy2020optimal,
  title = {Optimal Transport},
  author = {A and B},
  year = {2021},
  eprint = {2006.08161},
  url = {https://arxiv.org/abs/2006.08161},
  doi = {10.1007/s10994-021-06088-2}
}
"""

SAMPLE_CATALOG = """\
# Sources

## 1. Books & surveys

- `peyre2018computational` ★★★ **[Computational Optimal Transport](https://arxiv.org/abs/1803.00567)**
  Gabriel Peyré, Marco Cuturi — *FnT ML* — 2018-03-01 — `SURVEY` `OT`
  Standard reference.
  Second line.
- `carlsson2009topology` ★☆☆ **[Topology and data](https://doi.org/x)** 📝
  Gunnar Carlsson — *Bull. AMS* — 2009 — `TDA`

## 2. Another section

- `argintum2025rtdlite` ★★☆ **[RTD-Lite](https://github.com/ArGintum/RTD-Lite)**
  ArGintum — GitHub — 2025 — `TOOL`
  Code.
"""


def make_source(kind: SourceKind = SourceKind.ARXIV, **overrides) -> Source:
    defaults = {
        "key": "key2020",
        "kind": kind,
        "title": "A Title",
        "url": "https://arxiv.org/abs/2001.00001",
        "arxiv_id": "2001.00001" if kind is SourceKind.ARXIV else None,
        "doi": "10.1000/xyz" if kind is SourceKind.DOI else None,
    }
    return Source(**{**defaults, **overrides})


class FakeClient:
    """Scripted stand-in for :class:`HttpClient`.

    ``responses`` maps URL -> bytes (success), dict (JSON) or FetchError.
    Unknown URLs behave like a permanent 404.
    """

    def __init__(self, responses: dict):
        self.responses = responses
        self.requested: list[str] = []

    def fetch_file(self, url, dest: Path, validator, headers=None) -> FetchedFile:
        self.requested.append(url)
        outcome = self.responses.get(
            url, FetchError("HTTP 404", permanent=True, status=404)
        )
        if isinstance(outcome, FetchError):
            raise outcome
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        partial.write_bytes(outcome)
        try:
            validator(partial, "test/type")
            partial.replace(dest)
        finally:
            partial.unlink(missing_ok=True)
        return FetchedFile(
            dest, url, len(outcome), hashlib.sha256(outcome).hexdigest(), "t/t"
        )

    def get_json(self, url, headers=None) -> dict:
        self.requested.append(url)
        outcome = self.responses.get(
            url, FetchError("HTTP 404", permanent=True, status=404)
        )
        if isinstance(outcome, FetchError):
            raise outcome
        return outcome


@pytest.fixture
def make_context():
    def _make(responses: dict, **options) -> tuple[DownloadContext, FakeClient]:
        client = FakeClient(responses)
        return DownloadContext(client=client, **options), client

    return _make


@pytest.fixture
def sessions():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    repo.init_schema(engine)
    return sessionmaker(engine)


@pytest.fixture
def client():
    return QdrantClient(":memory:")


@pytest.fixture
def retriever(sessions, client, tmp_path):
    docs, cfg = make_docs(tmp_path), config()
    run_chunking(sessions, docs, cfg, COUNTER)
    embedder = FakeEmbedder()
    run_embedding(sessions, client, cfg, lambda: embedder, [d.key for d in docs])
    return Retriever(cfg, client, sessions, embedder_factory=lambda: embedder), cfg
