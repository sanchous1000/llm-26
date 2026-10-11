import pytest
from pydantic import ValidationError

from source.indexing.config import ChunkingConfig, PipelineConfig, load_config


def test_defaults_and_stable_digests():
    first, second = PipelineConfig(), PipelineConfig()

    assert first.chunking.chunk_size == 1024 and first.embedding.model == "BAAI/bge-m3"
    assert first.chunking.digest() == second.chunking.digest()
    assert first.collection_name.startswith("chunks_")


def test_digest_changes_with_every_chunking_parameter():
    base = ChunkingConfig()

    variants = [
        ChunkingConfig(chunk_size=2048),
        ChunkingConfig(overlap=64),
        ChunkingConfig(splitter="fixed"),
        ChunkingConfig(include_headings=False),
    ]
    assert len({base.digest(), *(v.digest() for v in variants)}) == 5


def test_overrides_are_typed_and_validated(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("chunking:\n  chunk_size: 512\nmin_priority: 2\n")

    config = load_config(
        path, ["chunking.overlap=32", "embedding.mode=dense", "index.hnsw_m=32"]
    )

    assert (config.chunking.chunk_size, config.chunking.overlap) == (512, 32)
    assert config.embedding.mode == "dense" and config.index.hnsw_m == 32
    assert config.min_priority == 2


@pytest.mark.parametrize(
    "override",
    [
        "chunking.overlap=5000",
        "chunking.splitter=magic",
        "chunking.unknown=1",
        "no-equals",
    ],
)
def test_invalid_overrides_are_rejected(override):
    with pytest.raises((ValidationError, ValueError)):
        load_config(None, [override])


def test_config_is_immutable():
    with pytest.raises(ValidationError):
        ChunkingConfig().chunk_size = 5  # type: ignore[misc]
