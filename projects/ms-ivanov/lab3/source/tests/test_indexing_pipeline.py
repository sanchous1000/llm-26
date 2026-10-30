import pytest
from sqlalchemy import select

from source.indexing import vectorstore
from source.indexing.config import PipelineConfig
from source.indexing.db import models as m
from source.indexing.db import repository as repo
from source.indexing.embedding import plan_batches
from source.indexing.pipeline import run_chunking, run_embedding
from source.tests.helpers import COUNTER, FakeEmbedder, config, make_docs


def counts(sessions):
    with sessions() as session:
        return repo.stats(session)


def test_chunking_is_incremental_and_idempotent(sessions, tmp_path):
    docs, cfg = make_docs(tmp_path), config()

    first = run_chunking(sessions, docs, cfg, COUNTER)
    again = run_chunking(sessions, docs, cfg, COUNTER)

    assert first.documents == 2 and first.skipped == 0 and first.chunks > 2
    assert again.documents == 0 and again.skipped == 2
    assert counts(sessions)["chunks"] == first.chunks

    docs[0].path.write_text(docs[0].path.read_text() + "\n\nextra paragraph " * 5)
    changed = run_chunking(sessions, docs, cfg, COUNTER)
    assert changed.documents == 1 and changed.skipped == 1


def test_different_configs_live_side_by_side_and_rebuild_resets(sessions, tmp_path):
    docs = make_docs(tmp_path)

    small = run_chunking(sessions, docs, config(chunk_size=30), COUNTER)
    large = run_chunking(sessions, docs, config(chunk_size=120), COUNTER)

    assert small.chunk_set_id != large.chunk_set_id
    assert counts(sessions)["chunk_sets"] == 2
    assert counts(sessions)["chunks"] == small.chunks + large.chunks
    rebuilt = run_chunking(sessions, docs, config(chunk_size=30), COUNTER, rebuild=True)
    assert (
        rebuilt.documents == 2
        and counts(sessions)["chunks"] == small.chunks + large.chunks
    )


def test_stored_chunks_match_the_document_text(sessions, tmp_path):
    docs, cfg = make_docs(tmp_path, 1), config()
    summary = run_chunking(sessions, docs, cfg, COUNTER)

    body = docs[0].path.read_text()
    with sessions() as session:
        rows = list(repo.iter_chunks(session, summary.chunk_set_id))
    assert [r.ord for r in rows] == list(range(len(rows)))
    assert all(body[r.start_char : r.end_char] == r.text for r in rows)
    assert rows[0].embed_text.startswith("Doc 0\n")
    assert session.get is not None


def test_embedding_fills_hybrid_collection_and_resumes(sessions, client, tmp_path):
    docs, cfg = make_docs(tmp_path), config()
    run_chunking(sessions, docs, cfg, COUNTER)
    embedder = FakeEmbedder()
    keys = [d.key for d in docs]

    first = run_embedding(sessions, client, cfg, lambda: embedder, keys, group_size=3)
    assert first.embedded == first.chunks > 0 and first.skipped == 0
    assert vectorstore.count(client, first.collection) == first.chunks

    point = client.scroll(first.collection, limit=1, with_vectors=True)[0][0]
    assert set(point.vector) == {"dense", "sparse"}
    assert {"doc_key", "ord", "section_path", "token_count"} <= set(point.payload)

    client.delete(first.collection, points_selector=[point.id])
    second = run_embedding(sessions, client, cfg, lambda: embedder, keys, group_size=3)
    assert second.embedded == 1 and second.skipped == first.chunks - 1
    third = run_embedding(sessions, client, cfg, lambda: embedder, keys)
    assert third.embedded == 0
    with sessions() as session:
        registered = session.scalars(select(m.VectorCollection)).one()
    assert registered.name == first.collection and registered.n_points == first.chunks


def test_dense_only_mode_stores_no_sparse_vectors(sessions, client, tmp_path):
    docs = make_docs(tmp_path, 1)
    cfg = PipelineConfig.model_validate(
        {
            "chunking": {"tokenizer": "words", "chunk_size": 60},
            "embedding": {"mode": "dense"},
        }
    )
    run_chunking(sessions, docs, cfg, COUNTER)

    summary = run_embedding(sessions, client, cfg, FakeEmbedder, ["doc0"])

    point = client.scroll(summary.collection, limit=1, with_vectors=True)[0][0]
    assert set(point.vector) == {"dense"}


def test_rebuild_recreates_the_collection(sessions, client, tmp_path):
    docs, cfg = make_docs(tmp_path, 1), config()
    run_chunking(sessions, docs, cfg, COUNTER)
    embedder = FakeEmbedder()
    run_embedding(sessions, client, cfg, lambda: embedder, ["doc0"])

    again = run_embedding(
        sessions, client, cfg, lambda: embedder, ["doc0"], rebuild=True
    )

    assert again.embedded == again.chunks and again.skipped == 0


def test_plan_batches_respects_budget_and_sorts_by_length():
    lengths = [10, 500, 20, 480, 5, 3000]

    batches = plan_batches(lengths, 1000)

    assert sorted(i for batch in batches for i in batch) == list(range(6))
    assert batches[0] == [5]  # longer than the budget -> alone
    assert all(
        len(batch) * lengths[batch[0]] <= 1000 or len(batch) == 1 for batch in batches
    )
    assert plan_batches([], 100) == []


def test_retrieval_metrics_count_tokens_precision_and_require_coverage():
    from types import SimpleNamespace

    from source.indexing.retrieval_check import summarise, to_result

    def hit(doc, start, end, tokens, point_id):
        payload = {
            "doc_key": doc,
            "start_char": start,
            "end_char": end,
            "token_count": tokens,
        }
        return SimpleNamespace(id=point_id, payload=payload)

    evidence = {"start": 100, "end": 200}
    hits = [
        hit("other", 100, 200, 500, "a"),
        hit("d", 0, 120, 700, "b"),  # covers only 20% of the evidence
        hit("d", 90, 300, 900, "c"),  # covers all of it
        hit("d", 150, 400, 100, "e"),  # covers half of it
    ]

    result = to_result(hits, "d", evidence)

    assert result.relevant == (False, False, True, True)
    assert result.first_rank == 3 and result.tokens_to_first == 2100
    scores = summarise("dense", [result, to_result([], "d", evidence)])
    assert scores.recall[5] == 0.5 and scores.recall[1] == 0.0
    assert scores.precision[5] == pytest.approx((2 / 5 + 0) / 2)
    assert scores.budget_recall[4096] == 0.5 and scores.budget_recall[1024] == 0.0


def test_check_retrieval_finds_the_chunk_that_covers_the_evidence(
    sessions, client, tmp_path
):
    from source.indexing.retrieval_check import check_retrieval

    docs, cfg = make_docs(tmp_path), config()
    run_chunking(sessions, docs, cfg, COUNTER)
    embedder = FakeEmbedder()
    name = run_embedding(
        sessions, client, cfg, lambda: embedder, ["doc0", "doc1"]
    ).collection
    with sessions() as session:
        target = next(iter(repo.iter_chunks(session, cfg.chunking.digest(), ["doc1"])))
    # the fake embedder is a text hash, so the query equal to the chunk text hits it exactly
    question = {
        "doc_key": "doc1",
        "question": target.embed_text,
        "evidence": {"start": target.start_char, "end": target.end_char},
    }

    scores = check_retrieval(
        client, name, embedder, [question], ("dense", "sparse", "hybrid")
    )

    assert [s.mode for s in scores] == ["dense", "sparse", "hybrid"]
    assert scores[0].recall[1] == 1.0 and scores[0].mrr == 1.0
    assert all(s.recall[10] == 1.0 for s in scores)
