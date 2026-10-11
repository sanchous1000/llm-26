import random

import pytest
from qdrant_client import QdrantClient

from source.indexing.embedding import EmbeddedText
from source.indexing.hnsw_report import (
    ANN_RECALL_TARGET,
    format_row,
    format_table,
    good_enough,
    recommend,
)
from source.indexing.hnsw_study import (
    BuildSpec,
    BuiltIndex,
    SearchSetting,
    StudyRow,
    Workload,
    _search_params,
    build_collection,
    estimated_graph_mb,
    evaluate,
    exact_top_ids,
    load_points,
)
from source.indexing.vectorstore import DENSE


def row(
    *, m=16, efc=100, ef=64, ann=0.995, r10=0.95, mrr=0.8, p50=3.0, quant="none"
) -> StudyRow:
    return StudyRow(
        m, efc, quant, ef, True, ann, 0.9, r10, 0.2, mrr, p50, p50 * 2, 1.0, m * 0.05
    )


def test_exact_row_is_the_reference_and_cheapest_acceptable_row_is_recommended():
    exact = row(ef=None, ann=1.0, r10=0.95, mrr=0.80)
    rows = [
        exact,
        row(m=4, ef=8, ann=0.6, r10=0.7, mrr=0.5),  # too lossy
        row(m=8, ef=32, ann=0.992, r10=0.95, mrr=0.80),  # ANN below the 0.995 target
        row(m=16, efc=128, ef=64, ann=0.997, r10=0.95, mrr=0.80),
        row(m=16, efc=64, ef=64, ann=0.996, r10=0.95, mrr=0.80),  # cheaper build
        row(m=16, efc=64, ef=128, ann=1.0, r10=0.95, mrr=0.80),  # over the ef budget
        row(m=32, ef=64, ann=0.999, r10=0.95, mrr=0.80),  # bigger graph
        row(
            m=8, ef=64, ann=0.999, r10=0.80, mrr=0.60, quant="scalar"
        ),  # quantized: not compared
    ]

    accepted = good_enough(rows)
    best = recommend(rows)

    assert exact not in accepted and len(accepted) == 3
    assert (best.m, best.ef_construct, best.ef) == (16, 64, 64)
    assert (
        recommend(rows, max_ef=128).ef == 64
    )  # still the cheapest in (m, efc, ef) order
    assert recommend([exact, row(ann=0.5)]) is None
    assert recommend([row()]) is None  # no exact baseline -> no reference


def test_ann_target_is_strict():
    assert ANN_RECALL_TARGET >= 0.995


def test_table_formatting_marks_exact_search():
    text = format_table([row(ef=None), row()])

    assert text.count("\n") == 3 and "| exact |" in text
    assert format_row(row()).startswith("| 16 | 100 | none | 64 |")


def test_graph_size_estimate_grows_with_m():
    assert estimated_graph_mb(1000, 32) > estimated_graph_mb(1000, 16) > 0


def test_search_params_for_exact_hnsw_and_quantized():
    assert _search_params(BuildSpec(16, 100), SearchSetting(None)).exact is True
    plain = _search_params(BuildSpec(16, 100), SearchSetting(48))
    assert plain.hnsw_ef == 48 and plain.quantization is None
    binary = _search_params(
        BuildSpec(16, 100, "binary"), SearchSetting(48, rescore=False)
    )
    assert (
        binary.quantization.rescore is False and binary.quantization.oversampling == 3.0
    )


def _server_available() -> bool:
    try:
        QdrantClient(host="localhost", port=6333, timeout=2).get_collections()
    except Exception:  # noqa: BLE001 - any failure means "no server"
        return False
    return True


@pytest.mark.skipif(not _server_available(), reason="Qdrant server is not running")
def test_benchmark_against_a_live_server():
    """Builds a real graph on random vectors and checks the measurements are sane."""
    client = QdrantClient(host="localhost", port=6333, timeout=30)
    rng = random.Random(1)
    source = "test_hnsw_source"
    if client.collection_exists(source):
        client.delete_collection(source)
    from qdrant_client import models

    client.create_collection(
        source,
        vectors_config={
            DENSE: models.VectorParams(size=16, distance=models.Distance.COSINE)
        },
    )
    vectors = [[rng.gauss(0, 1) for _ in range(16)] for _ in range(300)]
    client.upsert(
        source,
        [
            models.PointStruct(
                id=i,
                vector={DENSE: v},
                payload={
                    "doc_key": "d",
                    "start_char": i * 10,
                    "end_char": i * 10 + 10,
                    "token_count": 5,
                },
            )
            for i, v in enumerate(vectors)
        ],
        wait=True,
    )
    try:
        points = load_points(client, source)
        assert len(points) == 300
        spec = BuildSpec(8, 32)
        built_s = build_collection(client, "test_hnsw_bench", points, spec, "Cosine")
        built = BuiltIndex("test_hnsw_bench", spec, built_s, len(points))
        queries = [EmbeddedText(v, None, 0) for v in vectors[:20]]
        questions = [
            {"doc_key": "d", "evidence": {"start": i * 10, "end": i * 10 + 10}}
            for i in range(20)
        ]
        truth = exact_top_ids(client, "test_hnsw_bench", queries)
        load = Workload(queries, questions, truth, passes=1)

        exact = evaluate(client, built, SearchSetting(None), load)
        approx = evaluate(client, built, SearchSetting(64), load)

        assert exact.ann_recall == pytest.approx(1.0)
        assert approx.ann_recall > 0.9 and approx.mrr == pytest.approx(
            1.0
        )  # query = its own chunk
        assert approx.p95_ms >= approx.p50_ms > 0
    finally:
        for name in (source, "test_hnsw_bench"):
            if client.collection_exists(name):
                client.delete_collection(name)
