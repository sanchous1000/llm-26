"""Benchmark of Qdrant HNSW parameters on the real chunk vectors.

For every build setting (``m`` x ``ef_construct``) a scratch collection is
filled with the dense vectors of a source collection (no re-embedding), its
graph is built explicitly, and each search ``ef`` is evaluated on:

* **ANN recall@10** - overlap of the approximate top-10 with the exact
  (brute-force) top-10: how much the graph loses compared with exact search;
* retrieval quality on the evaluation questions (Recall/Precision/MRR);
* query latency (p50/p95, sequential single queries over gRPC);
* build time and an estimate of the graph size.
"""

import logging
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass

from qdrant_client import QdrantClient, models

from source.indexing.embedding import EmbeddedText
from source.indexing.retrieval_check import summarise, to_result
from source.indexing.vectorstore import DENSE, wait_until_indexed

logger = logging.getLogger(__name__)
TOP_K = 10
WARMUP_QUERIES = 20
_BUILD_TIMEOUT_S = 600


@dataclass(frozen=True)
class Point:
    point_id: str | int  # kept as stored: Qdrant ids are UUID strings or integers
    dense: list[float]
    payload: dict


@dataclass(frozen=True)
class BuildSpec:
    """Index-time settings of one scratch collection."""

    m: int
    ef_construct: int
    quantization: str = "none"  # none | scalar | binary

    @property
    def label(self) -> str:
        return f"m{self.m}_efc{self.ef_construct}_{self.quantization}"


@dataclass(frozen=True)
class StudyRow:
    m: int
    ef_construct: int
    quantization: str
    ef: int | None  # None = exact search
    rescore: bool
    ann_recall: float
    recall5: float
    recall10: float
    precision5: float
    mrr: float
    p50_ms: float
    p95_ms: float
    build_s: float
    graph_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_points(client: QdrantClient, collection: str) -> list[Point]:
    """All dense vectors with payloads of ``collection``."""
    points: list[Point] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=[DENSE],
        )
        points += [Point(p.id, p.vector[DENSE], p.payload) for p in batch]
        if offset is None:
            return points


def build_collection(
    client: QdrantClient,
    name: str,
    points: Sequence[Point],
    spec: BuildSpec,
    distance: str,
) -> float:
    """Create ``name`` and build its HNSW graph; returns the build time in seconds."""
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={
            DENSE: models.VectorParams(
                size=len(points[0].dense),
                distance=models.Distance(distance),
                hnsw_config=models.HnswConfigDiff(
                    m=spec.m, ef_construct=spec.ef_construct
                ),
            )
        },
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        quantization_config=_quantization(spec.quantization),
    )
    for start in range(0, len(points), 256):
        client.upsert(
            name,
            [
                models.PointStruct(
                    id=p.point_id, vector={DENSE: p.dense}, payload=p.payload
                )
                for p in points[start : start + 256]
            ],
            wait=True,
        )
    began = time.monotonic()
    client.update_collection(
        name, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1)
    )
    wait_until_indexed(client, name, _BUILD_TIMEOUT_S)
    return time.monotonic() - began


def _quantization(kind: str):
    if kind == "scalar":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, quantile=0.99, always_ram=True
            )
        )
    if kind == "binary":
        return models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True)
        )
    return None


def estimated_graph_mb(points: int, m: int) -> float:
    """Level-0 keeps up to 2m uint32 links per node; upper layers add about one more."""
    return points * (2 * m + 1) * 4 / 1e6


def exact_top_ids(
    client: QdrantClient, name: str, queries: Sequence[EmbeddedText]
) -> list[tuple[str, ...]]:
    """Brute-force top-k ids per query (ground truth for ANN recall)."""
    exact = models.SearchParams(exact=True)
    return [
        tuple(str(h.id) for h in _query(client, name, q, TOP_K, exact).points)
        for q in queries
    ]


def _query(client: QdrantClient, name: str, vector: EmbeddedText, limit: int, params):
    return client.query_points(
        name, query=vector.dense, using=DENSE, limit=limit,
        search_params=params, with_payload=True,
    )  # fmt: skip


@dataclass(frozen=True)
class BuiltIndex:
    """A scratch collection with its graph already built."""

    name: str
    spec: BuildSpec
    build_s: float
    n_points: int


@dataclass(frozen=True)
class SearchSetting:
    ef: int | None  # None = exact (brute-force) search
    rescore: bool = True  # only meaningful with quantization


@dataclass(frozen=True)
class Workload:
    """The queries every setting is measured on."""

    queries: Sequence[EmbeddedText]
    questions: Sequence[dict]
    truth: Sequence[tuple[str, ...]]  # exact top-k ids per query
    passes: int = 3


def evaluate(
    client: QdrantClient, built: BuiltIndex, setting: SearchSetting, load: Workload
) -> StudyRow:
    """Run all queries ``load.passes`` times with the given search settings."""
    params = _search_params(built.spec, setting)
    for query in load.queries[:WARMUP_QUERIES]:
        _query(client, built.name, query, TOP_K, params)
    latencies: list[float] = []
    results, ann = [], []
    for pass_number in range(load.passes):
        for query, question, expected in zip(
            load.queries, load.questions, load.truth, strict=True
        ):
            started = time.perf_counter()
            hits = _query(client, built.name, query, TOP_K, params).points
            latencies.append((time.perf_counter() - started) * 1000)
            if pass_number == 0:
                results.append(
                    to_result(hits, question["doc_key"], question["evidence"])
                )
                ann.append(len(set(expected) & {str(h.id) for h in hits}) / TOP_K)
    scores = summarise("dense", results)
    cuts = statistics.quantiles(latencies, n=100)
    spec = built.spec
    return StudyRow(
        spec.m, spec.ef_construct, spec.quantization, setting.ef, setting.rescore,
        statistics.fmean(ann), scores.recall[5], scores.recall[10], scores.precision[5],
        scores.mrr, statistics.median(latencies), cuts[94], built.build_s,
        estimated_graph_mb(built.n_points, spec.m),
    )  # fmt: skip


def _search_params(spec: BuildSpec, setting: SearchSetting) -> models.SearchParams:
    if setting.ef is None:
        return models.SearchParams(exact=True)
    quantization = None
    if spec.quantization != "none":
        quantization = models.QuantizationSearchParams(
            rescore=setting.rescore,
            oversampling=3.0 if spec.quantization == "binary" else 2.0,
        )
    return models.SearchParams(
        hnsw_ef=setting.ef, exact=False, quantization=quantization
    )


def run_grid(
    client: QdrantClient,
    source: str,
    queries: Sequence[EmbeddedText],
    questions: Sequence[dict],
    specs: Iterable[BuildSpec],
    *,
    efs: Sequence[int],
    passes: int = 3,
    keep: bool = False,
    on_row: Callable[[StudyRow], None] | None = None,
) -> list[StudyRow]:
    """Evaluate every build ``spec`` x search ``ef`` (plus the exact baseline)."""
    points = load_points(client, source)
    distance = client.get_collection(source).config.params.vectors[DENSE].distance.value
    logger.info("loaded %d dense vectors (%s) from %s", len(points), distance, source)
    rows: list[StudyRow] = []
    load: Workload | None = None
    for spec in specs:
        name = f"bench_{spec.label}"
        built = BuiltIndex(
            name,
            spec,
            build_collection(client, name, points, spec, distance),
            len(points),
        )
        logger.info("%s built in %.1fs", spec.label, built.build_s)
        if load is None:
            truth = exact_top_ids(client, name, queries)
            load = Workload(queries, questions, truth, passes)
            rows.append(evaluate(client, built, SearchSetting(None), load))
        quantized = spec.quantization != "none"
        for ef in efs:
            for rescore in (True, False) if quantized else (True,):
                row = evaluate(client, built, SearchSetting(ef, rescore), load)
                rows.append(row)
                if on_row:
                    on_row(row)
        if not keep:
            client.delete_collection(name)
    return rows
