"""Метрики retrieval (п.5): эталонные цитаты, сетка FAISS, Recall/Precision/MRR.

Запуск из lab2/eval:
    python evaluate.py
Эталон: questions.json рядом со скриптом.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_EVAL_DIR = Path(__file__).resolve().parent
_SRC = _EVAL_DIR.parent / "src"
for _p in (str(_SRC), str(_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import faiss
import numpy as np

from config import Config, load_config
from index.load_to_vector_store import (
    INDEX_FILE,
    build_collection,
    dense_depth,
    load_bm25,
    rank_indices,
    set_ef_search,
)
from report import METRIC_KEYS, index_title, render_report

KS = (5, 10)
COVERAGE = 0.6
EF_SEARCH = (16, 64, 128)
RUN_IDS = (
    "dense-frida-markdown_header-cs400-ov50-dcc2c0ee",
    "dense-frida-recursive-cs400-ov50-85f6432f",
)


def discover_run_ids(artifacts_dir: Path) -> tuple[str, ...]:
    found = sorted(
        p.name
        for p in artifacts_dir.iterdir()
        if p.is_dir() and (p / "manifest.json").is_file() and (p / "chunks.jsonl").is_file()
    )
    return tuple(found) if found else RUN_IDS
# M=32 и ef_construction=200 у flat не используются, но проходят валидацию конфига.
BUILDS: tuple[dict[str, Any], ...] = (
    {"type": "flat", "M": 32, "ef_construction": 200, "quantization": "none"},
    {"type": "hnsw", "M": 16, "ef_construction": 64, "quantization": "none"},
    {"type": "hnsw", "M": 16, "ef_construction": 200, "quantization": "none"},
    {"type": "hnsw", "M": 32, "ef_construction": 200, "quantization": "none"},
    {"type": "hnsw", "M": 32, "ef_construction": 200, "quantization": "sq8"},
    {"type": "hnsw", "M": 32, "ef_construction": 200, "quantization": "fp16"},
)

_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def covers(quote: str, chunk: str, threshold: float = COVERAGE) -> bool:
    """Цитата покрыта целиком или непрерывным фрагментом не короче threshold."""
    quote_norm = normalize_text(quote)
    chunk_norm = normalize_text(chunk)
    if not quote_norm:
        return False
    if quote_norm in chunk_norm:
        return True
    quote_tokens = tokens(quote_norm)
    if not quote_tokens:
        return False
    need = math.ceil(threshold * len(quote_tokens) - 1e-12)
    if need <= 0:
        return False
    haystack = " ".join(tokens(chunk_norm))
    last = len(quote_tokens) - need + 1
    for start in range(last):
        window = " ".join(quote_tokens[start : start + need])
        if window in haystack:
            return True
    return False


def evidence_hits(chunks: list[str], evidence: list[dict[str, Any]]) -> list[set[int]]:
    return [
        {i for i, text in enumerate(chunks) if covers(item["quote"], text)}
        for item in evidence
    ]


def ranking_metrics(
    ranking: list[int],
    hits: list[set[int]],
    ks: tuple[int, ...] = KS,
) -> dict[str, float]:
    """Recall — доля пассажей, Precision — доля чанков top-k, MRR — по всему списку."""
    flags = [any(idx in passage for passage in hits) for idx in ranking]
    n_passages = len(hits)
    metrics: dict[str, float] = {}
    for k in ks:
        if k <= 0:
            raise ValueError("k должен быть > 0")
        top = set(ranking[:k])
        found = sum(1 for passage in hits if passage & top)
        metrics[f"recall@{k}"] = found / n_passages if n_passages else 0.0
        metrics[f"precision@{k}"] = sum(flags[:k]) / k
    first = next((rank for rank, flag in enumerate(flags, start=1) if flag), None)
    metrics["mrr"] = 0.0 if first is None else 1.0 / first
    return metrics


def macro_average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Нет строк для усреднения")
    return {key: sum(row[key] for row in rows) / len(rows) for key in METRIC_KEYS}


def top_overlap(rankings: list[list[int]], reference: list[list[int]], k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k должен быть > 0")
    if len(rankings) != len(reference) or not rankings:
        raise ValueError("Списки ранжирований разной длины")
    total = 0.0
    for got, exact in zip(rankings, reference):
        total += len(set(got[:k]) & set(exact[:k])) / k
    return total / len(rankings)


def gold_errors(
    questions: list[dict[str, Any]],
    diploma: str,
    chunk_sets: dict[str, list[str]],
) -> list[str]:
    diploma_norm = normalize_text(diploma)
    errors: list[str] = []
    for question in questions:
        for item in question["evidence"]:
            quote = item["quote"]
            if normalize_text(quote) not in diploma_norm:
                errors.append(f"{question['id']}: цитаты нет в diploma.md: {quote[:80]}")
                continue
            for run_id, chunks in chunk_sets.items():
                if not any(covers(quote, text) for text in chunks):
                    errors.append(f"{question['id']}: цитату не покрывает {run_id}: {quote[:80]}")
    return errors


def load_questions(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"В {path} нет вопросов")
    for question in questions:
        if not question.get("id") or not question.get("question") or not question.get("evidence"):
            raise ValueError(f"Неполный вопрос: {question.get('id')}")
        for item in question["evidence"]:
            if not item.get("quote"):
                raise ValueError(f"{question['id']}: пустая цитата")
    source = payload.get("source", "src/preprocessing/output/diploma.md")
    return questions, source


def load_chunk_texts(path: Path) -> list[str]:
    texts = [
        json.loads(line)["text"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not texts:
        raise ValueError(f"Пустой файл чанков: {path}")
    return texts


def index_overrides(build: dict[str, Any], ef_search: int) -> list[str]:
    return [
        f"index.type={build['type']}",
        f"index.quantization={build['quantization']}",
        f"index.hnsw.M={build['M']}",
        f"index.hnsw.ef_construction={build['ef_construction']}",
        f"index.hnsw.ef_search={ef_search}",
    ]


def round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(metrics[key], 6) for key in METRIC_KEYS}


def build_indexes(config_path: str | None, run_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    specs = []
    for run_id in run_ids:
        for build in BUILDS:
            config = load_config(config_path, index_overrides(build, EF_SEARCH[1]))
            path = build_collection(config, run_id, activate=False)
            specs.append({"run_id": run_id, "build": build, "path": path, "config": config})
    return specs


def embed_questions(config: Config, questions: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    from index.embedders import ChunkEmbeddings

    print(f"Считаю эмбеддинги {len(questions)} вопросов, модель {config.embedding.model}")
    started = time.perf_counter()
    embedder = ChunkEmbeddings(config.embedding)
    vectors = embedder.embed_queries([question["question"] for question in questions])
    elapsed = time.perf_counter() - started
    print(f"Запросы: {vectors.shape}, device={embedder.device}, {elapsed:.1f} с")
    return np.ascontiguousarray(vectors, dtype=np.float32), elapsed


def run_manifest(config: Config, run_id: str) -> dict[str, Any]:
    path = config.artifacts_dir / run_id / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Нет манифеста прогона: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_specs(
    config: Config,
    questions: list[dict[str, Any]],
    queries: np.ndarray,
    specs: list[dict[str, Any]],
    run_ids: tuple[str, ...] = RUN_IDS,
) -> list[dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        manifest = run_manifest(config, run_id)
        vectorization = manifest["config"]["embedding"]["vectorization"]
        splitter = manifest["config"]["splitter"].get("type", "markdown_header")
        retrieval = "hybrid" if vectorization == "hybrid" else "dense"
        if vectorization not in {"dense", "hybrid"}:
            raise ValueError(f"{run_id}: vectorization={vectorization}, для FAISS нужен dense или hybrid")
        chunks = load_chunk_texts(config.artifacts_dir / run_id / "chunks.jsonl")
        bm25 = None
        if retrieval == "hybrid":
            bm25_path = config.artifacts_dir / run_id / "bm25.pkl"
            if not bm25_path.is_file():
                raise FileNotFoundError(f"Нет {bm25_path}")
            bm25 = load_bm25(bm25_path)
        prepared[run_id] = {
            "splitter": splitter,
            "retrieval": retrieval,
            "chunks": chunks,
            "hits": [evidence_hits(chunks, question["evidence"]) for question in questions],
            "bm25": bm25,
        }

    rows: list[dict[str, Any]] = []
    question_texts = [question["question"] for question in questions]
    for run_id in run_ids:
        info = prepared[run_id]
        run_specs = [spec for spec in specs if spec["run_id"] == run_id]
        run_specs.sort(key=lambda spec: spec["build"]["type"] != "flat")
        flat_rankings: list[list[int]] | None = None
        for spec in run_specs:
            build = spec["build"]
            index = faiss.read_index(str(spec["path"] / INDEX_FILE))
            ef_values: tuple[int | None, ...] = (None,) if build["type"] == "flat" else EF_SEARCH
            for ef_search in ef_values:
                if ef_search is not None:
                    spec["config"].index.hnsw.ef_search = ef_search
                    set_ef_search(index, spec["config"])
                depth = dense_depth(build["type"], ef_search, len(info["chunks"]), min_k=KS[-1])
                rank_indices(
                    index, queries, question_texts, info["bm25"], depth, config.rag.rrf_k
                )  # прогрев: первый вызов FAISS заметно медленнее
                started = time.perf_counter()
                rankings, _scores = rank_indices(
                    index, queries, question_texts, info["bm25"], depth, config.rag.rrf_k
                )
                search_ms = (time.perf_counter() - started) * 1000 / len(questions)
                if build["type"] == "flat":
                    flat_rankings = rankings
                if flat_rankings is None:
                    raise RuntimeError(f"Для {run_id} нет flat-ранжирования")
                per_question = [
                    {"id": question["id"], **round_metrics(ranking_metrics(ranking, hits))}
                    for question, hits, ranking in zip(questions, info["hits"], rankings)
                ]
                summary = round_metrics(macro_average(per_question))
                rows.append({
                    "run_id": run_id,
                    "splitter": info["splitter"],
                    "retrieval": info["retrieval"],
                    "n_chunks": len(info["chunks"]),
                    "index_type": build["type"],
                    "M": build["M"],
                    "ef_construction": build["ef_construction"],
                    "ef_search": ef_search,
                    "quantization": build["quantization"],
                    "index_bytes": (spec["path"] / INDEX_FILE).stat().st_size,
                    "search_ms": round(search_ms, 3),
                    "flat_recall@10": round(top_overlap(rankings, flat_rankings), 6),
                    **summary,
                    "per_question": per_question,
                })
                print(
                    f"{info['splitter']}/{info['retrieval']} {index_title(rows[-1])}: "
                    f"R@10={summary['recall@10']:.3f} MRR={summary['mrr']:.3f}"
                )
    return rows


def evaluate(config_path: str | None = None, questions_path: str | None = None) -> None:
    config = load_config(config_path)
    run_ids = discover_run_ids(config.artifacts_dir)
    questions_file = Path(questions_path) if questions_path else config.root / "eval" / "questions.json"
    questions, source = load_questions(questions_file)
    diploma = (config.root / source).read_text(encoding="utf-8")
    chunk_sets = {
        run_id: load_chunk_texts(config.artifacts_dir / run_id / "chunks.jsonl")
        for run_id in run_ids
    }
    errors = gold_errors(questions, diploma, chunk_sets)
    if errors:
        raise ValueError("Эталон не покрыт чанками:\n" + "\n".join(errors))
    print(f"Эталон: {len(questions)} вопросов, цитаты покрыты прогонами: {run_ids}")

    specs = build_indexes(config_path, run_ids)

    queries, embed_seconds = embed_questions(config, questions)
    rows = evaluate_specs(config, questions, queries, specs, run_ids)
    out_dir = config.root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "questions": questions_file.name,
        "n_questions": len(questions),
        "k": list(KS),
        "coverage_threshold": COVERAGE,
        "rrf_k": config.rag.rrf_k,
        "query_embed_seconds": round(embed_seconds, 3),
        "rows": rows,
    }
    metrics_path = out_dir / "metrics.json"
    report_path = out_dir / "report.md"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(
        render_report(rows, questions, embed_seconds, COVERAGE, config.rag.rrf_k),
        encoding="utf-8",
    )
    print(f"Метрики: {metrics_path}")
    print(f"Отчёт: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Метрики retrieval (п.5)")
    parser.add_argument("--config", default=None, help="путь к config.yaml")
    parser.add_argument("--questions", default=None, help="путь к eval/questions.json")
    args = parser.parse_args()
    evaluate(args.config, args.questions)


if __name__ == "__main__":
    main()
