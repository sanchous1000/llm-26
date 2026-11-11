"""Проверки покрытия цитат и формул Recall/Precision/MRR/RRF."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL.parent / "src"))
sys.path.insert(0, str(_EVAL))

from evaluate import (
    RUN_IDS,
    covers,
    discover_run_ids,
    evidence_hits,
    gold_errors,
    load_chunk_texts,
    load_questions,
    macro_average,
    ranking_metrics,
    top_overlap,
)
from langchain_core.documents import Document

from config import Config, load_config
from index.embedders import build_bm25
from index.load_to_vector_store import bm25_rankings, rrf_fuse
from index.splitters import split_documents

ROOT = _EVAL.parent


class CoverTests(unittest.TestCase):
    def test_substring_ignores_whitespace_and_yo(self) -> None:
        self.assertTrue(covers("Учёт  данных", "перед учет\nданных после"))

    def test_empty_quote_is_not_covered(self) -> None:
        self.assertFalse(covers("", "текст"))
        self.assertFalse(covers("текст", ""))

    def test_contiguous_window_accepts_a_split_tail(self) -> None:
        quote = "один два три четыре пять шесть семь восемь девять десять"
        self.assertTrue(covers(quote, "хвост пять шесть семь восемь девять десять"))
        self.assertFalse(covers(quote, "один два три четыре пять"))

    def test_bag_of_words_does_not_count(self) -> None:
        quote = "альфа бета гамма дельта эпсилон зета эта тета"
        shuffled = "тета эта зета эпсилон дельта гамма бета альфа"
        self.assertFalse(covers(quote, shuffled))
        self.assertTrue(covers(quote, "префикс альфа бета гамма дельта эпсилон хвост"))


class MetricTests(unittest.TestCase):
    def test_two_passages_in_one_chunk(self) -> None:
        metrics = ranking_metrics([0, 1, 2, 3, 4], [{0}, {0}])
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["precision@5"], 0.2)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_partial_recall_and_reciprocal_rank(self) -> None:
        metrics = ranking_metrics([0, 1, 2, 3, 4, 5], [{1}, {5}])
        self.assertEqual(metrics["recall@5"], 0.5)
        self.assertEqual(metrics["recall@10"], 1.0)
        self.assertEqual(metrics["precision@5"], 0.2)
        self.assertEqual(metrics["precision@10"], 0.2)
        self.assertAlmostEqual(metrics["mrr"], 0.5)

    def test_miss_is_zero(self) -> None:
        metrics = ranking_metrics([0, 1], [{3}])
        self.assertEqual(metrics["mrr"], 0.0)
        self.assertEqual(metrics["recall@5"], 0.0)
        self.assertEqual(metrics["precision@5"], 0.0)

    def test_macro_average(self) -> None:
        averaged = macro_average([
            {"recall@5": 1.0, "recall@10": 1.0, "precision@5": 0.2, "precision@10": 0.1, "mrr": 1.0},
            {"recall@5": 0.0, "recall@10": 0.5, "precision@5": 0.0, "precision@10": 0.1, "mrr": 0.0},
        ])
        self.assertAlmostEqual(averaged["recall@5"], 0.5)
        self.assertAlmostEqual(averaged["mrr"], 0.5)

    def test_rrf_prefers_the_shared_top(self) -> None:
        self.assertEqual(rrf_fuse([[2, 0], [2]]), [2, 0])

    def test_rrf_tie_breaks_by_index(self) -> None:
        self.assertEqual(rrf_fuse([[0], [1]]), [0, 1])

    def test_top_overlap(self) -> None:
        score = top_overlap([[0, 1, 2]], [[0, 2, 3]], k=2)
        self.assertAlmostEqual(score, 0.5)

    def test_evidence_hits_follow_covers(self) -> None:
        hits = evidence_hits(["альфа бета", "гамма"], [{"quote": "альфа бета"}])
        self.assertEqual(hits, [{0}])


class GoldFileTests(unittest.TestCase):
    def test_quotes_are_in_diploma_and_both_chunk_sets(self) -> None:
        questions, source = load_questions(ROOT / "eval" / "questions.json")
        self.assertGreaterEqual(len(questions), 10)
        self.assertLessEqual(len(questions), 20)
        diploma = (ROOT / source).read_text(encoding="utf-8")
        chunk_sets = {}
        for run_id in discover_run_ids(ROOT / "artifacts"):
            path = ROOT / "artifacts" / run_id / "chunks.jsonl"
            if not path.is_file():
                self.skipTest(f"нет {path}")
            chunk_sets[run_id] = load_chunk_texts(path)
        self.assertEqual(gold_errors(questions, diploma, chunk_sets), [])


class SplitterBudgetTests(unittest.TestCase):
    @staticmethod
    def _word_len(text: str) -> int:
        return len(text.split())

    def _embed_len(self, config: Config):
        prefix = config.embedding.document_prefix
        return lambda text: len((prefix + text).split()) + 2  # +2 спецтокена (BOS/EOS)

    def test_nested_headings_prepended_without_duplicate_markdown_header(self) -> None:
        config = Config.model_validate(
            {"splitter": {"chunk_size": 50, "chunk_overlap": 5}}
        )
        page = Document(
            page_content="# Глава 1\n## Раздел 1.1\n### Подраздел 1.1.1\nПервый абзац текста.",
            metadata={"source": "doc.pdf", "page": 1},
        )
        chunks = split_documents(
            [page], config, length_fn=self._word_len, embed_length_fn=self._embed_len(config)
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].page_content,
            "Глава 1 > Раздел 1.1 > Подраздел 1.1.1\n\nПервый абзац текста.",
        )
        self.assertNotIn("###", chunks[0].page_content)

    def test_header_only_page_is_preserved(self) -> None:
        config = Config.model_validate(
            {"splitter": {"chunk_size": 50, "chunk_overlap": 5}}
        )
        page = Document(
            page_content="## Титульный лист\n### Университет ИТМО",
            metadata={"source": "doc.pdf", "page": 1, "section": "Титульный лист"},
        )
        chunks = split_documents(
            [page], config, length_fn=self._word_len, embed_length_fn=self._embed_len(config)
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("Университет ИТМО", chunks[0].page_content)

    def test_chunks_strictly_fit_chunk_size_and_embedder_window(self) -> None:
        config = Config.model_validate(
            {
                "splitter": {
                    "chunk_size": 20,
                    "chunk_overlap": 3,
                },
                "embedding": {"max_seq_length": 22, "document_prefix": "search_document: "},
            }
        )
        body = " ".join(f"слово{i}" for i in range(60))
        page = Document(
            page_content=f"# Глава первая\n## Раздел второй\n### Подраздел третий\n{body}",
            metadata={"source": "doc.pdf", "page": 2},
        )
        chunks = split_documents(
            [page], config, length_fn=self._word_len, embed_length_fn=self._embed_len(config)
        )
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertTrue(
                chunk.page_content.startswith("Глава первая > Раздел второй > Подраздел третий\n\n")
            )
            self.assertLessEqual(chunk.metadata["n_tokens"], config.splitter.chunk_size)
            self.assertLessEqual(chunk.metadata["embed_tokens"], config.embedding.max_seq_length)

    def test_h1_carries_across_pages_as_separate_chunks(self) -> None:
        config = Config.model_validate(
            {"splitter": {"chunk_size": 50, "chunk_overlap": 5}}
        )
        pages = [
            Document(
                page_content="# ГЛАВА 2 Сравнительный анализ\nВводный текст главы.\n## 2.1 Показатели\nТекст 2.1.",
                metadata={"source": "doc.pdf", "page": 32},
            ),
            Document(
                page_content="Продолжение 2.1.\n## 2.2 Модели\nТекст 2.2.",
                metadata={"source": "doc.pdf", "page": 33},
            ),
        ]
        chunks = split_documents(
            pages, config, length_fn=self._word_len, embed_length_fn=self._embed_len(config)
        )
        self.assertEqual(len(chunks), 4)
        section = "ГЛАВА 2 Сравнительный анализ > 2.1 Показатели"
        self.assertEqual(chunks[1].metadata["section"], section)
        self.assertEqual(chunks[1].metadata["page"], 32)
        self.assertIn("Текст 2.1.", chunks[1].page_content)
        self.assertNotIn("Продолжение 2.1.", chunks[1].page_content)
        self.assertEqual(chunks[2].metadata["section"], section)
        self.assertEqual(chunks[2].metadata["page"], 33)
        self.assertIn("Продолжение 2.1.", chunks[2].page_content)
        self.assertTrue(
            chunks[3].page_content.startswith("ГЛАВА 2 Сравнительный анализ > 2.2 Модели\n\n")
        )
        self.assertEqual(chunks[3].metadata["h1"], "ГЛАВА 2 Сравнительный анализ")
        self.assertEqual(chunks[3].metadata["h2"], "2.2 Модели")

    def test_bm25_ignores_zero_score_chunks_and_normalizes_yo(self) -> None:
        config = Config()
        bm25 = build_bm25(
            ["оценка надёжности системы", "совершенно другой текст", "третий нерелевантный чанк"],
            config,
        )
        rankings = bm25_rankings(bm25, ["надежности"], n_chunks=3)
        self.assertEqual(rankings, [[0]])


class ReportAndConfigTests(unittest.TestCase):
    @staticmethod
    def _row(
        splitter: str,
        retrieval: str,
        index_type: str,
        mrr: float,
        recall10: float,
        nbytes: int = 4096,
        ms: float = 0.2,
        q_recall: float = 1.0,
    ) -> dict:
        metrics = {
            "recall@5": recall10,
            "recall@10": recall10,
            "precision@5": 0.2,
            "precision@10": 0.1,
            "mrr": mrr,
        }
        per = {"id": "q1", **metrics, "recall@10": q_recall}
        return {
            "splitter": splitter,
            "retrieval": retrieval,
            "n_chunks": 10,
            "index_type": index_type,
            "M": 16,
            "ef_construction": 64,
            "ef_search": None if index_type == "flat" else 64,
            "quantization": "none",
            "index_bytes": nbytes,
            "search_ms": ms,
            "flat_recall@10": 1.0,
            **metrics,
            "per_question": [per],
        }

    def test_report_one_row_does_not_assume_two_runs(self) -> None:
        from report import render_report

        text = render_report(
            [self._row("markdown_header", "dense", "flat", 0.5, 1.0)],
            [{"id": "q1", "question": "Что это?"}],
            1.0,
            0.6,
            60,
        )
        self.assertIn("markdown_header, dense", text)
        self.assertNotIn("два готовых прогона", text)
        self.assertNotIn("обоих", text)
        self.assertIn("На flat у всех вопросов Recall@10 = 1.", text)

    def test_report_three_rows(self) -> None:
        from report import render_report

        rows = [
            self._row("markdown_header", "dense", "flat", 0.4, 0.5, q_recall=0.5),
            self._row("recursive", "dense", "flat", 0.8, 1.0, nbytes=2048),
            self._row("recursive", "hybrid", "hnsw", 0.9, 1.0, nbytes=1024, ms=0.05),
        ]
        text = render_report(rows, [{"id": "q1", "question": "Что это?"}], 1.2, 0.6, 60)
        self.assertIn("markdown_header, dense", text)
        self.assertIn("recursive, dense", text)
        self.assertIn("recursive, dense+BM25", text)
        self.assertIn("1/(60+rank)", text)
        self.assertIn("q1 Что это?", text)

    def test_lab_config_loads_and_removed_fields_fail(self) -> None:
        from pydantic import ValidationError

        config = load_config(ROOT / "config.yaml")
        self.assertEqual(config.embedding.vectorization, "dense")
        self.assertEqual(config.rag.rrf_k, 60)
        self.assertFalse(hasattr(config.index, "metric"))
        with self.assertRaises(ValidationError):
            Config.model_validate({"splitter": {"type": "recursive"}})
        with self.assertRaises(ValidationError):
            Config.model_validate({"splitter": {"include_headings": "both"}})
        with self.assertRaises(ValidationError):
            Config.model_validate({"embedding": {"vectorization": "sparse"}})
        with self.assertRaises(ValidationError):
            Config.model_validate({"index": {"metric": "cosine"}})
        with self.assertRaises(ValidationError):
            Config.model_validate({"rag": {"min_score": 0.2}})

    def test_collection_hash_keeps_cosine_key(self) -> None:
        import hashlib
        import json

        from index.load_to_vector_store import collection_hash

        config = Config()
        payload = {
            "type": config.index.type,
            "metric": "cosine",
            "M": config.index.hnsw.M,
            "ef_construction": config.index.hnsw.ef_construction,
            "quantization": config.index.quantization,
            "embeddings_sha256": "abc",
        }
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.assertEqual(collection_hash(config, "abc"), expected)

    def test_rank_indices_matches_dense_order_and_rrf(self) -> None:
        import faiss
        import numpy as np

        from index.load_to_vector_store import dense_rankings, rank_indices

        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(2)
        index.add(np.ascontiguousarray(vectors))
        queries = np.ascontiguousarray(np.array([[1.0, 0.0]], dtype=np.float32))

        orders, scores = rank_indices(index, queries, ["альфа"], None, depth=2)
        self.assertEqual(orders, dense_rankings(index, queries, 2))
        self.assertEqual(scores, [{}])

        bm25 = build_bm25(["альфа бета", "гамма", "альфа"], Config())
        orders, score_maps = rank_indices(index, queries, ["альфа"], bm25, depth=3, rrf_k=60)
        dense = dense_rankings(index, queries, 3)[0]
        sparse = bm25_rankings(bm25, ["альфа"], 3)[0]
        self.assertEqual(orders[0], rrf_fuse([dense, sparse], 60))
        self.assertIn(orders[0][0], score_maps[0])


class PromptWindowTests(unittest.TestCase):
    def test_prompt_fits_when_sum_equals_window(self) -> None:
        from server.llm import ensure_prompt_fits

        ensure_prompt_fits("Qwen/Qwen3-4B", 100, 700, 800)

    def test_prompt_one_token_over_raises(self) -> None:
        from server.llm import PromptTooLong, ensure_prompt_fits

        with self.assertRaises(PromptTooLong) as caught:
            ensure_prompt_fits("Qwen/Qwen3-4B", 101, 700, 800)
        message = str(caught.exception)
        self.assertIn("Qwen/Qwen3-4B", message)
        self.assertIn("101", message)
        self.assertIn("700", message)
        self.assertIn("800", message)

    def test_context_length_reads_max_model_len(self) -> None:
        from server.llm import context_length

        payload = {"data": [{"id": "Qwen/Qwen3-4B", "max_model_len": 8192}]}
        self.assertEqual(context_length(payload, "Qwen/Qwen3-4B"), 8192)

    def test_missing_max_model_len_cannot_confirm_window(self) -> None:
        from server.llm import LlmUnavailable, context_length

        payload = {"data": [{"id": "Qwen/Qwen3-4B"}]}
        with self.assertRaises(LlmUnavailable) as caught:
            context_length(payload, "Qwen/Qwen3-4B")
        self.assertIn("max_model_len", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
