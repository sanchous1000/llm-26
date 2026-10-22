"""Агрегация статистики и сборка отчёта по модели.

Запуск: python source/main.py test
"""

from __future__ import annotations

import unittest

from llm_lab.evaluation.checks import HALLUCINATION_MARKER
from llm_lab.reporting.model_report import build_report
from llm_lab.reporting.stats import has_hallucination, stability, summarize


def _record(prompt_id: str, preset: str, repeat: int, answer: str, score: float, tokens: int) -> dict:
    return {
        "prompt_id": prompt_id,
        "preset_key": preset,
        "preset_label": preset,
        "mode": "tuned",
        "repeat": repeat,
        "answer": answer,
        "quality_score": score,
        "error": "",
        "checks": [{"name": "c", "score": score, "passed": score >= 1.0, "detail": "ok"}],
        "metrics": {
            "completion_tokens": tokens,
            "prompt_tokens": 100,
            "latency_s": 1.0,
            "ttft_s": None,
            "output_tps": 10.0,
            "truncated": False,
            "words": 10,
            "repeat_ngram_ratio": 0.0,
            "tokens_estimated": False,
        },
    }


class StatsTest(unittest.TestCase):
    def test_summarize_excludes_failed_runs(self) -> None:
        failed = _record("P1", "B", 2, "", 0.0, 0)
        failed["error"] = "timeout"
        stats = summarize("B", [_record("P1", "B", 1, "ок", 1.0, 50), failed])
        self.assertEqual(stats.runs, 2)
        self.assertEqual(stats.errors, 1)
        self.assertEqual(stats.quality, 1.0)

    def test_stability_detects_determinism(self) -> None:
        same = [_record("P1", "B", i, "один и тот же", 1.0, 50) for i in (1, 2, 3)]
        result = stability(same)
        self.assertTrue(result.identical)
        self.assertEqual(result.distinct_answers, 1)
        self.assertEqual(result.mean_similarity, 1.0)

    def test_stability_detects_variance(self) -> None:
        varied = [
            _record("P1", "C", 1, "первый вариант", 1.0, 50),
            _record("P1", "C", 2, "совершенно другой текст", 0.5, 80),
        ]
        result = stability(varied)
        self.assertFalse(result.identical)
        self.assertEqual(result.tokens_spread, 30)
        self.assertAlmostEqual(result.quality_spread, 0.5)

    def test_has_hallucination_reads_check_details(self) -> None:
        record = _record("P3", "A", 1, "x", 0.5, 10)
        self.assertFalse(has_hallucination(record))
        record["checks"][0]["detail"] = f"{HALLUCINATION_MARKER} — выдумано"
        self.assertTrue(has_hallucination(record))


class ReportSmokeTest(unittest.TestCase):
    def test_report_builds_from_minimal_records(self) -> None:
        records = tuple(
            _record("P1", preset, repeat, f"текст {preset}{repeat}", 0.9, 40)
            for preset in ("A_baseline", "B_precise")
            for repeat in (1, 2)
        )
        for record in records:
            record.update({
                "model": "test-model",
                "prompt_title": "Тест",
                "prompt_type": "generation",
                "params_desc": "temperature=0.2",
                "timestamp": "2026-09-14T00:00:00+00:00",
                "repeats_total": 2,
                "system": "s",
                "user": "u",
                "reference": "эталон",
                "stream": False,
            })
        markdown = build_report(records, {})
        self.assertIn("# Отчёт по модели `test-model`", markdown)
        self.assertIn("## 5. Стабильность и детерминизм", markdown)
        self.assertIn("## 9. Автоматические наблюдения", markdown)


if __name__ == "__main__":
    unittest.main()
