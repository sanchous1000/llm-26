"""Обслуживание результатов: пересчёт метрик и склейка прогонов.

Запуск: python source/main.py test
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from llm_lab.inference.speed import METRICS_VERSION
from llm_lab.maintenance.merge_runs import Part, cleanup, merge, missing_cells
from llm_lab.maintenance.recompute_metrics import classify_v1_timing, migrate_record
from tests.fixtures import make_record


class RecomputeMetricsTest(unittest.TestCase):
    """Цифры взяты из реальной записи qwen3.5-35b-a3b, P1__A_baseline__r1."""

    def _v1_record(self, **metrics: Any) -> dict[str, Any]:
        base = {
            "content": "ответ", "reasoning": "мысли",
            "latency_s": 47.474, "ttft_s": 44.947, "completion_tokens": 2638,
            "output_tps": 1043.91, "chars": 501, "reasoning_chars": 8350,
        }
        return {"run_id": "P1__A_baseline__r1", "stream": True, "error": "", "metrics": {**base, **metrics}}

    def test_separate_reasoning_stream_is_detected(self) -> None:
        ttft, ttfa, label = classify_v1_timing(self._v1_record()["metrics"], stream=True)
        self.assertEqual(label, "separate_reasoning")
        self.assertIsNone(ttft)
        self.assertEqual(ttfa, 44.947)

    def test_inline_reasoning_keeps_ttft(self) -> None:
        metrics = self._v1_record(ttft_s=0.4)["metrics"]
        ttft, ttfa, label = classify_v1_timing(metrics, stream=True)
        self.assertEqual((ttft, ttfa, label), (0.4, None, "inline_reasoning"))

    def test_migration_fixes_absurd_speed(self) -> None:
        migrated, label = migrate_record(self._v1_record())
        speed = migrated["metrics"]["output_tps"]
        self.assertEqual(label, "separate_reasoning")
        self.assertLess(speed, 60.0)
        self.assertGreater(speed, 50.0)
        self.assertNotIn("content", migrated["metrics"])
        self.assertEqual(migrated["metrics"]["metrics_version"], METRICS_VERSION)

    def test_migration_does_not_mutate_input(self) -> None:
        original = self._v1_record()
        snapshot = json.dumps(original, sort_keys=True)
        migrate_record(original)
        self.assertEqual(json.dumps(original, sort_keys=True), snapshot)

    def test_migration_is_idempotent(self) -> None:
        once, _ = migrate_record(self._v1_record())
        twice, label = migrate_record(once)
        self.assertEqual(label, "up_to_date")
        self.assertEqual(once, twice)

    def test_no_reasoning_model_keeps_decode_speed(self) -> None:
        record = self._v1_record(
            latency_s=7.053, ttft_s=1.52, completion_tokens=155, chars=468, reasoning_chars=0,
        )
        migrated, label = migrate_record(record)
        self.assertEqual(label, "no_reasoning")
        self.assertAlmostEqual(migrated["metrics"]["output_tps"], 28.01, places=1)


class MergeRunsTest(unittest.TestCase):
    def test_later_part_wins_and_duplicate_is_recorded(self) -> None:
        first = Part(Path("runs_1.jsonl"), (
            make_record("P1", "A_baseline", 1, "2026-01-01"),
            make_record("P5", "A_baseline", 1, "2026-01-01", answer="старый"),
        ))
        second = Part(Path("runs_5.jsonl"), (
            make_record("P5", "A_baseline", 1, "2026-01-02", answer="новый"),
        ))
        records, dropped, sources = merge((first, second))
        p5 = next(r for r in records if r["prompt_id"] == "P5")
        self.assertEqual(p5["answer"], "новый")
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["source"], "runs_1.jsonl")
        self.assertEqual(dropped[0]["answer"], "старый")
        self.assertEqual(sources["P5__A_baseline__r1"], "runs_5.jsonl")

    def test_records_sorted_in_canonical_order(self) -> None:
        part = Part(Path("runs.jsonl"), (
            make_record("P6", "A_baseline", 1, "t"),
            make_record("P1", "G_budget", 1, "t"),
            make_record("P1", "A_baseline", 2, "t"),
            make_record("P1", "A_baseline", 1, "t"),
        ))
        records, _, _ = merge((part,))
        self.assertEqual(
            [r["run_id"] for r in records],
            ["P1__A_baseline__r1", "P1__A_baseline__r2", "P1__G_budget__r1", "P6__A_baseline__r1"],
        )

    def test_missing_cells_reported(self) -> None:
        records = (make_record("P1", "A_baseline", 1, "t"),)
        self.assertEqual(missing_cells(records), ["P1__A_baseline__r2"])

    def test_cleanup_removes_only_suffixed_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            for name in ("runs.jsonl", "runs_5.jsonl", "report_5.md", "manifest_5.json", "report.md"):
                (model_dir / name).write_text("x", encoding="utf-8")
            (model_dir / "answers_5").mkdir()
            (model_dir / "answers_5" / "a.md").write_text("x", encoding="utf-8")
            parts = (Part(model_dir / "runs_5.jsonl", ()), Part(model_dir / "runs.jsonl", ()))

            removed = cleanup(model_dir, parts)

            self.assertEqual(sorted(removed), ["answers_5", "manifest_5.json", "report_5.md", "runs_5.jsonl"])
            self.assertTrue((model_dir / "runs.jsonl").exists())
            self.assertTrue((model_dir / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
