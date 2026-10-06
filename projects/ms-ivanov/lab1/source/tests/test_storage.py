"""Хранение прогонов: JSONL и карточки ответов.

Запуск: python source/main.py test
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_lab.storage import ModelPaths, load_runs, rewrite_answers, write_runs
from tests.fixtures import make_record


class StorageRoundTripTest(unittest.TestCase):
    def test_write_runs_and_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = ModelPaths(root=Path(tmp), model_id="org/model")
            records = (make_record("P1", "A_baseline", 1, "t"), make_record("P1", "A_baseline", 2, "t"))
            write_runs(paths.runs_file, records)
            self.assertEqual(load_runs(paths.runs_file), records)
            self.assertEqual(rewrite_answers(paths, records), 2)
            self.assertEqual(len(list(paths.answers_dir.glob("*.md"))), 2)

    def test_write_runs_refuses_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_runs(Path(tmp) / "runs.jsonl", ())


if __name__ == "__main__":
    unittest.main()
