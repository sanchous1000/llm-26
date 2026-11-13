import json
import unittest
from pathlib import Path
from unittest.mock import patch

from analyze_results import (
    canonical_content,
    evaluate_p1,
    evaluate_p2,
    evaluate_p3,
    extract_json,
    variability_metrics,
)
from run_experiment import build_payload, warm_up_model


HERE = Path(__file__).resolve().parent


class PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        cls.prompt = cls.config["prompts"][0]

    def test_baseline_has_no_sampling_hyperparameters(self) -> None:
        payload = build_payload("test-model", self.prompt, "baseline", self.config["tuned_parameters"])
        self.assertEqual(set(payload), {"model", "messages"})

    def test_tuned_changes_at_least_two_hyperparameters(self) -> None:
        payload = build_payload("test-model", self.prompt, "tuned", self.config["tuned_parameters"])
        changed = {"temperature", "top_p", "max_tokens"} & set(payload)
        self.assertGreaterEqual(len(changed), 2)

    @patch("run_experiment.request_json")
    def test_warmup_is_separate_minimal_request(self, request_json_mock) -> None:
        warm_up_model("http://127.0.0.1:11434/v1", "test-model", 30.0)
        url, payload, timeout = request_json_mock.call_args.args
        self.assertEqual(url, "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["max_tokens"], 1)
        self.assertEqual(timeout, 30.0)


class EvaluationTests(unittest.TestCase):
    def test_extracts_json_from_markdown_fence(self) -> None:
        self.assertEqual(extract_json('```json\n{"A": "Оплата"}\n```'), {"A": "Оплата"})

    def test_perfect_classification_scores_one(self) -> None:
        answer = json.dumps(
            {"A": "Оплата", "B": "Техподдержка", "C": "Продажи", "D": "Аккаунт"},
            ensure_ascii=False,
        )
        self.assertEqual(evaluate_p2(answer)["score"], 1.0)

    def test_markdown_fence_is_reported_as_format_error(self) -> None:
        answer = '```json\n{"A":"Оплата","B":"Техподдержка","C":"Продажи","D":"Аккаунт"}\n```'
        result = evaluate_p2(answer)
        self.assertIn("формат: только JSON", result["failed_checks"])
        self.assertLess(result["score"], 1.0)

    def test_perfect_extraction_scores_one(self) -> None:
        answer = json.dumps(
            {
                "name": "Nordic Kettle X2",
                "volume_l": 1.7,
                "power_w": 2200,
                "materials": ["боросиликатное стекло", "нержавеющая сталь"],
                "features": ["автоотключение", "защита от включения без воды"],
                "price_rub": 4590,
                "warranty_months": 24,
                "smartphone_control": False,
                "summary": "Стеклянный чайник с двумя функциями безопасности.",
            },
            ensure_ascii=False,
        )
        self.assertEqual(evaluate_p3(answer)["score"], 1.0)

    def test_extraction_rejects_wrong_steel_type(self) -> None:
        answer = json.dumps(
            {
                "name": "Nordic Kettle X2",
                "volume_l": 1.7,
                "power_w": 2200,
                "materials": ["боросиликатное стекло", "никелиевая сталь"],
                "features": ["автоотключение", "защита от включения без воды"],
                "price_rub": 4590,
                "warranty_months": 24,
                "smartphone_control": False,
                "summary": "Стеклянный чайник с двумя функциями безопасности.",
            },
            ensure_ascii=False,
        )
        self.assertFalse(evaluate_p3(answer)["checks"]["materials"])

    def test_generation_evaluation_counts_words_and_facts(self) -> None:
        answer = (
            "Практика локальных LLM\n\n15 ноября 2026 года, Кронверкский проспект, 49. "
            "Участие бесплатно, регистрация до 10 ноября, доступно 150 мест."
        )
        result = evaluate_p1(answer)
        self.assertEqual(sum(result["checks"].values()), 7)

    def test_generation_evaluation_rejects_multiple_body_paragraphs(self) -> None:
        answer = "Заголовок\n\nПервый абзац.\n\nВторой абзац."
        self.assertFalse(evaluate_p1(answer)["checks"]["заголовок и один абзац"])

    def test_json_variability_ignores_formatting(self) -> None:
        plain = '{"A":"Оплата"}'
        fenced = '```json\n{\n  "A": "Оплата"\n}\n```'
        self.assertEqual(canonical_content(plain), canonical_content(fenced))

    def test_variability_detects_identical_repeats(self) -> None:
        records = [
            {
                "response": "одинаковый ответ",
                "usage": {"completion_tokens": 2},
                "metrics": {"wall_time_s": time},
                "finish_reason": "length" if time == 1.1 else "stop",
            }
            for time in (1.0, 1.1, 0.9)
        ]
        metrics = variability_metrics(records)
        self.assertEqual(metrics["unique_texts"], 1)
        self.assertEqual(metrics["exact_match_rate"], 1.0)
        self.assertEqual(metrics["token_cv"], 0.0)
        self.assertEqual(metrics["length_capped_runs"], 1)


if __name__ == "__main__":
    unittest.main()
