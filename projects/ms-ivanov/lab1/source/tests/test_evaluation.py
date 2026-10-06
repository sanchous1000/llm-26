"""Оценка ответов: разбор текста и автопроверки.

Запуск: python source/main.py test
"""

from __future__ import annotations

import unittest

from llm_lab.evaluation.checks import (
    HALLUCINATION_MARKER,
    ContainsAll,
    JsonSchemaCheck,
    LabelsMatch,
    NotContainsAny,
    NumbersFromSource,
    NumericAnswer,
    RefusalExpected,
    WordBudget,
    aggregate_score,
)
from llm_lab.evaluation.text import (
    extract_json_object,
    extract_numbers,
    has_visible_answer,
    mean_pairwise_similarity,
    split_reasoning,
    word_count,
)


class TextUtilsTest(unittest.TestCase):
    def test_split_reasoning_removes_think_block(self) -> None:
        answer, reasoning = split_reasoning("<think>сначала посчитаю</think>Итого: 1431 ₽")
        self.assertEqual(answer, "Итого: 1431 ₽")
        self.assertEqual(reasoning, "сначала посчитаю")

    def test_unclosed_think_block_leaves_no_answer(self) -> None:
        answer, reasoning = split_reasoning("<think>рассуждение оборвалось по лимиту")
        self.assertEqual(answer, "")
        self.assertIn("оборвалось", reasoning)

    def test_plain_answer_untouched(self) -> None:
        self.assertEqual(split_reasoning("обычный ответ"), ("обычный ответ", ""))

    def test_extract_numbers_handles_separators(self) -> None:
        self.assertEqual(extract_numbers("не прошло 3 204 платежа, это 1,5 %"), (3204.0, 1.5))

    def test_extract_json_from_markdown_fence(self) -> None:
        parsed, error = extract_json_object('Вот результат:\n```json\n{"a": 1}\n```')
        self.assertEqual(error, "")
        self.assertEqual(parsed, {"a": 1})

    def test_extract_json_with_trailing_text(self) -> None:
        parsed, _ = extract_json_object('{"a": 1, "b": "х"} — готово')
        self.assertEqual(parsed, {"a": 1, "b": "х"})

    def test_extract_json_reports_error(self) -> None:
        parsed, error = extract_json_object("совсем не json")
        self.assertIsNone(parsed)
        self.assertTrue(error)

    def test_similarity_of_identical_answers(self) -> None:
        self.assertEqual(mean_pairwise_similarity(("a b c", "a b c")), 1.0)

    def test_word_count(self) -> None:
        self.assertEqual(word_count("одно два три"), 3)


class ChecksTest(unittest.TestCase):
    def test_contains_all_partial_score(self) -> None:
        result = ContainsAll(values=("альфа", "бета")).evaluate("только альфа")
        self.assertAlmostEqual(result.score, 0.5)
        self.assertFalse(result.passed)

    def test_word_budget_penalises_overflow(self) -> None:
        over = WordBudget(max_words=10).evaluate(" ".join(["сл"] * 20))
        self.assertLess(over.score, 1.0)
        exact = WordBudget(max_words=10).evaluate(" ".join(["сл"] * 10))
        self.assertEqual(exact.score, 1.0)

    def test_not_contains_marks_hallucination(self) -> None:
        result = NotContainsAny(values=("штраф",)).evaluate("начислен штраф 100 рублей")
        self.assertEqual(result.score, 0.0)
        self.assertIn(HALLUCINATION_MARKER, result.detail)

    def test_labels_match_counts_accuracy(self) -> None:
        check = LabelsMatch(expected=("Billing", "Sales"), allowed=("Billing", "Sales", "Other"))
        good = check.evaluate("1: Billing\n2: Sales")
        self.assertEqual(good.score, 1.0)
        half = check.evaluate("1: Billing\n2: Other")
        self.assertAlmostEqual(half.score, 0.5)

    def test_labels_match_tolerates_formatting(self) -> None:
        check = LabelsMatch(expected=("Billing",), allowed=("Billing",))
        self.assertEqual(check.evaluate("1. billing").score, 1.0)

    def test_json_schema_detects_invented_fields(self) -> None:
        check = JsonSchemaCheck(
            required_fields=("a", "price"),
            expected_values={"a": "x"},
            null_fields=("price",),
        )
        clean = check.evaluate('{"a": "x", "price": null}')
        self.assertEqual(clean.score, 1.0)
        dirty = check.evaluate('{"a": "x", "price": 18900}')
        self.assertLess(dirty.score, 1.0)
        self.assertIn(HALLUCINATION_MARKER, dirty.detail)

    def test_json_schema_zero_on_broken_json(self) -> None:
        check = JsonSchemaCheck(required_fields=("a",))
        self.assertEqual(check.evaluate("не json").score, 0.0)

    def test_numbers_from_source_flags_invented(self) -> None:
        check = NumbersFromSource(source="было 40 соединений из 400")
        self.assertEqual(check.evaluate("лимит 40 вместо 400").score, 1.0)
        bad = check.evaluate("пострадало 9 500 платежей")
        self.assertLess(bad.score, 1.0)
        self.assertIn(HALLUCINATION_MARKER, bad.detail)

    def test_numeric_answer_grades_final_value(self) -> None:
        check = NumericAnswer(expected=1431.0)
        self.assertEqual(check.evaluate("Итого: 1431 ₽").score, 1.0)
        self.assertEqual(check.evaluate("Итого: 1530 ₽").score, 0.0)
        partial = check.evaluate("1431 без скидки, но Итого: 1590")
        self.assertAlmostEqual(partial.score, 0.6)

    def test_refusal_expected_rewards_honesty(self) -> None:
        check = RefusalExpected(
            markers=("не существует",),
            fabrication_markers=("статья 14.7 устанавливает",),
        )
        honest = check.evaluate("Такого закона не существует, подтвердить не могу.")
        self.assertEqual(honest.score, 1.0)
        invented = check.evaluate("Статья 14.7 устанавливает обязанность регистрации двойников.")
        self.assertEqual(invented.score, 0.0)
        self.assertIn(HALLUCINATION_MARKER, invented.detail)

    def test_aggregate_score_respects_weights(self) -> None:
        heavy = NumericAnswer(expected=1.0, weight=3.0).evaluate("Итого: 1")
        light = WordBudget(max_words=1, weight=1.0).evaluate("много лишних слов тут")
        score = aggregate_score((heavy, light))
        self.assertGreater(score, 0.7)


class VisibleAnswerTest(unittest.TestCase):
    def test_detects_answer_after_inline_think(self) -> None:
        self.assertFalse(has_visible_answer("<think>ещё думаю"))
        self.assertFalse(has_visible_answer("<think>думал</think>   "))
        self.assertTrue(has_visible_answer("<think>думал</think>Итого"))
        self.assertTrue(has_visible_answer("Итого: 1431"))


if __name__ == "__main__":
    unittest.main()
