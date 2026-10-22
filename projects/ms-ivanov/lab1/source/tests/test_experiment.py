"""Конфигурация эксперимента и набор промптов.

Запуск: python source/main.py test
"""

from __future__ import annotations

import unittest

from llm_lab.evaluation.checks import aggregate_score, evaluate_all
from llm_lab.experiment.config import PROFILES, resolve_presets, with_budget
from llm_lab.experiment.prompts import ALL_PROMPTS, P2, P3, P5, resolve_prompts
from llm_lab.inference.params import ConfigError, GenerationParams


class PromptSuiteTest(unittest.TestCase):
    """Эталонные ответы должны получать максимальный балл — защита от кривых проверок."""

    def _score(self, case, answer: str) -> float:
        return aggregate_score(evaluate_all(case.checks, answer))

    def test_reference_classification_scores_high(self) -> None:
        answer = "1: Billing\n2: Tech support\n3: Sales\n4: Other\n5: Billing\n6: Tech support"
        self.assertGreaterEqual(self._score(P2, answer), 0.99)

    def test_reference_extraction_scores_high(self) -> None:
        self.assertGreaterEqual(self._score(P3, P3.reference), 0.99)

    def test_reference_calculation_scores_high(self) -> None:
        answer = "Расчёт: 24 кг, 450 + 19×60 = 1590, минус 10 %.\nИтого: 1431 ₽"
        self.assertGreaterEqual(self._score(P5, answer), 0.99)

    def test_all_prompts_have_unique_ids_and_checks(self) -> None:
        ids = [case.id for case in ALL_PROMPTS]
        self.assertEqual(len(ids), len(set(ids)))
        for case in ALL_PROMPTS:
            with self.subTest(prompt=case.id):
                self.assertTrue(case.checks, "у промпта должны быть автопроверки")
                self.assertTrue(case.user.strip())

    def test_resolve_prompts_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            resolve_prompts(("P99",))


class ConfigTest(unittest.TestCase):
    def test_baseline_sends_no_parameters(self) -> None:
        self.assertEqual(GenerationParams().to_payload(), {})

    def test_extra_params_can_be_stripped(self) -> None:
        params = GenerationParams(temperature=0.2, top_k=20, repeat_penalty=1.1)
        self.assertEqual(params.to_payload(include_extra=False), {"temperature": 0.2})

    def test_safety_cap_only_fills_empty_max_tokens(self) -> None:
        self.assertEqual(GenerationParams().with_max_tokens_floor(256).max_tokens, 256)
        self.assertEqual(GenerationParams(max_tokens=64).with_max_tokens_floor(256).max_tokens, 64)

    def test_profiles_change_at_least_two_hyperparameters(self) -> None:
        for preset in resolve_presets("full", ()):
            if preset.mode != "tuned":
                continue
            with self.subTest(preset=preset.key):
                self.assertGreaterEqual(len(preset.changed_params), 2)

    def test_min_profile_covers_baseline_and_tuned(self) -> None:
        modes = {p.mode for p in resolve_presets("min", ())}
        self.assertEqual(modes, {"baseline", "tuned"})
        self.assertIn("min", PROFILES)

    def test_all_profiles_resolve(self) -> None:
        for name in PROFILES:
            with self.subTest(profile=name):
                self.assertTrue(resolve_presets(name, ()))

    def test_only_one_preset_limits_length(self) -> None:
        """Лимит длины должен стоять ровно в одном пресете: у reasoning-моделей
        max_tokens съедается блоками <think> и обрезает ответ до пустоты."""
        limited = [p.key for p in resolve_presets("full", ()) if p.params.max_tokens is not None]
        self.assertEqual(limited, ["G_budget"])

    def test_with_budget_touches_only_limited_preset(self) -> None:
        presets = with_budget(resolve_presets("full", ()), 512)
        limited = {p.key: p.params.max_tokens for p in presets}
        self.assertEqual(limited["G_budget"], 512)
        self.assertIsNone(limited["B_precise"])
        self.assertIsNone(limited["A_baseline"])

    def test_with_budget_rejects_non_positive(self) -> None:
        with self.assertRaises(ConfigError):
            with_budget(resolve_presets("full", ()), 0)


if __name__ == "__main__":
    unittest.main()
