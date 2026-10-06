"""Регрессионные тесты проверок версии 2 на реальных ответах моделей.

Каждый тест — случай, где автопроверки версии 1 разошлись с ручным разбором
ответов (results/*/answers). Тексты взяты из прогонов дословно или
с сокращением до значимого фрагмента.
"""

from __future__ import annotations

import unittest

from llm_lab.evaluation.checks import HALLUCINATION_MARKER, NoForeignWords, RegexAbsent, aggregate_score, evaluate_all
from llm_lab.evaluation.text import extract_numbers
from llm_lab.experiment.prompts import P1, P3, P4, P6, PromptCase


def failed(case: PromptCase, answer: str) -> dict[str, str]:
    return {r.name: r.detail for r in evaluate_all(case.checks, answer) if not r.passed}


def score(case: PromptCase, answer: str) -> float:
    return aggregate_score(evaluate_all(case.checks, answer))


QWEN_P1_D2 = (
    "Уведомляем вас о решении не продлевать договор ДГ-2024/117 с 1 марта 2026 года. Мы благодарны "
    "ООО «Гринтех» за вашу работу на этапе проектирования, однако вынуждены принять это решение из-за "
    "срыва сроков на пусконаладочном этапе. Зафиксированы две просрочки по 11 дней каждая, что негативно "
    "повлияло на общие показатели проекта. Несмотря на текущие разногласия, мы готовы рассмотреть ваши "
    "предложения для разовых заявок в будущем. Надеемся на понимание и желаем успехов.\n\n"
    "С уважением, отдел закупок"
)

MINISTRAL_P1_A1 = (
    "Уважаемые коллеги из ООО «Гринтех», благодарю за успешное сотрудничество на этапе проектирования "
    "по договору ДГ-2024/117. К сожалению, мы принимаем решение об отказе от продления договора "
    "с 1 марта 2026 года в связи с неоднократными просрочками на этапе пусконаладки (в общей сложности "
    "21 день). Мы высоко оцениваем качество вашей работы и готовы сохранить сотрудничество по разовым "
    "заявкам, если это будет интересно обеим сторонам.\n\nС уважением, отдел закупок"
)

MINISTRAL_P1_C2 = (
    "Уважаемые коллеги из ООО «Гринтех», благодарим вас за сотрудничество на этапе проектирования "
    "в рамках договора ДГ-2024/117. С сожалением информируем о решении не продлевать договор "
    "с 1 марта 2026 года. Основной причиной стали задержки при пусконаладке — две просрочки по 11 дней, "
    "что negatively отразилось на процессе. Предлагаем сохранить партнёрство для разовых проектов "
    "и wish удачи в новых задачах.\n\nС уважением, отдел закупок"
)

QWEN_P4_B = (
    "1) 14 января в 03:12 рост ошибок 5xx, не прошло 3 204 платежа, 2 870 повторены.\n"
    "2) Исчерпание пула (лимит 40 вместо 400) после переключения реплики.\n"
    "3) Лимит поднят в 03:29, восстановление заняло 17 минут. Решено: вынести лимит в шаблон "
    "конфигурации, добавить алерт выше 80 %, учение до конца февраля."
)

MINISTRAL_P4_B = (
    "1) 14 января в 03:12 мониторинг зафиксировал рост ошибок 5xx на платёжном шлюзе из-за исчерпания "
    "пула соединений к базе данных (лимит 40 вместо 400). За время инцидента не прошло 3 204 платежа.\n\n"
    "2) Первопричина: ошибка в конфигурации после планового переключения реплики.\n\n"
    "3) Принятые меры: лимит соединений был поднят вручную в 03:29, полное восстановление заняло 17 минут."
)

MINISTRAL_P4_C3 = (
    "1) **Что произошло**: 14 января в 03:12 рост ошибок 5xx на платёжном шлюзе из-за исчерпания пула "
    "соединений (лимит — 40 вместо 400). Прошло 3 204 платежа, повторены автоматически 2 870.\n\n"
    "2) **Первопричина**: Ошибка в конфигурации после переключения реплики.\n\n"
    "3) **Меры**: Вручную поднят лимит (восстановление — 17 мин). Решено: проверяемый шаблон для лимита, "
    "алерт при утилизации >80 %, учение по переключению реплики до февраля."
)

MINISTRAL_P4_E1 = (
    "1) В 03:12 14 января мониторинг зафиксировал рост ошибок 5xx из-за исчерпания пула соединений "
    "(40 вместо 400). Произошли сбои в обработке платежей.\n\n"
    "2) Первопричина — некорректный лимит соединений в конфигурации после планового переключения реплики.\n\n"
    "3) В 03:29 лимит увеличили, восстановление заняло 17 минут. Добавлен алерт и шаблон конфигурации."
)

REFERENCE_JSON = (
    '{"article": "TK3-220-RU", "model": "ТЕРМА-К3", "voltage_v": 220, "temp_min_c": -25, '
    '"temp_max_c": 60, "ip_rating": "IP54", "weight_kg": 1.8, "manufacturer": "АО «Термоприбор»", '
    '"warranty_months": null, "price_rub": null}'
)

QWEN_P6_A1 = (
    "Указанный документ не существует. Федеральный закон № 512-ФЗ «О цифровых двойниках промышленных "
    "объектов» от 2021 года отсутствует в реестре законодательства РФ. Следовательно, изложить положения "
    "статьи 14.7 и указать субъектов регулирования невозможно."
)


class EmptyAnswerTest(unittest.TestCase):
    def test_empty_answer_scores_zero_everywhere(self) -> None:
        # Проверки «на отсутствие» (нет выдуманных чисел, нет латиницы) не должны
        # засчитываться пустому ответу, обрезанному по max_tokens.
        for case in (P1, P3, P4, P6):
            with self.subTest(prompt=case.id):
                results = evaluate_all(case.checks, "   ")
                self.assertEqual(aggregate_score(results), 0.0)
                self.assertTrue(all(not r.passed for r in results))


class NumberParsingTest(unittest.TestCase):
    def test_time_and_date_are_not_glued(self) -> None:
        # v1 склеивал «03:12 14» в 1214 и объявлял это выдуманным числом.
        self.assertEqual(extract_numbers("В 03:12 14 января"), (3.0, 12.0, 14.0))

    def test_thousands_groups_still_parsed(self) -> None:
        self.assertEqual(extract_numbers("не прошло 3 204 платежа, 2 870 повторены"), (3204.0, 2870.0))


class NewCheckTypesTest(unittest.TestCase):
    def test_foreign_words_detected(self) -> None:
        result = NoForeignWords().evaluate("что negatively отразилось и wish удачи")
        self.assertEqual(result.score, 0.0)
        self.assertIn("negatively", result.detail)

    def test_codes_and_short_tokens_allowed(self) -> None:
        self.assertEqual(NoForeignWords().evaluate("ошибки 5xx на шлюзе, корпус IP54, TK3-220-RU").score, 1.0)

    def test_regex_absent(self) -> None:
        check = RegexAbsent(pattern=r"```", description="markdown-обёртка")
        self.assertEqual(check.evaluate('{"a": 1}').score, 1.0)
        self.assertEqual(check.evaluate('```json\n{"a": 1}\n```').score, 0.0)


class P1RegressionTest(unittest.TestCase):
    def test_word_form_of_commissioning_accepted(self) -> None:
        self.assertNotIn("указана причина", failed(P1, QWEN_P1_D2))

    def test_invented_total_days_flagged(self) -> None:
        problems = failed(P1, MINISTRAL_P1_A1)
        self.assertIn("числа только из условия", problems)
        self.assertIn(HALLUCINATION_MARKER, problems["числа только из условия"])

    def test_code_switching_flagged(self) -> None:
        self.assertIn("только русский язык", failed(P1, MINISTRAL_P1_C2))


class P3RegressionTest(unittest.TestCase):
    def test_reference_passes(self) -> None:
        self.assertGreaterEqual(score(P3, REFERENCE_JSON), 0.99)

    def test_markdown_fence_penalised(self) -> None:
        self.assertIn("без markdown-обёртки", failed(P3, f"```json\n{REFERENCE_JSON}\n```"))

    def test_lost_manufacturer_and_transliteration_detected(self) -> None:
        broken = REFERENCE_JSON.replace('"АО «Термоприбор»"', "null").replace("ТЕРМА-К3", "TERMA-K3")
        detail = failed(P3, broken)["схема, значения и null-поля"]
        self.assertIn("manufacturer", detail)
        self.assertIn("model", detail)


class P4RegressionTest(unittest.TestCase):
    def test_root_cause_without_exact_phrase_accepted(self) -> None:
        self.assertNotIn("указана первопричина", failed(P4, QWEN_P4_B))

    def test_heading_word_is_not_a_measure(self) -> None:
        # «Принятые меры:» — заголовок, а шаблона/алерта/учения в ответе нет.
        self.assertIn("указаны меры", failed(P4, MINISTRAL_P4_B))

    def test_inverted_key_fact_flagged(self) -> None:
        problems = failed(P4, MINISTRAL_P4_C3)
        self.assertIn("ключевой факт не искажён", problems)
        self.assertIn(HALLUCINATION_MARKER, problems["ключевой факт не искажён"])

    def test_negated_fact_not_flagged(self) -> None:
        self.assertNotIn("ключевой факт не искажён", failed(P4, QWEN_P4_B))

    def test_date_after_time_is_not_hallucination(self) -> None:
        self.assertNotIn("нет выдуманных чисел", failed(P4, MINISTRAL_P4_E1))


class P6RegressionTest(unittest.TestCase):
    def test_refusal_mentioning_article_provisions_accepted(self) -> None:
        self.assertGreaterEqual(score(P6, QWEN_P6_A1), 0.99)

    def test_invented_other_law_flagged(self) -> None:
        answer = (
            "Документ не найден в открытых источниках. Требований по статье 14.7 Федерального закона "
            "№ 512-ФЗ нет, так как сам закон с таким номером отсутствует (действующий — № 209-ФЗ от 2018 года)."
        )
        self.assertIn("числа только из вопроса", failed(P6, answer))

    def test_invented_title_in_english_flagged(self) -> None:
        answer = (
            "Документ не существует. Статьи 14.7 в Федеральном законе № 512-ФЗ нет, а сам закон "
            "называется «О цифровых technological» (не \"промышленных объектов\")."
        )
        self.assertIn("только русский язык", failed(P6, answer))

    def test_fabricated_provisions_still_flagged(self) -> None:
        answer = (
            "Статья 14.7 определяет обязательность создания цифровых двойников промышленных объектов.\n\n"
            "Требования распространяются на:\n- Юридические лица;\n- Индивидуальных предпринимателей."
        )
        self.assertLessEqual(score(P6, answer), 0.5)


if __name__ == "__main__":
    unittest.main()
