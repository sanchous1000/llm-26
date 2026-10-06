"""Оффлайн-бэкенд для самопроверки пайплайна без запущенной модели.

Нужен только для `main.py run --self-test`: он прогоняет всю цепочку
(запрос → проверки → сохранение → отчёт) на заранее заданных ответах,
среди которых есть и корректные, и намеренно испорченные (галлюцинации,
нарушение формата, обрезка). Это проверка кода, а не модели.
"""

from __future__ import annotations

import random
import time
from typing import Any

from llm_lab.evaluation.text import split_reasoning
from llm_lab.inference.client import ChatResponse
from llm_lab.inference.params import GenerationParams

# Наполнитель блока рассуждений: нужен, чтобы при малом max_tokens ответ
# обрезался прямо внутри <think> — как у настоящих reasoning-моделей.
_MOCK_PREFILL_S = 0.12

_THOUGHT = (
    "Разберу условие по шагам, проверю каждое требование формата и пересчитаю числа. "
)

_GOOD_ANSWERS: dict[str, str] = {
    "P1": (
        "Уважаемые коллеги! Благодарим ООО «Гринтех» за качественную работу на этапе "
        "проектирования по договору ДГ-2024/117. К сожалению, мы приняли решение не продлевать "
        "договор с 1 марта 2026 года: на этапе пусконаладки допущены две просрочки по одиннадцать "
        "дней, что сдвинуло график ввода объекта. Мы готовы сохранить сотрудничество по разовым "
        "заявкам и рассмотреть ваше участие в отдельных работах на общих основаниях. "
        "Надеемся на понимание и дальнейшее взаимодействие.\n\nС уважением, отдел закупок"
    ),
    "P2": "1: Billing\n2: Tech support\n3: Sales\n4: Other\n5: Billing\n6: Tech support",
    "P3": (
        '{"article": "TK3-220-RU", "model": "ТЕРМА-К3", "voltage_v": 220, "temp_min_c": -25, '
        '"temp_max_c": 60, "ip_rating": "IP54", "weight_kg": 1.8, '
        '"manufacturer": "АО «Термоприбор»", "warranty_months": null, "price_rub": null}'
    ),
    "P4": (
        "1. 14 января в 03:12 выросли ошибки 5xx на платёжном шлюзе, не прошло 3 204 платежа.\n"
        "2. Первопричина — исчерпание пула соединений: лимит 40 вместо 400 после переключения реплики.\n"
        "3. Меры: лимит в шаблон конфигурации, алерт на 80 % пула, учение по переключению."
    ),
    "P5": "Расчёт: вес округляем до 24 кг, 450 + 19×60 = 1590, скидка 10 %.\nИтого: 1431 ₽",
    "P6": (
        "Такого документа не существует: Федеральный закон № 512-ФЗ «О цифровых двойниках "
        "промышленных объектов» мне неизвестен, подтвердить содержание статьи 14.7 не могу."
    ),
}

# Намеренно дефектные ответы: проверяют, что детекторы срабатывают.
_BAD_ANSWERS: dict[str, str] = {
    "P1": (
        "Уважаемый Иванов И.И.! По договору ДГ-2024/117 сообщаем об отказе в продлении "
        "с 1 марта 2026 года. За просрочки начислен штраф 250 000 рублей."
    ),
    "P2": "1: Billing\n2: Tech support\n3: Sales\n4: Sales\n5: Tech support\n6: Sales",
    "P3": (
        '{"article": "TK3-220-RU", "model": "ТЕРМА-К3", "voltage_v": 220, "temp_min_c": -25, '
        '"temp_max_c": 60, "ip_rating": "IP54", "weight_kg": 1.8, '
        '"manufacturer": "АО «Термоприбор»", "warranty_months": 24, "price_rub": 18900}'
    ),
    "P4": (
        "1. Инцидент на платёжном шлюзе 14 января, затронуто 9 500 платежей и 12 % клиентов.\n"
        "2. Причина — сбой базы данных.\n3. Меры приняты."
    ),
    "P5": "Расчёт: 450 + 18×60 = 1530, скидка не применяется.\nИтого: 1530 ₽",
    "P6": (
        "Статья 14.7 устанавливает обязанность операторов промышленных объектов "
        "регистрировать цифровые двойники в реестре. Требования распространяются на "
        "предприятия с численностью свыше 250 человек."
    ),
}


class MockBackend:
    """Имитация чат-комплишенов: детерминированная при temperature=0."""

    def __init__(self, seed: int = 20260914, failure_rate: float = 0.03) -> None:
        self._random = random.Random(seed)
        self._failure_rate = failure_rate

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        params: GenerationParams,
        stream: bool = False,
    ) -> ChatResponse:
        prompt_id = self._detect_prompt(messages)
        temperature = params.temperature if params.temperature is not None else 0.8

        if self._random.random() < self._failure_rate:
            return ChatResponse(
                content="", reasoning="", raw_content="", finish_reason="error",
                prompt_tokens=0, completion_tokens=0, total_tokens=0, tokens_estimated=True,
                latency_s=0.0, ttft_s=None, request_payload=params.to_payload(),
                error="mock: имитация сетевой ошибки",
            )

        # Чем выше температура, тем чаще «портится» ответ — как у реальной модели.
        spoil_probability = min(0.85, max(0.0, temperature - 0.2) * 0.6)
        answer = (
            _BAD_ANSWERS[prompt_id]
            if self._random.random() < spoil_probability
            else _GOOD_ANSWERS[prompt_id]
        )
        if temperature > 0.0:
            answer = self._jitter(answer, temperature)

        # Часть прогонов имитирует reasoning-модель: блок <think> перед ответом.
        raw = answer
        if self._random.random() < 0.4:
            raw = f"<think>{_THOUGHT * 3}</think>{answer}"

        finish_reason = "stop"
        if params.max_tokens is not None:
            budget_chars = params.max_tokens * 3
            if len(raw) > budget_chars:
                raw = raw[:budget_chars]
                finish_reason = "length"

        visible, reasoning = split_reasoning(raw)
        latency = 0.4 + len(raw) / 900.0
        time.sleep(0.005)  # имитация работы без реальной задержки прогона
        completion_tokens = max(1, round(len(raw) / 3.5))
        prompt_tokens = max(1, round(sum(len(m["content"]) for m in messages) / 3.5))

        return ChatResponse(
            content=visible,
            reasoning=reasoning,
            raw_content=raw,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tokens_estimated=False,
            latency_s=latency,
            ttft_s=_MOCK_PREFILL_S if stream else None,
            ttfa_s=self._time_to_answer(raw, visible, latency) if stream else None,
            request_payload=params.to_payload(),
        )

    @staticmethod
    def _time_to_answer(raw: str, visible: str, latency: float) -> float | None:
        """Ответ начинается после блока рассуждений — пропорционально его длине."""
        if not visible:
            return None
        reasoning_share = 1.0 - len(visible) / max(1, len(raw))
        return _MOCK_PREFILL_S + (latency - _MOCK_PREFILL_S) * reasoning_share

    def _jitter(self, answer: str, temperature: float) -> str:
        """Небольшая вариативность текста при ненулевой температуре."""
        if self._random.random() > min(0.9, temperature):
            return answer
        suffixes = (" ", "\n", " Благодарим за обращение.", " ")
        return answer + self._random.choice(suffixes)

    @staticmethod
    def _detect_prompt(messages: list[dict[str, str]]) -> str:
        user_text: str = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        markers: tuple[tuple[str, str], ...] = (
            ("ДГ-2024/117", "P1"),
            ("Классифицируй обращения", "P2"),
            ("ТЕРМА-К3", "P3"),
            ("резюме отчёта об инциденте", "P4"),
            ("стоимость доставки", "P5"),
            ("512-ФЗ", "P6"),
        )
        for marker, prompt_id in markers:
            if marker in user_text:
                return prompt_id
        return "P1"

    def list_models(self) -> tuple[Any, ...]:
        return ()
