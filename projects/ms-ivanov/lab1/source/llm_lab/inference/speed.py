"""Производные метрики скорости генерации.

Единая формула для новых прогонов (inference/client.py) и для пересчёта уже
сохранённых (maintenance/recompute_metrics.py), чтобы цифры в отчётах были сопоставимы.

Термины:
    ttft_s  — время до ПЕРВОГО токена любого вида (рассуждение или ответ);
    ttfa_s  — время до первого токена ВИДИМОГО ответа. У reasoning-моделей
              ≈ время, потраченное на рассуждения;
    output_tps — токены вывода / время декодирования. Если TTFT известен,
              из окна вычитается prefill («decode»), иначе делим на полное
              время запроса («end_to_end», немного занижено на время prefill).

История ошибки (метрики версии 1): TTFT засекался по первому токену поля
`content`. Когда LM Studio отдаёт рассуждения отдельным полем
`reasoning_content`, это время до начала ОТВЕТА, т.е. 95–97 % всего запроса.
Все токены (рассуждения + ответ) делились на короткий хвост — отсюда тысячи ток/с.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

METRICS_VERSION = 2

# Оценка токенов видимого ответа по символам (русский текст, JSON, числа).
# Ответы короткие, поэтому абсолютная ошибка оценки мала по сравнению
# с тысячами токенов рассуждений.
ANSWER_CHARS_PER_TOKEN = 3.0

TPS_METHOD_DECODE = "decode"
TPS_METHOD_END_TO_END = "end_to_end"


@dataclass(frozen=True)
class SpeedInputs:
    latency_s: float
    ttft_s: float | None
    ttfa_s: float | None
    completion_tokens: int
    answer_chars: int
    reasoning_chars: int


def estimate_answer_tokens(answer_chars: int, completion_tokens: int) -> int:
    if answer_chars <= 0 or completion_tokens <= 0:
        return 0
    return min(completion_tokens, max(1, round(answer_chars / ANSWER_CHARS_PER_TOKEN)))


def derive_speed_metrics(inputs: SpeedInputs) -> dict[str, Any]:
    """Скорость, время до ответа и оценка токенов рассуждений/ответа."""
    latency = max(0.0, inputs.latency_s)
    tokens = max(0, inputs.completion_tokens)

    if inputs.ttft_s is not None and 0.0 <= inputs.ttft_s < latency:
        window = latency - inputs.ttft_s
        method = TPS_METHOD_DECODE
    else:
        window = latency
        method = TPS_METHOD_END_TO_END
    output_tps = tokens / window if tokens > 0 and window > 0 else None

    if inputs.reasoning_chars > 0:
        answer_tokens = estimate_answer_tokens(inputs.answer_chars, tokens)
        reasoning_tokens = tokens - answer_tokens
    else:
        answer_tokens = tokens
        reasoning_tokens = 0

    return {
        "ttft_s": _round(inputs.ttft_s, 3),
        "ttfa_s": _round(inputs.ttfa_s, 3),
        "output_tps": _round(output_tps, 2),
        "tps_method": method,
        "answer_tokens_est": answer_tokens,
        "reasoning_tokens_est": reasoning_tokens,
        "reasoning_time_share": _reasoning_time_share(inputs, latency),
        "metrics_version": METRICS_VERSION,
    }


def _reasoning_time_share(inputs: SpeedInputs, latency: float) -> float | None:
    """Доля времени запроса до начала видимого ответа."""
    if latency <= 0:
        return None
    if inputs.ttfa_s is not None:
        return round(min(1.0, inputs.ttfa_s / latency), 4)
    if inputs.reasoning_chars > 0 and inputs.answer_chars == 0:
        return 1.0  # ответ так и не начался: всё время ушло на рассуждения
    return None


def _round(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None
