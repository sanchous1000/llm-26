"""Параметры запроса к серверу инференса: адрес API и гиперпараметры генерации.

Модуль не знает об устройстве эксперимента (пресетах, промптах) — его
использует и HTTP-клиент, и конфигурация эксперимента.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"  # LM Studio не проверяет ключ, но клиент его требует
DEFAULT_TIMEOUT_S = 900.0

# Параметры, которых нет в спецификации OpenAI, но которые понимает бэкенд
# llama.cpp внутри LM Studio. Могут быть отключены флагом --no-extra-params.
EXTRA_PARAM_NAMES = ("top_k", "repeat_penalty", "min_p")


class ConfigError(ValueError):
    """Ошибка валидации пользовательской конфигурации."""


@dataclass(frozen=True)
class EndpointConfig:
    """Адрес OpenAI-совместимого REST API."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    timeout_s: float = DEFAULT_TIMEOUT_S

    @staticmethod
    def from_env() -> "EndpointConfig":
        raw_timeout = os.environ.get("LLM_TIMEOUT_S", str(DEFAULT_TIMEOUT_S))
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise ConfigError(f"LLM_TIMEOUT_S должен быть числом, получено: {raw_timeout!r}") from exc
        if timeout <= 0:
            raise ConfigError("LLM_TIMEOUT_S должен быть больше нуля")
        return EndpointConfig(
            base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            api_key=os.environ.get("LLM_API_KEY", DEFAULT_API_KEY),
            timeout_s=timeout,
        )

    def with_base_url(self, base_url: str | None) -> "EndpointConfig":
        if not base_url:
            return self
        return replace(self, base_url=base_url.rstrip("/"))


@dataclass(frozen=True)
class GenerationParams:
    """Гиперпараметры генерации. `None` = параметр не отправляется вовсе.

    Пустой объект (всё None) — это и есть «базовый» режим: движок использует
    собственные дефолты, а мы не навязываем ни одного значения.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repeat_penalty: float | None = None
    seed: int | None = None

    def to_payload(self, include_extra: bool = True) -> dict[str, Any]:
        """Словарь для тела запроса: только явно заданные параметры."""
        payload: dict[str, Any] = {}
        for name, value in self.as_dict().items():
            if value is None:
                continue
            if name in EXTRA_PARAM_NAMES and not include_extra:
                continue
            payload[name] = value
        return payload

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repeat_penalty": self.repeat_penalty,
            "seed": self.seed,
        }

    def describe(self) -> str:
        """Человекочитаемое описание для отчёта."""
        explicit = self.to_payload()
        if not explicit:
            return "дефолты движка (ни один параметр не передан)"
        return ", ".join(f"{k}={v}" for k, v in sorted(explicit.items()))

    def with_max_tokens_floor(self, cap: int | None) -> "GenerationParams":
        """Страховочный лимит для прогонов, где max_tokens не задан."""
        if cap is None or self.max_tokens is not None:
            return self
        return replace(self, max_tokens=cap)
