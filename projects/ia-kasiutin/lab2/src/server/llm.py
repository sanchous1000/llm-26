from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI

from config import LlmConfig

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


class LlmUnavailable(RuntimeError):
    """vLLM недоступен или не отдаёт ожидаемую модель."""


class PromptTooLong(Exception):
    """Промпт вместе с запасом на ответ не влезает в окно модели."""


def resolve_base_url(config: LlmConfig) -> str:
    return os.environ.get("VLLM_BASE_URL") or config.base_url or DEFAULT_BASE_URL


def resolve_api_key() -> str:
    return os.environ.get("VLLM_API_KEY", "EMPTY")


def fetch_models_payload(config: LlmConfig) -> dict[str, Any]:
    """Сырой JSON GET /v1/models: у vLLM там есть max_model_len, SDK его отбрасывает."""
    from openai import OpenAI

    base_url = resolve_base_url(config)
    client = OpenAI(base_url=base_url, api_key=resolve_api_key(), timeout=config.timeout_s, max_retries=0)
    try:
        payload = client.models.with_raw_response.list().json()
    except Exception as error:  # сеть, туннель, авторизация
        raise LlmUnavailable(
            f"vLLM недоступен по {base_url}: {error}. "
            "Проверьте SSH-туннель и переменную VLLM_BASE_URL."
        ) from error
    if not isinstance(payload, dict):
        raise LlmUnavailable(f"vLLM по {base_url} отдал не объект /v1/models")
    return payload


def model_ids(payload: dict[str, Any]) -> list[str]:
    return [item["id"] for item in payload.get("data") or [] if isinstance(item, dict) and item.get("id")]


def context_length(payload: dict[str, Any], model_id: str) -> int:
    """max_model_len выбранной модели. Нет поля — окно подтвердить нельзя."""
    for item in payload.get("data") or []:
        if not isinstance(item, dict) or item.get("id") != model_id:
            continue
        raw = item.get("max_model_len")
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw
        raise LlmUnavailable(
            f"vLLM не сообщил max_model_len для модели {model_id}: окно подтвердить нельзя"
        )
    available = model_ids(payload)
    raise LlmUnavailable(f"Модель {model_id!r} не отдаётся сервером, доступны: {available}")


def list_model_ids(config: LlmConfig) -> list[str]:
    return model_ids(fetch_models_payload(config))


def resolve_model(config: LlmConfig, payload: dict[str, Any] | None = None) -> str:
    if payload is None:
        payload = fetch_models_payload(config)
    available = model_ids(payload)
    if not available:
        raise LlmUnavailable("vLLM не отдаёт ни одной модели (/v1/models пуст)")
    if config.model == "auto":
        return available[0]
    if config.model not in available:
        raise LlmUnavailable(f"Модель {config.model!r} не отдаётся сервером, доступны: {available}")
    return config.model


@lru_cache(maxsize=4)
def load_chat_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as error:
        raise LlmUnavailable(
            f"Не удалось загрузить токенизатор модели {model_id}: {error}. Окно подтвердить нельзя."
        ) from error


def count_prompt_tokens(tokenizer, messages: list[dict], model_id: str) -> int:
    """Токены чат-шаблона: роли, служебные токены и запрос на генерацию."""
    kwargs = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **kwargs,
        )
    except Exception as error:
        raise LlmUnavailable(
            f"Не удалось посчитать токены промпта модели {model_id}: {error}. Окно подтвердить нельзя."
        ) from error
    if not isinstance(encoded, list):
        raise LlmUnavailable(
            f"Токенизатор модели {model_id} вернул не список токенов. Окно подтвердить нельзя."
        )
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def ensure_prompt_fits(model_id: str, prompt_tokens: int, max_tokens: int, max_model_len: int) -> None:
    if prompt_tokens + max_tokens <= max_model_len:
        return
    raise PromptTooLong(
        f"Промпт не влезает в окно модели {model_id}: "
        f"{prompt_tokens} токенов промпта + {max_tokens} на ответ > {max_model_len}"
    )


def extra_body_for(model_id: str, config: LlmConfig) -> dict:
    extra: dict = {"repetition_penalty": config.repetition_penalty}
    if "qwen3" in model_id.lower():
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    return extra


def make_llm(config: LlmConfig, model_id: str | None = None) -> ChatOpenAI:
    model_id = model_id or resolve_model(config)
    return ChatOpenAI(
        model=model_id,
        base_url=resolve_base_url(config),
        api_key=resolve_api_key(),
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout=config.timeout_s,
        max_retries=1,
        extra_body=extra_body_for(model_id, config),
    )
