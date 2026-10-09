from __future__ import annotations

import json
import sys
import time
from typing import Any

import requests

from config import MODELS, NATIVE_PROBE_TOKENS, REQUEST_TIMEOUT_SECONDS, TUNED_OPTIONS


def api_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def get_models_from_ollama(base_url: str) -> set[str]:
    response = requests.get(api_url(base_url, "/api/tags"), timeout=15)
    response.raise_for_status()
    return {item.get("name", "") for item in response.json().get("models", [])}


def ensure_server_and_models(base_url: str) -> None:
    try:
        response = requests.get(api_url(base_url, "/v1/models"), timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Не удалось подключиться к Ollama по адресу {base_url}. "
            "Установи и запусти Ollama, затем повтори команду.\n"
            f"Подробности: {exc}"
        ) from exc

    installed = get_models_from_ollama(base_url)
    missing = [item["model"] for item in MODELS if item["model"] not in installed]
    if missing:
        commands = "\n".join(f"  ollama pull {model}" for model in missing)
        raise RuntimeError("Не найдены следующие модели. Выполни команды в PowerShell и повтори запуск:\n" + commands)


def completion_request(
    base_url: str,
    model: str,
    prompt: str,
    mode: str,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    params: dict[str, Any] = {}
    if mode == "B_tuned":
        params = dict(TUNED_OPTIONS)
        payload.update(params)
    started = time.perf_counter()
    response = requests.post(api_url(base_url, "/v1/chat/completions"), json=payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"API не вернул choices: {json.dumps(data, ensure_ascii=False)[:500]}")
    answer = choices[0].get("message", {}).get("content") or ""
    usage = data.get("usage") or {}
    return {
        "answer": answer,
        "response_time_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "parameters": params,
    }


def native_probe(
    base_url: str,
    model: str,
    prompt: str,
    mode: str,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": "5m",
        "options": {"num_predict": NATIVE_PROBE_TOKENS},
    }
    if mode == "B_tuned":
        payload["options"].update({"temperature": TUNED_OPTIONS["temperature"], "top_p": TUNED_OPTIONS["top_p"]})
    try:
        response = requests.post(api_url(base_url, "/api/chat"), json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        prompt_count = data.get("prompt_eval_count")
        prompt_duration = data.get("prompt_eval_duration")
        eval_count = data.get("eval_count")
        eval_duration = data.get("eval_duration")
        load_duration = data.get("load_duration")
        prompt_seconds = prompt_duration / 1_000_000_000 if prompt_duration else None
        eval_seconds = eval_duration / 1_000_000_000 if eval_duration else None
        return {
            "prompt_eval_tokens": prompt_count,
            "prompt_eval_time_s": prompt_seconds,
            "prompt_eval_tokens_per_sec": prompt_count / prompt_seconds if prompt_count and prompt_seconds else None,
            "probe_output_tokens": eval_count,
            "probe_eval_time_s": eval_seconds,
            "probe_generation_tokens_per_sec": eval_count / eval_seconds if eval_count and eval_seconds else None,
            "model_load_time_s": load_duration / 1_000_000_000 if load_duration else None,
            "error": None,
        }
    except (requests.RequestException, ValueError, TypeError) as exc:
        return {
            "prompt_eval_tokens": None,
            "prompt_eval_time_s": None,
            "prompt_eval_tokens_per_sec": None,
            "probe_output_tokens": None,
            "probe_eval_time_s": None,
            "probe_generation_tokens_per_sec": None,
            "model_load_time_s": None,
            "error": str(exc),
        }


def warmup_model(base_url: str, model: str) -> float | None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Ответь одним словом: готово."}],
        "stream": False,
        "keep_alive": "5m",
        "options": {"num_predict": 1},
    }
    try:
        response = requests.post(api_url(base_url, "/api/chat"), json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        load_ns = response.json().get("load_duration")
        return load_ns / 1_000_000_000 if load_ns else None
    except requests.RequestException as exc:
        print(f"[Предупреждение] Прогрев {model} не выполнен: {exc}", file=sys.stderr)
        return None


def check_api(base_url: str) -> int:
    print(f"Проверка сервера: {base_url}")
    try:
        ensure_server_and_models(base_url)
        response = requests.post(
            api_url(base_url, "/v1/chat/completions"),
            json={
                "model": MODELS[0]["model"],
                "messages": [{"role": "user", "content": "Ответь одним словом: готово."}],
                "max_tokens": 8,
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        print("OpenAI-совместимый POST /v1/chat/completions: работает")
        print("Установленные модели:")
        for item in MODELS:
            print(f"  - {item['family']}: {item['model']}")
        return 0
    except (RuntimeError, requests.RequestException) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1
