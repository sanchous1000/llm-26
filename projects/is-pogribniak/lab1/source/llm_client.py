"""Запросы к модели через OpenAI-совместимый REST (Ollama, /v1/chat/completions)."""

import json
import random
import time

import httpx


def chat(client: httpx.Client, base_url: str, model: str, messages: list, params: dict) -> dict:
    """Streaming-запрос. Время до первого токена (TTFT) и полное время меряем сами,
    число токенов берём из usage, который Ollama присылает последним чанком."""
    body = {"model": model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, **params}
    t0 = time.perf_counter()
    ttft, content = None, []
    finish_reason, usage = None, {}

    with client.stream("POST", f"{base_url}/chat/completions", json=body) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            usage = chunk.get("usage") or usage
            if not chunk.get("choices"):
                continue
            choice = chunk["choices"][0]
            delta = choice.get("delta") or {}
            piece = delta.get("content") or ""
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            content.append(piece)
            finish_reason = choice.get("finish_reason") or finish_reason

    return {
        "response": "".join(content).strip(),
        "finish_reason": finish_reason,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_s": ttft,
        "latency_s": time.perf_counter() - t0,
    }


def model_defaults(client: httpx.Client, native_url: str, model: str) -> dict:
    """Параметры из Modelfile — это и есть «дефолты движка» для режима A.
    Читаются служебным /api/show: в OpenAI-совместимом API такой информации нет.
    """
    resp = client.post(f"{native_url}/api/show", json={"model": model})
    resp.raise_for_status()
    info = resp.json()
    return {"parameters": info.get("parameters"), "quantization": info.get("details", {}).get("quantization_level")}


PREFILL_WORDS = ["данные", "система", "отчёт", "клиент", "заказ", "платёж", "сервис", "модель",
                 "запрос", "ответ", "график", "таблица", "процесс", "результат", "доставка",
                 "магазин", "покупатель", "товар", "скидка", "оплата", "договор", "менеджер",
                 "поддержка", "приложение", "обновление", "пользователь", "компания", "сотрудник",
                 "неделя", "решение", "проблема", "вопрос", "задача", "проект", "срок", "цена"]


def bench_prefill(client: httpx.Client, base_url: str, model: str, extra: dict, n: int = 3) -> list:
    """Скорость чтения промпта: уникальные длинные тексты (кеш не срабатывает), max_tokens=1."""
    out = []
    for i in range(1, n + 1):
        rnd = random.Random(f"prefill-{i}")
        text = "Прочитай текст и ответь одним словом «ок».\n" + " ".join(
            rnd.choice(PREFILL_WORDS) for _ in range(1500))
        res = chat(client, base_url, model, [{"role": "user", "content": text}], {**extra, "max_tokens": 1})
        fresh = (res["prompt_tokens"] or 0) - res["cached_tokens"]
        tps = fresh / res["ttft_s"] if res["ttft_s"] else None
        out.append({"fresh_tokens": fresh, "ttft_s": round(res["ttft_s"], 3), "prompt_tps": round(tps)})
        print(f"[{model}] prefill #{i}: {fresh} ток. за {res['ttft_s']:.2f} c = {tps:.0f} ток/с", flush=True)
    return out
