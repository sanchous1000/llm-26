import time

from openai import OpenAI

from config import API_KEY, BASE_URL


def make_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def list_model_ids(client: OpenAI) -> list[str]:
    return [model.id for model in client.models.list().data]


def chat(client: OpenAI, model: str, messages: list[dict], request_params: dict) -> dict:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **request_params,
    )
    latency_s = time.perf_counter() - started

    choice = response.choices[0]
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else None
    completion_tokens = usage.completion_tokens if usage else None
    total_tokens = usage.total_tokens if usage else None
    tokens_per_second = None
    if completion_tokens is not None and latency_s > 0:
        tokens_per_second = completion_tokens / latency_s

    return {
        "response_text": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "latency_s": latency_s,
        "tokens_per_second": tokens_per_second,
    }
