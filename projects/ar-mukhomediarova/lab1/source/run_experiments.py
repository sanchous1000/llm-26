"""Прогон всех комбинаций модель × промпт × режим × повтор через OpenAI-совместимый API.

Запуск из каталога lab1:
    python source/run_experiments.py                  # все модели, REPEATS повторов
    python source/run_experiments.py --models qwen2.5:7b --repeats 1
"""
import argparse
import json
import sys
import time
from datetime import datetime

from openai import OpenAI

import config
from prompts import PROMPTS


def stream_chat(client: OpenAI, model: str, prompt: str, params: dict) -> dict:
    """Один запрос в потоковом режиме: так можно измерить время до первого токена (TTFT)
    и отдельно скорость обработки промпта (prefill) и генерации (decode)."""
    t0 = time.perf_counter()
    ttft = None
    parts, n_chunks = [], 0
    usage, finish_reason = None, None

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
        **params,
    )
    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta.content or ""
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(delta)
            n_chunks += 1
        if choice.finish_reason:
            finish_reason = choice.finish_reason
    total = time.perf_counter() - t0

    prompt_tokens = usage.prompt_tokens if usage else None
    # если сервер не вернул usage, считаем чанки (Ollama отдаёт ~1 токен на чанк)
    completion_tokens = usage.completion_tokens if usage else n_chunks
    ttft = ttft if ttft is not None else total
    decode_time = total - ttft

    return {
        "response": "".join(parts),
        "finish_reason": finish_reason,
        "latency_s": round(total, 3),
        "ttft_s": round(ttft, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usage_from_server": usage is not None,
        # скорость обработки входа (токенизация + prefill), ток/с
        "prefill_tps": round(prompt_tokens / ttft, 2) if prompt_tokens and ttft > 0 else None,
        # скорость генерации: первый токен приходит к моменту TTFT, остальные — за decode_time
        "decode_tps": round((completion_tokens - 1) / decode_time, 2)
        if completion_tokens and completion_tokens > 1 and decode_time > 0 else None,
    }


def warm_up(client: OpenAI, model: str) -> None:
    """Загрузка модели в память, чтобы время загрузки не попало в замеры."""
    print(f"[warm-up] {model} ...", flush=True)
    t0 = time.perf_counter()
    client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Привет"}], max_tokens=1
    )
    print(f"[warm-up] {model}: модель загружена за {time.perf_counter() - t0:.1f} с", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=config.MODELS)
    parser.add_argument("--repeats", type=int, default=config.REPEATS)
    parser.add_argument("--out", default=str(config.RAW_RESULTS))
    parser.add_argument("--append", action="store_true", help="дописывать в файл, а не перезаписывать")
    args = parser.parse_args()

    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=config.REQUEST_TIMEOUT)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(args.models) * len(PROMPTS) * len(config.MODES) * args.repeats
    done = 0
    with open(args.out, "a" if args.append else "w", encoding="utf-8") as f:
        for model in args.models:
            try:
                warm_up(client, model)
            except Exception as e:  # модель не скачана / сервер не запущен
                print(f"[error] {model}: {e}\n  Проверьте `ollama serve` и `ollama pull {model}`", file=sys.stderr)
                continue
            for prompt_id, prompt in PROMPTS.items():
                for mode, params in config.MODES.items():
                    for rep in range(1, args.repeats + 1):
                        done += 1
                        record = {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "model": model,
                            "prompt_id": prompt_id,
                            "mode": mode,
                            "repeat": rep,
                            "params": params,
                        }
                        try:
                            record.update(stream_chat(client, model, prompt, params))
                        except Exception as e:
                            record["error"] = repr(e)
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        print(
                            f"[{done}/{total_runs}] {model} | {prompt_id} | {mode} | #{rep}: "
                            + (f"{record.get('completion_tokens')} ток., {record.get('latency_s')} с, "
                               f"{record.get('decode_tps')} ток/с"
                               if "error" not in record else f"ОШИБКА {record['error']}"),
                            flush=True,
                        )
    print(f"\nГотово. Сырые результаты: {args.out}\nДальше: python source/analyze.py")


if __name__ == "__main__":
    main()
