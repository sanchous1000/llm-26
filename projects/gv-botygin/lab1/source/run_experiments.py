import argparse
import json
import time
from datetime import datetime, timezone

from openai import OpenAI

import config
from prompts import PROMPTS


def run_once(client: OpenAI, model: str, prompt: str, params: dict) -> dict:
    """Один запрос в режиме стриминга: так можно отдельно измерить время до первого токена
    (≈ обработка промпта) и скорость генерации."""
    params = dict(params)
    extra_body = params.pop("extra_body", None)

    start = time.perf_counter()
    first_token_at = None
    chunks, n_content_chunks = [], 0
    usage, finish_reason = None, None

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra_body,
        **params,
    )
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.delta and choice.delta.content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            chunks.append(choice.delta.content)
            n_content_chunks += 1
        if choice.finish_reason:
            finish_reason = choice.finish_reason
    end = time.perf_counter()

    latency = end - start
    ttft = (first_token_at or end) - start
    prompt_tokens = usage.prompt_tokens if usage else None
    completion_tokens = usage.completion_tokens if usage else n_content_chunks
    decode_time = latency - ttft

    return {
        "response": "".join(chunks),
        "finish_reason": finish_reason,
        "latency_s": round(latency, 3),
        "ttft_s": round(ttft, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usage_estimated": usage is None,
        # скорость обработки входа (prefill) — грубая оценка, т.к. TTFT включает сетевые накладные расходы
        "prompt_tps": round(prompt_tokens / ttft, 1) if prompt_tokens and ttft > 0 else None,
        # скорость генерации (decode): токены после первого / время после первого токена
        "gen_tps": round((completion_tokens - 1) / decode_time, 1) if completion_tokens > 1 and decode_time > 0 else None,
    }


def warm_up(client: OpenAI, model: str) -> None:
    """Первый запрос загружает модель в память — его время не должно попасть в замеры."""
    print(f"  прогрев {model} ...", flush=True)
    client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Привет"}], max_tokens=1
    )


def load_done() -> set:
    done = set()
    if config.RAW_RESULTS_FILE.exists():
        for line in config.RAW_RESULTS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    done.add((r["model"], r["prompt_id"], r["mode"], r["repeat"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=config.MODELS)
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    parser.add_argument("--modes", nargs="+", default=["A", "B"], choices=["A", "B"])
    parser.add_argument("--repeats", type=int, default=config.N_REPEATS)
    parser.add_argument("--overwrite", action="store_true", help="удалить старые результаты и прогнать всё заново")
    args = parser.parse_args()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.overwrite and config.RAW_RESULTS_FILE.exists():
        config.RAW_RESULTS_FILE.unlink()
    done = load_done()

    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=config.REQUEST_TIMEOUT)
    total = len(args.models) * len(args.prompts) * len(args.modes) * args.repeats
    n = 0

    with config.RAW_RESULTS_FILE.open("a", encoding="utf-8") as out:
        for model in args.models:
            print(f"\n=== {model} ===", flush=True)
            warmed = False
            for prompt_id in args.prompts:
                for mode in args.modes:
                    params = {} if mode == "A" else config.TUNED_PARAMS[prompt_id]
                    for repeat in range(1, args.repeats + 1):
                        n += 1
                        tag = f"[{n}/{total}] {model} {prompt_id} {mode} #{repeat}"
                        if (model, prompt_id, mode, repeat) in done:
                            print(f"{tag}: уже есть, пропуск")
                            continue
                        if not warmed:
                            warm_up(client, model)
                            warmed = True

                        record = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "base_url": config.BASE_URL,
                            "model": model,
                            "prompt_id": prompt_id,
                            "mode": mode,
                            "repeat": repeat,
                            "params": params,
                        }
                        try:
                            record |= run_once(client, model, PROMPTS[prompt_id]["text"], params)
                            print(
                                f"{tag}: {record['latency_s']} s, "
                                f"{record['prompt_tokens']}→{record['completion_tokens']} tok, "
                                f"{record['gen_tps']} tok/s",
                                flush=True,
                            )
                        except Exception as e:
                            record["error"] = f"{type(e).__name__}: {e}"
                            print(f"{tag}: ОШИБКА {record['error']}", flush=True)

                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out.flush()

    print(f"\nГотово. Результаты: {config.RAW_RESULTS_FILE}")


if __name__ == "__main__":
    main()
