import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from prompts import BASE_PARAMS, PROMPTS, TUNED_PARAMS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------- один запрос со стримингом ----------

def run_once(client, model, prompt_text, params):
    sampling = {k: v for k, v in params.items() if k in ("temperature", "top_p", "max_tokens")}
    extra = {k: v for k, v in params.items() if k in ("top_k", "repetition_penalty")}

    t0 = time.perf_counter()
    ttft = None
    parts, usage, finish = [], None, None
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra,
        **sampling,
    )
    for chunk in stream:
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                parts.append(choice.delta.content)
            if choice.finish_reason:
                finish = choice.finish_reason
        if chunk.usage:
            usage = chunk.usage
    total = time.perf_counter() - t0

    prompt_tokens = usage.prompt_tokens if usage else None
    completion_tokens = usage.completion_tokens if usage else None
    decode_time = total - ttft if ttft is not None else None
    return {
        "output": "".join(parts),
        "finish_reason": finish,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "total_s": total,
        # скорость декода: первый токен считаем в TTFT, поэтому токенов на 1 меньше
        "decode_tok_s": (completion_tokens - 1) / decode_time
        if decode_time and completion_tokens and completion_tokens > 1 else None,
        "prefill_tok_s": prompt_tokens / ttft if ttft and prompt_tokens else None,
    }


# ---------- скорость токенизации ----------

def bench_tokenizer(model, texts, reps=200):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    n_tokens = 0
    t0 = time.perf_counter()
    for _ in range(reps):
        for text in texts:
            n_tokens += len(tok(text)["input_ids"])
    dt = time.perf_counter() - t0
    return {"tokens": n_tokens, "seconds": dt, "tokens_per_s": n_tokens / dt}


def server_version(base_url):
    try:
        root = base_url.rsplit("/v1", 1)[0]
        with urllib.request.urlopen(f"{root}/version", timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--repeats", type=int, default=3, help="повторов каждого прогона")
    ap.add_argument("--no-tokenizer-bench", action="store_true")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    model = client.models.list().data[0].id
    print(f"Модель: {model}")

    # прогрев: первый запрос после старта заметно медленнее
    run_once(client, model, "Привет!", {"max_tokens": 8})

    runs = []
    for pid, prompt in PROMPTS.items():
        for mode, params in (("A", BASE_PARAMS), ("B", TUNED_PARAMS)):
            for rep in range(1, args.repeats + 1):
                r = run_once(client, model, prompt["text"], params)
                r.update({
                    "model": model, "prompt_id": pid, "prompt_type": prompt["type"],
                    "mode": mode, "repeat": rep, "params": params,
                })
                runs.append(r)
                print(f"{pid} {mode} #{rep}: {r['completion_tokens']} ток., "
                      f"TTFT {r['ttft_s']:.2f} c, всего {r['total_s']:.2f} c, "
                      f"{(r['decode_tok_s'] or 0):.1f} ток/с, {r['finish_reason']}")

    meta = {
        "model": model,
        "date": datetime.now().isoformat(timespec="seconds"),
        "vllm_version": server_version(args.base_url),
        "repeats": args.repeats,
        "base_params": BASE_PARAMS,
        "tuned_params": TUNED_PARAMS,
    }
    if not args.no_tokenizer_bench:
        meta["tokenizer_bench"] = bench_tokenizer(model, [p["text"] for p in PROMPTS.values()])

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / (model.replace("/", "__") + ".json")
    out.write_text(json.dumps({"meta": meta, "runs": runs}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()