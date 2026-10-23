"""Основной прогон (модель × промпт × режим × повторы) и разовый эксперимент.

Каждый прогон — строка в results/runs.jsonl (без текста промпта: он в prompts/). При перезапуске старые строки
тех же моделей и режимов удаляются, поэтому A и B можно гонять по отдельности.
"""

import json

import httpx

from .config import RESULTS, build_messages, build_params, find
from .llm_client import bench_prefill, chat, model_defaults
from .scoring import score

RUNS = RESULTS / "runs.jsonl"
MODELS = RESULTS / "models.json"


def drop_old_runs(models: list, modes: list) -> None:
    if not RUNS.exists():
        return
    keep = [line for line in RUNS.read_text(encoding="utf-8").splitlines()
            if not ((r := json.loads(line))["model"] in models and r["mode"] in modes)]
    RUNS.write_text("".join(line + "\n" for line in keep), encoding="utf-8")


def run_experiments(cfg: dict, models: list, modes: list, prefill: bool = True) -> None:
    """Прогоны — в runs.jsonl; дефолты моделей и замер prefill — в models.json."""
    client = httpx.Client(timeout=cfg["request_timeout"])
    RESULTS.mkdir(exist_ok=True)
    drop_old_runs([m["name"] for m in models], modes)
    info = json.loads(MODELS.read_text(encoding="utf-8")) if MODELS.exists() else {}

    for m in models:
        name, extra = m["name"], m.get("extra_params") or {}
        warm = chat(client, cfg["base_url"], name, [{"role": "user", "content": "Привет"}], {**extra, "max_tokens": 5})
        print(f"[{name}] прогрев {warm['latency_s']:.1f} c", flush=True)

        old_prefill = info.get(name, {}).get("prefill")
        info[name] = model_defaults(client, cfg["native_url"], name)
        info[name]["prefill"] = bench_prefill(client, cfg["base_url"], name, extra) if prefill else old_prefill
        MODELS.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

        with RUNS.open("a", encoding="utf-8") as f:
            for p in cfg["prompts"]:
                messages = build_messages(p)
                for mode in modes:
                    params = build_params(m, p, mode)
                    for rep in range(1, cfg["repeats"] + 1):
                        res = chat(client, cfg["base_url"], name, messages, params)
                        rec = {"model": name, "prompt_id": p["id"], "mode": mode, "rep": rep, "params": params, **res}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        print(f"[{name}] {p['id']} {mode} #{rep}: {res['prompt_tokens']}→{res['completion_tokens']} ток, "
                              f"TTFT {res['ttft_s'] or 0:.2f} c, всего {res['latency_s']:.2f} c", flush=True)


def run_trial(cfg: dict, model_name: str, prompt_id: str, mode: str, overrides: dict, repeats: int = 1) -> None:
    """Одна модель, один промпт, свои параметры — для подбора параметров B. Ничего не сохраняет."""
    model = find(cfg["models"], "name", model_name)
    prompt = find(cfg["prompts"], "id", prompt_id)
    params = {**build_params(model, prompt, mode), **overrides}
    messages = build_messages(prompt)
    client = httpx.Client(timeout=cfg["request_timeout"])

    print(f"{model_name} · {prompt_id} · {params or '{} (дефолты движка)'}\n")
    for rep in range(1, repeats + 1):
        res = chat(client, cfg["base_url"], model_name, messages, params)
        sc = score(prompt_id, res["response"], messages)
        shown = {k: sc[k] for k in ("quality", "missed", "toxic_list", "invented_numbers") if sc.get(k) not in (None, "")}
        print(f"=== #{rep}: {res['prompt_tokens']}→{res['completion_tokens']} ток, {res['latency_s']:.2f} c | {shown}")
        print(res["response"], "\n")
