"""Последовательный прогон локальных LLM через OpenAI-совместимый REST Ollama.

Скрипт намеренно не запускает модели параллельно: после всех запросов к одной
модели она выгружается из RAM до перехода к следующей.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
LAB_DIR = HERE.parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_RESULTS = LAB_DIR / "results"


def run_command(args: list[str], timeout: float = 10.0) -> str:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip()


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Не удалось подключиться к {url}: {error.reason}") from error


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def unload_model(model_id: str) -> None:
    run_command(["ollama", "stop", model_id], timeout=30.0)
    for _ in range(30):
        if model_id not in run_command(["ollama", "ps"], timeout=5.0):
            return
        time.sleep(1)
    raise RuntimeError(f"Модель {model_id} не выгрузилась за 30 секунд")


def warm_up_model(base_url: str, model_id: str, timeout: float) -> None:
    """Загружает модель до начала измерений, исключая холодный старт."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Ответь одним словом: готов."}],
        "temperature": 0,
        "max_tokens": 1,
    }
    request_json(f"{base_url}/chat/completions", payload, timeout)


def build_payload(
    model_id: str,
    prompt: dict[str, str],
    mode: str,
    tuned_parameters: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
    }
    if mode == "tuned":
        payload.update(tuned_parameters)
    return payload


def run_one(
    base_url: str,
    model: dict[str, str],
    prompt: dict[str, str],
    mode: str,
    tuned_parameters: dict[str, Any],
    timeout: float,
    repeat: int = 1,
) -> dict[str, Any]:
    payload = build_payload(model["id"], prompt, mode, tuned_parameters)
    started_at = datetime.now(timezone.utc)
    monotonic_start = time.monotonic()
    response = request_json(f"{base_url}/chat/completions", payload, timeout)
    elapsed = time.monotonic() - monotonic_start

    choice = response.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    throughput = completion_tokens / elapsed if completion_tokens and elapsed else None

    return {
        "run_id": (
            f"{safe_name(model['id'])}-{prompt['id']}-{mode}"
            + (f"-r{repeat}" if repeat > 1 else "")
        ),
        "started_at": started_at.isoformat(),
        "model": model,
        "prompt": {
            "id": prompt["id"],
            "type": prompt["type"],
            "title": prompt["title"],
            "system": prompt["system"],
            "user": prompt["user"],
        },
        "mode": mode,
        "repeat": repeat,
        "parameters": {} if mode == "baseline" else tuned_parameters,
        "request_endpoint": "/v1/chat/completions",
        "response": content,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "metrics": {
            "wall_time_s": round(elapsed, 3),
            "end_to_end_completion_tokens_per_s": (
                round(throughput, 3) if throughput is not None else None
            ),
            "cold_start_excluded": True,
        },
        "raw_response": response,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Число повторов каждой комбинации модель–промпт–режим",
    )
    parser.add_argument("--models", nargs="*", help="Прогнать только указанные ID")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("baseline", "tuned"),
        default=("baseline", "tuned"),
        help="Прогнать только указанные режимы (по умолчанию оба)",
    )
    parser.add_argument("--resume", action="store_true", help="Не повторять уже сохранённые прогоны")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats должен быть не меньше 1")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = config["models"]
    if args.models:
        selected = set(args.models)
        models = [model for model in models if model["id"] in selected]
        missing = selected - {model["id"] for model in models}
        if missing:
            raise SystemExit(f"Неизвестные модели: {', '.join(sorted(missing))}")

    responses_dir = args.results / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")

    request_json(f"{base_url}/models", None, timeout=15.0)
    print(f"OpenAI-compatible API: {base_url}")
    print(f"Platform: {platform.platform()}")

    all_model_ids = [model["id"] for model in config["models"]]
    for model_id in all_model_ids:
        unload_model(model_id)

    completed = 0
    try:
        for model in models:
            print(f"\n=== {model['family']}: {model['id']} ===", flush=True)
            print(f"WARMUP {model['id']} (время не учитывается)", flush=True)
            warm_up_model(base_url, model["id"], args.timeout)
            for prompt in config["prompts"]:
                for mode in args.modes:
                    for repeat in range(1, args.repeats + 1):
                        suffix = f"_r{repeat}" if args.repeats > 1 else ""
                        filename = responses_dir / (
                            f"{safe_name(model['id'])}_{prompt['id']}_{mode}{suffix}.json"
                        )
                        if args.resume and filename.exists():
                            print(f"SKIP {filename.name}")
                            continue
                        print(
                            f"RUN  {prompt['id']} {mode} "
                            f"({repeat}/{args.repeats})",
                            flush=True,
                        )
                        result = run_one(
                            base_url=base_url,
                            model=model,
                            prompt=prompt,
                            mode=mode,
                            tuned_parameters=config["tuned_parameters"],
                            timeout=args.timeout,
                            repeat=repeat,
                        )
                        filename.write_text(
                            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        completed += 1
                        metrics = result["metrics"]
                        usage = result["usage"]
                        print(
                            f"OK   {metrics['wall_time_s']} s; "
                            f"tokens={usage.get('prompt_tokens')}/"
                            f"{usage.get('completion_tokens')}",
                            flush=True,
                        )

            print(f"UNLOAD {model['id']}", flush=True)
            unload_model(model["id"])
            time.sleep(2)
    finally:
        for model_id in all_model_ids:
            unload_model(model_id)

    print(f"\nГотово: сохранено новых прогонов: {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
