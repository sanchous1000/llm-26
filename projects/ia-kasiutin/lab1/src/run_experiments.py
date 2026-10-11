import json
import sys
from datetime import datetime, timezone

from openai import APIConnectionError, APIStatusError

from config import EXPECTED_MODELS, MODES, REPEATS, RESULTS_DIR, RUNS_PATH, build_request_params
from prompts import PROMPTS
from vllm_client import chat, list_model_ids, make_client


def load_done(path) -> set[tuple]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((row["model"], row["prompt_id"], row["mode"], row["repeat"]))
    return done


def served_model(client) -> str:
    model_ids = list_model_ids(client)
    if not model_ids:
        raise SystemExit("Сервер не вернул ни одной модели с GET /v1/models.")
    if len(model_ids) > 1:
        print(f"На сервере несколько моделей, беру первую: {model_ids[0]}")
    model_id = model_ids[0]
    if model_id not in EXPECTED_MODELS:
        print(
            f"Загружена {model_id}. Это не один из ожидаемых id "
            f"({', '.join(EXPECTED_MODELS)}). Прогон всё равно запишется."
        )
    return model_id


def main() -> None:
    client = make_client()
    try:
        model_id = served_model(client)
    except APIConnectionError as error:
        raise SystemExit(
            "Нет соединения с vLLM. Проверьте VLLM_BASE_URL и что сервер слушает порт 8000."
        ) from error
    except APIStatusError as error:
        detail = getattr(error, "message", None) or str(error)
        raise SystemExit(f"vLLM вернул {error.status_code} на GET /v1/models: {detail}") from error

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = load_done(RUNS_PATH)
    print(f"Модель: {model_id}")
    print(f"Уже записано комбинаций: {len(done)}")

    with RUNS_PATH.open("a", encoding="utf-8") as handle:
        for prompt in PROMPTS:
            for mode in MODES:
                request_params = build_request_params(model_id, mode)
                for repeat in range(1, REPEATS + 1):
                    key = (model_id, prompt["id"], mode, repeat)
                    if key in done:
                        print(f"пропуск {prompt['id']} {mode} #{repeat}")
                        continue
                    print(f"запрос {prompt['id']} {mode} #{repeat} ...", flush=True)
                    try:
                        result = chat(client, model_id, prompt["messages"], request_params)
                    except APIStatusError as error:
                        detail = getattr(error, "message", None) or str(error)
                        raise SystemExit(
                            f"vLLM вернул {error.status_code} на {prompt['id']} {mode} #{repeat}: {detail}"
                        ) from error
                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": model_id,
                        "prompt_id": prompt["id"],
                        "task": prompt["task"],
                        "mode": mode,
                        "repeat": repeat,
                        "request_params": request_params,
                        "messages": prompt["messages"],
                        "response_text": result["response_text"],
                        "finish_reason": result["finish_reason"],
                        "usage": result["usage"],
                        "latency_s": result["latency_s"],
                        "tokens_per_second": result["tokens_per_second"],
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    usage = result["usage"]
                    print(
                        f"  {usage['completion_tokens']} ток. за {result['latency_s']:.2f} с, "
                        f"finish={result['finish_reason']}"
                    )

    print(f"Записано в {RUNS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
