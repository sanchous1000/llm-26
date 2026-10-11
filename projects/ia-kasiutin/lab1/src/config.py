import os
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_ROOT / "results"
RUNS_PATH = RESULTS_DIR / "runs.jsonl"
SUMMARY_PATH = RESULTS_DIR / "summary.md"

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

REPEATS = 10

# Ожидаемые id. Скрипт всё равно берёт модель, которую сейчас отдаёт сервер.
EXPECTED_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "microsoft/Phi-4-mini-instruct",
    "mistralai/Ministral-3-3B-Instruct-2512",
]

MODES = {
    "baseline": {},
    "tuned": {
        "temperature": 0.1,
        "top_p": 0.9,
        "extra_body": {"repetition_penalty": 1.15},
    },
}


def extra_body_for(model_id: str) -> dict:
    """Параметры шаблона чата, одинаковые для baseline и tuned. Это не сэмплинг."""
    if "qwen3" in model_id.lower():
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def build_request_params(model_id: str, mode: str) -> dict:
    mode_params = MODES[mode]
    extra = extra_body_for(model_id)
    extra.update(mode_params.get("extra_body") or {})
    params = {key: value for key, value in mode_params.items() if key != "extra_body"}
    if extra:
        params["extra_body"] = extra
    return params
