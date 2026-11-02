import os
from pathlib import Path

# OpenAI-совместимый endpoint. Ollama: http://localhost:11434/v1
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")

MODELS = [
    "qwen2.5:7b",   # Qwen
    "llama3.1:8b",  # Meta Llama
    "mistral:7b",   # Mistral AI
]

N_REPEATS = 3

REQUEST_TIMEOUT = 300

TUNED_PARAMS = {
    # генерация: чуть больше креативности, жёсткий лимит длины, штраф за повторы
    "P1": {"temperature": 0.9, "top_p": 0.95, "max_tokens": 300, "frequency_penalty": 0.3},
    # классификация: максимально детерминированно, короткий ответ
    "P2": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 150},
    # извлечение: детерминированно, достаточно токенов на JSON
    "P3": {"temperature": 0.1, "top_p": 0.9, "max_tokens": 250},
}

LAB_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_DIR / "results"
RAW_RESULTS_FILE = RESULTS_DIR / "raw_results.jsonl"
