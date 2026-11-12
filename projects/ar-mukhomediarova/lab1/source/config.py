"""Настройки эксперимента: адрес сервера, модели, режимы генерации."""
import os
from pathlib import Path

# OpenAI-совместимый REST API Ollama
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # Ollama ключ не проверяет, но клиенту он нужен
REQUEST_TIMEOUT = 600  # секунд

# Три модели из разных семейств (квантизация Q4_K_M по умолчанию в Ollama)
MODELS = [
    "qwen2.5:7b",   # Qwen (Alibaba)
    "llama3.1:8b",  # Llama (Meta)
    "mistral:7b",   # Mistral (Mistral AI)
]

# A — базовый режим: в запрос не передаётся ни одного гиперпараметра (дефолты движка/Modelfile).
# B — тюнинг: меняем три гиперпараметра из списка задания.
MODES = {
    "A_base": {},
    "B_tuned": {
        "temperature": 0.2,  # меньше случайности -> выше детерминизм
        "top_p": 0.8,        # сужаем ядро выборки
        "max_tokens": 512,   # ограничиваем длину ответа
    },
}

# Сколько раз повторять каждую комбинацию модель × промпт × режим (для оценки стабильности)
REPEATS = 3

LAB_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_DIR / "results"
RAW_RESULTS = RESULTS_DIR / "raw_runs.jsonl"
