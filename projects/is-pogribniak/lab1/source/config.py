"""Пути, конфиг эксперимента и эталоны; сборка сообщений и параметров запроса."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONFIG = ROOT / "config.yaml"
GOLD = json.loads((ROOT / "prompts" / "gold.json").read_text(encoding="utf-8"))
MODES = ("A", "B")


def load_config(path=CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def find(items: list, key: str, value: str) -> dict:
    for item in items:
        if item[key] == value:
            return item
    raise KeyError(f"нет {value} в config.yaml")


def build_messages(prompt: dict) -> list:
    """system — правила ответа, user — задача и данные."""
    read = lambda key: (ROOT / prompt[key]).read_text(encoding="utf-8").strip()
    return [
        {"role": "system", "content": read("system")},
        {"role": "user", "content": read("user")},
    ]


def build_params(model: dict, prompt: dict, mode: str) -> dict:
    """Тело запроса сверх model/messages.

    extra_params модели передаются в обоих режимах (например, отключение рассуждений Qwen3);
    гиперпараметры: A — ничего (дефолты движка), B — params_b этой задачи.
    """
    params = dict(model.get("extra_params") or {})
    if mode == "B":
        params.update(prompt.get("params_b") or {})
    return params


def missing_params_b(cfg: dict) -> list:
    return [p["id"] for p in cfg["prompts"] if not p.get("params_b")]
