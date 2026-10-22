"""Утилиты разбора текста ответов LLM."""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

# Блоки рассуждений reasoning-моделей, которые не являются частью ответа.
_REASONING_BLOCKS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
    ("<|begin_of_thought|>", "<|end_of_thought|>"),
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)
# Разряды через пробел склеиваются только группами ровно по 3 цифры («3 204»),
# иначе «03:12 14 января» превращалось в выдуманное число 1214.
_THOUSANDS_SEP = "[ \u00a0\u202f]"
_NUMBER_RE = re.compile(
    rf"-?\d{{1,3}}(?:{_THOUSANDS_SEP}\d{{3}})+(?!\d)(?:[.,]\d+)?"
    r"|-?\d+(?:[.,]\d+)?"
)
_WORD_RE = re.compile(r"[\w'\-]+", re.UNICODE)


def split_reasoning(raw: str) -> tuple[str, str]:
    """Делит сырой ответ на (видимый ответ, извлечённые рассуждения).

    Незакрытый блок рассуждений (обрезка по max_tokens) трактуется как
    полностью «мысли»: видимого ответа модель не успела выдать.
    """
    if not raw:
        return "", ""

    reasoning_parts: list[str] = []
    text = raw
    for open_tag, close_tag in _REASONING_BLOCKS:
        while True:
            start = text.lower().find(open_tag)
            if start == -1:
                break
            end = text.lower().find(close_tag, start)
            if end == -1:
                reasoning_parts.append(text[start + len(open_tag):].strip())
                text = text[:start]
                break
            reasoning_parts.append(text[start + len(open_tag):end].strip())
            text = text[:start] + text[end + len(close_tag):]
    return text.strip(), "\n\n".join(p for p in reasoning_parts if p).strip()


def has_visible_answer(text: str) -> bool:
    """Есть ли в (частичном) тексте ответ вне блоков рассуждений."""
    if not text or not text.strip():
        return False
    lowered = text.lower()
    if not any(open_tag in lowered for open_tag, _ in _REASONING_BLOCKS):
        return True
    return bool(split_reasoning(text)[0])


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_for_match(text: str) -> str:
    """Нормализация для нестрогого сравнения: регистр, ё/е, пунктуация."""
    lowered = normalize_ws(text).lower().replace("ё", "е")
    return re.sub(r"[«»\"'`.,;:!?()\[\]]", "", lowered)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def char_count(text: str) -> int:
    return len(text or "")


def extract_numbers(text: str) -> tuple[float, ...]:
    """Все числа из текста; разделители разрядов и запятая-как-точка учтены."""
    values: list[float] = []
    for match in _NUMBER_RE.findall(text or ""):
        cleaned = re.sub(_THOUSANDS_SEP, "", match).replace(",", ".")
        cleaned = cleaned.rstrip(".")
        if not cleaned or cleaned in {"-", "."}:
            continue
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return tuple(values)


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Возвращает (распарсенный объект | None, сообщение об ошибке)."""
    if not text or not text.strip():
        return None, "пустой ответ"

    candidates: list[str] = []
    for fenced in _JSON_FENCE_RE.findall(text):
        candidates.append(fenced.strip())
    candidates.append(text.strip())
    balanced = _balanced_json_slice(text)
    if balanced:
        candidates.append(balanced)

    last_error = "JSON-объект не найден"
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"невалидный JSON: {exc.msg}"
            continue
        if isinstance(parsed, dict):
            return parsed, ""
        last_error = f"ожидался объект, получен {type(parsed).__name__}"
    return None, last_error


def _balanced_json_slice(text: str) -> str:
    """Первый сбалансированный по фигурным скобкам фрагмент."""
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return ""


def similarity(left: str, right: str) -> float:
    """Схожесть двух ответов в [0, 1] по нормализованному тексту."""
    a, b = normalize_for_match(left), normalize_for_match(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def mean_pairwise_similarity(texts: tuple[str, ...]) -> float | None:
    """Средняя попарная схожесть — метрика стабильности прогонов."""
    if len(texts) < 2:
        return None
    scores = [
        similarity(texts[i], texts[j])
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    ]
    return sum(scores) / len(scores)


def max_repeated_ngram_ratio(text: str, size: int = 5) -> float:
    """Доля самой частой n-граммы — индикатор зацикливания генерации."""
    words = _WORD_RE.findall(normalize_for_match(text))
    if len(words) < size * 2:
        return 0.0
    grams: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - size + 1):
        gram = tuple(words[i:i + size])
        grams[gram] = grams.get(gram, 0) + 1
    total = len(words) - size + 1
    return max(grams.values()) / total


def truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …[обрезано]"
