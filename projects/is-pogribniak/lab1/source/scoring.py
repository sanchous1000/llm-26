"""Оценка одного ответа под задачу по эталонам из prompts/gold.json.
P1, P3 (свободный текст) — чеклист обязательных пунктов, запреты, объём, выдуманные числа.
P2 (классификация)       — accuracy по меткам, отдельно — обращения не по теме (Unknown).
"""

import json
import re

from .config import GOLD

WORD_RX = r"[\wёЁ]+(?:-[\wёЁ]+)*"


def numbers(text: str) -> set:
    return set(re.findall(r"\d+", re.sub(r"(?<=\d)[ ,  ](?=\d{3}(?!\d))", "", text)))


def json_items(text: str) -> tuple[list, bool]:
    """Объекты из JSON-массива в ответе. Если массив битый (например, у одного элемента
    нет "label"), достаём уцелевшие объекты по одному; второе значение — валиден ли JSON."""
    decoder = json.JSONDecoder()
    start = text.find("[")
    if start >= 0:
        try:
            data = decoder.raw_decode(text, start)[0]
            if isinstance(data, list):
                return data, True
        except json.JSONDecodeError:
            pass
    items, i = [], 0
    while (i := text.find("{", i)) >= 0:
        try:
            obj, i = decoder.raw_decode(text, i)
            items.append(obj)
        except json.JSONDecodeError:
            i += 1
    return items, False


def score_text(pid: str, text: str, prompt_text: str) -> dict:
    g = GOLD[pid]
    words = re.findall(WORD_RX, text)
    hits = {name: bool(re.search(rx, text, re.IGNORECASE)) for name, rx in g["checklist"].items()}
    banned = {name: bool(re.search(rx, text, re.IGNORECASE)) for name, rx in g.get("forbidden", {}).items()}
    checks = {**hits, **{name: not v for name, v in banned.items()}}
    checks["объём"] = g["words_min"] <= len(words) <= g["words_max"]
    letters = [c for c in text if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
    if "max_caps_share" in g:
        checks["без капслока"] = caps <= g["max_caps_share"]

    invented = sorted(numbers(text) - numbers(prompt_text) - set(g["allowed_numbers"]))
    gate_failed = [k for k in g.get("gate", []) if not checks[k]]
    toxic_rx = "|".join(f"(?:{g['forbidden'][k]})" for k in g.get("toxic_count", []))
    toxic = [w for w in words if toxic_rx and re.search(toxic_rx, w, re.IGNORECASE)]

    return {
        "quality": 0.0 if gate_failed else sum(checks.values()) / len(checks),
        "missed": ", ".join(k for k, v in checks.items() if not v),
        "words": len(words),
        "forbidden_hits": sum(banned.values()),
        "refused": banned.get("отказ", False),
        "toxic_words": len(toxic),
        "toxic_list": ", ".join(toxic),
        "invented_numbers": ", ".join(invented),
        "hallucinations": len(invented),
        "parse_ok": True,
        "canonical": text,
    }


def score_p2(text: str) -> dict:
    g = GOLD["P2"]
    items, json_ok = json_items(text)
    pred = {str(it["id"]): str(it["label"]).strip()
            for it in items if isinstance(it, dict) and "id" in it and "label" in it}
    correct = [k for k, v in g["gold"].items() if pred.get(k) == v]
    oos = g["out_of_scope"]
    in_scope = [k for k in g["gold"] if k not in oos]
    invalid = [v for v in pred.values() if v not in g["labels"]]
    forced = [k for k in oos if pred.get(k) in set(g["labels"]) - {"Unknown"}]
    return {
        "quality": len(correct) / len(g["gold"]),
        "missed": ", ".join(f"{k}:{pred.get(k)}" for k in g["gold"] if k not in correct),
        "acc_in_scope": sum(k in correct for k in in_scope) / len(in_scope),
        "unknown_recall": sum(pred.get(k) == "Unknown" for k in oos) / len(oos),
        "false_unknown": sum(pred.get(k) == "Unknown" for k in in_scope),
        "forced_labels": len(forced),
        "hallucinations": len(invalid) + len(forced),
        "parse_ok": json_ok,
        "canonical": json.dumps([pred.get(k) for k in g["gold"]], ensure_ascii=False) if pred else text,
    }


def score(prompt_id: str, response: str, messages: list) -> dict:
    if prompt_id == "P2":
        return score_p2(response)
    return score_text(prompt_id, response, "\n".join(m["content"] for m in messages))
