"""Автоматическая оценка ответов под каждую задачу."""
import json
import re

from prompts import (
    P1_FACTS, P1_MAX_WORDS,
    P2_ITEMS, P2_LABELS,
    P3_GOLD, P3_SUMMARY_MAX_WORDS,
)


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def repeated_trigram_share(text: str) -> float:
    """Доля повторяющихся словесных триграмм внутри ответа (0 — повторов нет)."""
    w = words(text)
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return round(1 - len(set(grams)) / len(grams), 3) if grams else 0.0


# ---------- P1 ----------
def score_p1(text: str) -> dict:
    found = {name: bool(re.search(rx, text, re.IGNORECASE)) for name, rx in P1_FACTS.items()}
    n_words = len(words(text))
    return {
        "score": round(sum(found.values()) / len(found), 3),  # доля обязательных фактов в письме
        "missing_facts": [k for k, v in found.items() if not v],
        "word_count": n_words,
        "length_ok": n_words <= P1_MAX_WORDS,
    }


# ---------- P2 ----------
_LABEL_RX = re.compile(
    r"^\W*(\d+)\W+(" + "|".join(re.escape(l) for l in P2_LABELS) + r")\b",
    re.IGNORECASE | re.MULTILINE,
)


def score_p2(text: str) -> dict:
    canon = {l.lower(): l for l in P2_LABELS}
    pred = {}
    for num, label in _LABEL_RX.findall(text):
        pred.setdefault(int(num), canon[label.lower()])
    gold = {i: lbl for i, (_, lbl) in enumerate(P2_ITEMS, 1)}
    correct = sum(pred.get(i) == g for i, g in gold.items())
    non_empty_lines = [l for l in text.strip().splitlines() if l.strip()]
    return {
        "score": round(correct / len(gold), 3),  # accuracy
        "predicted": pred,
        "errors": {i: {"gold": g, "pred": pred.get(i)} for i, g in gold.items() if pred.get(i) != g},
        # формат соблюдён: ровно по строке на обращение и ничего лишнего
        "format_ok": len(pred) == len(gold) and len(non_empty_lines) == len(gold),
    }


# ---------- P3 ----------
def _extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _to_number(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = v.replace(" ", "").replace(" ", "").replace(",", ".")
        m = re.search(r"-?\d+(\.\d+)?", s)
        return float(m.group()) if m else None
    return None


_CURRENCY_OK = {"rub", "rur", "руб", "рубль", "рубли", "рублей", "₽"}


def _field_ok(key, pred, gold) -> bool:
    if gold is None:
        return pred in (None, "", "null")
    if key == "currency":
        return str(pred).strip().lower().rstrip(".") in _CURRENCY_OK
    if isinstance(gold, (int, float)):
        num = _to_number(pred)
        return num is not None and abs(num - gold) < 1e-6
    return str(pred).strip().strip("«»\"'").lower() == str(gold).lower()


def score_p3(text: str) -> dict:
    data = _extract_json(text)
    if not isinstance(data, dict):
        return {"score": 0.0, "json_valid": False, "wrong_fields": list(P3_GOLD), "hallucinated_warranty": None}
    wrong = [k for k, g in P3_GOLD.items() if not _field_ok(k, data.get(k), g)]
    summary = data.get("summary") or ""
    return {
        "score": round(1 - len(wrong) / len(P3_GOLD), 3),  # доля верно извлечённых полей
        "json_valid": True,
        # строгий формат: ответ — чистый JSON, без пояснений и ```-обёрток
        "json_only": text.strip().startswith("{") and text.strip().endswith("}"),
        "wrong_fields": wrong,
        "hallucinated_warranty": data.get("warranty_months") not in (None, "", "null"),
        "summary_words": len(words(str(summary))),
        "summary_ok": 0 < len(words(str(summary))) <= P3_SUMMARY_MAX_WORDS,
    }


SCORERS = {
    "P1_generation": score_p1,
    "P2_classification": score_p2,
    "P3_extraction": score_p3,
}


def evaluate(prompt_id: str, text: str) -> dict:
    result = SCORERS[prompt_id](text or "")
    result["repeated_trigrams"] = repeated_trigram_share(text or "")
    return result
