import json
import re

# ---------------------------------------------------------------- P1: генерация

P1_PROMPT = """Напиши вежливое письмо клиенту интернет-магазина.
Ситуация: доставка заказа №48213 задерживается на 3 дня из-за сбоя у логистического партнёра.
Требования:
- извинись за задержку и кратко объясни причину;
- укажи номер заказа;
- предложи компенсацию — промокод на скидку 10% на следующий заказ;
- объём письма — не более 120 слов;
- подпись: «Служба поддержки магазина „Техномир“».
Выведи только текст письма."""


def eval_p1(text: str) -> dict:
    words = len(re.findall(r"\w+", text))
    checks = {
        "has_order_number": "48213" in text,
        "has_discount": bool(re.search(r"10\s*%|10\s*процент", text, re.I)),
        "has_apology": bool(re.search(r"извин|\bпрости|прос(им|ти) прощ|сожале", text, re.I)),
        "has_signature": "техномир" in text.lower(),
        "within_120_words": words <= 120,
    }
    return {"score": sum(checks.values()) / len(checks), "word_count": words, **checks}


# ---------------------------------------------------------------- P2: классификация

P2_LABELS = ["Billing", "Tech support", "Sales"]

P2_ITEMS = [
    ("С карты списали деньги дважды за одну подписку, верните, пожалуйста, лишний платёж.", "Billing"),
    ("После обновления приложение вылетает при запуске на Android 14.", "Tech support"),
    ("Хотим купить 50 лицензий для нашей компании, есть ли корпоративная скидка?", "Sales"),
    ("Не могу войти в личный кабинет: пишет «неверный токен», хотя пароль правильный.", "Tech support"),
    ("Пришлите, пожалуйста, закрывающие документы и счёт-фактуру за сентябрь.", "Billing"),
    ("Расскажите, чем тариф «Бизнес» отличается от «Про» и можно ли получить демо?", "Sales"),
]

P2_PROMPT = (
    "Ты — классификатор обращений в службу поддержки.\n"
    f"Отнеси каждое обращение ровно к одной метке из списка: {json.dumps(P2_LABELS, ensure_ascii=False)}.\n"
    'Ответ дай строго в формате JSON-объекта без пояснений: {"1": "<метка>", "2": "<метка>", ...}\n\n'
    "Обращения:\n"
    + "\n".join(f"{i}. {text}" for i, (text, _) in enumerate(P2_ITEMS, 1))
)


def eval_p2(text: str) -> dict:
    data = extract_json(text)
    if not isinstance(data, dict):
        return {"score": 0.0, "valid_json": False, "correct": 0, "invalid_labels": None}
    correct = invalid = 0
    for i, (_, gold) in enumerate(P2_ITEMS, 1):
        pred = str(data.get(str(i), data.get(i, ""))).strip()
        if pred not in P2_LABELS:
            invalid += 1
        if pred.lower() == gold.lower():
            correct += 1
    return {
        "score": correct / len(P2_ITEMS),
        "valid_json": True,
        "correct": correct,
        "invalid_labels": invalid,
    }


# ---------------------------------------------------------------- P3: извлечение

P3_DESCRIPTION = (
    "Беспроводные наушники Sony WH-1000XM5 в чёрном цвете — флагман с активным шумоподавлением. "
    "Время автономной работы до 30 часов, быстрая зарядка: 3 минуты дают 3 часа прослушивания. "
    "Вес всего 250 г. Поддерживаются кодеки LDAC и AAC, Bluetooth 5.2. "
    "Официальная гарантия 12 месяцев. Цена в нашем магазине — 32 990 руб., доставка бесплатно."
)

P3_GOLD = {
    "brand": "Sony",
    "model": "WH-1000XM5",
    "color": "чёрный",
    "price_rub": 32990,
    "weight_g": 250,
    "battery_hours": 30,
    "warranty_months": 12,
}

P3_PROMPT = (
    "Извлеки из описания товара характеристики и верни строго JSON-объект с полями:\n"
    "brand (строка), model (строка), color (строка, в именительном падеже мужского рода), "
    "price_rub (целое число), weight_g (целое число), battery_hours (целое число), "
    "warranty_months (целое число).\n"
    "Если значения нет в тексте — укажи null. Не добавляй пояснений.\n\n"
    f"Описание: {P3_DESCRIPTION}"
)


def _field_matches(pred, gold) -> bool:
    if pred is None:
        return False
    if isinstance(gold, int):
        # "32 990 руб." / 32990 / "32990" считаем одним и тем же
        digits = re.sub(r"\D", "", str(pred).split(".")[0])
        return digits == str(gold)
    pred_s = str(pred).strip().lower().replace("ё", "е")
    gold_s = gold.lower().replace("ё", "е")
    # для цвета допускаем другую форму слова: "черный"/"черного"/"черные"
    return pred_s == gold_s or pred_s[:4] == gold_s[:4]


def eval_p3(text: str) -> dict:
    data = extract_json(text)
    if not isinstance(data, dict):
        return {"score": 0.0, "valid_json": False, "fields_correct": 0}
    correct = sum(_field_matches(data.get(key), gold) for key, gold in P3_GOLD.items())
    return {"score": correct / len(P3_GOLD), "valid_json": True, "fields_correct": correct}


# ---------------------------------------------------------------- общее

def extract_json(text: str):
    """Достаёт первый JSON-объект из ответа (модели любят оборачивать его в ```json ... ```)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


PROMPTS = {
    "P1": {"task": "Генерация (письмо клиенту)", "text": P1_PROMPT, "evaluate": eval_p1},
    "P2": {"task": "Классификация обращений", "text": P2_PROMPT, "evaluate": eval_p2},
    "P3": {"task": "Извлечение полей товара в JSON", "text": P3_PROMPT, "evaluate": eval_p3},
}
