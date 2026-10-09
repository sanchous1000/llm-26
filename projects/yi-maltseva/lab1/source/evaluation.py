from __future__ import annotations

import json
import re

from test_cases import PROMPTS


def score_answer(prompt_id: str, answer: str) -> tuple[float | None, dict]:
    text = answer or ""
    lowered = text.casefold()

    if prompt_id == "P1":
        checks = {
            "polite_greeting": bool(re.search(r"здравствуйте|добрый день|уважаемые", lowered)),
            "reason_for_change": bool(re.search(r"в связи|из-за|по причине|техническ", lowered)),
            "date_or_time": bool(re.search(r"13\s+октябр|вторник|15\s*[:.]\s*00|15 часов", lowered)),
            "apology_or_confirmation": bool(re.search(r"извин|неудобств|подтверд.*получ|подтверждени.*получ|благодар", lowered)),
        }
        return sum(checks.values()) / len(checks), checks

    if prompt_id == "P2":
        found = []
        for match in re.finditer(r"Billing|Tech\s+support|Sales", text, flags=re.IGNORECASE):
            normalized = re.sub(r"\s+", " ", match.group(0)).casefold()
            label = {"billing": "Billing", "tech support": "Tech support", "sales": "Sales"}.get(normalized)
            if label:
                found.append(label)
        expected = PROMPTS["P2"]["expected"]
        correct = sum(i < len(found) and found[i] == expected[i] for i in range(len(expected)))
        return correct / len(expected), {
            "predicted_labels": found,
            "expected_labels": expected,
            "correct_labels": correct,
            "number_of_labels_correct": len(found) == len(expected),
            "exact_sequence": found == expected,
        }

    if prompt_id == "P3":
        expected = PROMPTS["P3"]["expected"]
        parsed = None
        try:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end + 1])
        except (json.JSONDecodeError, TypeError):
            parsed = None

        field_hits = {}
        if isinstance(parsed, dict):
            for key, expected_value in expected.items():
                actual = parsed.get(key)
                if isinstance(expected_value, str):
                    field_hits[key] = isinstance(actual, str) and expected_value.casefold() in actual.casefold()
                elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
                    field_hits[key] = abs(float(actual) - float(expected_value)) < 1e-6
                elif isinstance(actual, str):
                    try:
                        field_hits[key] = abs(float(actual.replace(",", ".").replace(" ", "")) - float(expected_value)) < 1e-6
                    except ValueError:
                        field_hits[key] = False
                else:
                    field_hits[key] = False
            parse_ok = True
        else:
            compact = re.sub(r"\s+", " ", lowered)
            cues = {
                "model": r"novabook\s+air",
                "cpu": r"i5\s*[- ]?1235u",
                "ram_gb": r"16\s*(гб|gb)",
                "storage_gb": r"512\s*(гб|gb)|512\s*гб\s*ssd|ssd\s*(на\s*)?512",
                "screen_inches": r"14\s*(дюйм|inch|\")",
                "weight_kg": r"1[,.]3\s*(кг|kg)",
                "battery_hours": r"14\s*(час|ч\b|hours?)",
                "price_rub": r"79\s?990|79990",
            }
            field_hits = {key: bool(re.search(pattern, compact)) for key, pattern in cues.items()}
            parse_ok = False
        return sum(field_hits.values()) / len(expected), {
            "json_parse_success": parse_ok,
            "fields_correct": field_hits,
            "correct_fields": sum(field_hits.values()),
            "total_fields": len(expected),
        }

    return None, {"error": "Неизвестный тип теста"}
