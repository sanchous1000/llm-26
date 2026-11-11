from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests

OLLAMA_HOST = "http://127.0.0.1:11434"
TIMEOUT_S = 600
KEEP_ALIVE = "10m"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS: list[str] | None = ["qwen3:8b", "gemma3:4b", "mistral:7b"]

MODES: dict[str, dict[str, Any]] = {
    "A_base": {
        "label": "Базовый",
        "options": {},
    },
    "B_tune": {
        "label": "Тюнинг",
        "options": {
            "temperature": 0.1,
            "top_k": 20,
            "top_p": 0.7,
            "repeat_penalty": 1.25,
        },
    },
}

LABELS = ["Биллинг", "Техподдержка", "Продажи"]

P2_ITEMS = [
    {"id": 1, "text": "Мне дважды списали оплату за тариф Про, верните деньги.", "gold": "Биллинг"},
    {"id": 2, "text": "VPN на ноутбуке после обновления не подключается, ошибка 809.", "gold": "Техподдержка"},
    {"id": 3, "text": "Нужно 30 лицензий для отдела продаж, есть скидка при годовой оплате?", "gold": "Продажи"},
    {"id": 4, "text": "Счёт за август не пришёл на почту, пришлите повторно.", "gold": "Биллинг"},
]

P3_SOURCE = (
    "Беспроводные наушники Aurora X2. Драйвер 40 мм, Bluetooth 5.3, ANC до 32 дБ, "
    "до 36 ч работы с кейсом, вес 255 г, цвет графит. Цена 8 990 ₽. "
    "В комплекте USB-C кабель и чехол. Гарантия 24 месяца. Страна бренда — не указана."
)

P3_GOLD = {
    "name": "Aurora X2",
    "price_rub": 8990,
    "bluetooth": "5.3",
    "anc_db": 32,
    "battery_hours": 36,
    "weight_g": 255,
    "warranty_months": 24,
    "color": "графит",
}

PROMPTS: dict[str, dict[str, Any]] = {
    "P1": {
        "task": "generation",
        "title": "Генерация: вежливое письмо",
        "text": (
            "Напиши вежливое деловое письмо клиенту компании «СеверТех».\n"
            "Факты (нельзя выдумывать другие):\n"
            "- поставка серверов задерживается на 5 рабочих дней из-за логистики;\n"
            "- новая дата прибытия — пятница следующей недели;\n"
            "- компенсация — скидка 7% на следующую услугу сопровождения.\n\n"
            "Требования:\n"
            "- 120–180 слов;\n"
            "- обращение на «вы», без канцелярита и без извинений «за любые неудобства» как единственной сути;\n"
            "- не добавляй телефон, email, ФИО менеджера и трек-номер, если их нет во входных фактах;\n"
            "- закончи письмом с предложением подтвердить получение."
        ),
    },
    "P2": {
        "task": "classification",
        "title": "Классификация обращений",
        "gold": {item["id"]: item["gold"] for item in P2_ITEMS},
        "text": (
            "Классифицируй каждое обращение ровно одной меткой из списка:\n"
            f"{json.dumps(LABELS, ensure_ascii=False)}\n\n"
            "Правила:\n"
            '- ответь ТОЛЬКО JSON-массивом вида [{"id": 1, "label": "..."}, ...];\n'
            "- никаких комментариев, markdown и текста вокруг JSON;\n"
            "- если сомневаешься, выбери наиболее вероятную метку, не выдумывай новые.\n\n"
            "Обращения:\n"
            + "\n".join(f'{item["id"]}. «{item["text"]}»' for item in P2_ITEMS)
        ),
    },
    "P3": {
        "task": "extraction",
        "title": "Извлечение полей + краткое резюме",
        "gold": P3_GOLD,
        "text": (
            "Из описания товара извлеки поля и сделай краткое резюме.\n\n"
            f"Описание:\n«{P3_SOURCE}»\n\n"
            "Верни ТОЛЬКО JSON-объект со ключами:\n"
            "{\n"
            '  "name": строка,\n'
            '  "price_rub": число,\n'
            '  "bluetooth": строка,\n'
            '  "anc_db": число,\n'
            '  "battery_hours": число,\n'
            '  "weight_g": число,\n'
            '  "warranty_months": число,\n'
            '  "color": строка,\n'
            '  "country": строка или null если не указано,\n'
            '  "summary": ровно 2 предложения на русском\n'
            "}\n\n"
            "Не выдумывай характеристики, которых нет в тексте. Никакого текста вокруг JSON."
        ),
    },
}

JSON_RE = re.compile(r"(\[.*\]|\{.*\})", re.DOTALL)


def ollama_get(path: str) -> Any:
    r = requests.get(f"{OLLAMA_HOST}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def ns_to_s(ns: int | None) -> float | None:
    if ns is None:
        return None
    return ns / 1e9


def tok_per_s(count: int | None, duration_ns: int | None) -> float | None:
    if not count or not duration_ns:
        return None
    return count / (duration_ns / 1e9)


def chat_ollama(model: str, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
    }
    if options:
        payload["options"] = options

    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=TIMEOUT_S)
    wall_s = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()

    message = data.get("message") or {}
    prompt_tokens = data.get("prompt_eval_count")
    output_tokens = data.get("eval_count")
    prompt_ns = data.get("prompt_eval_duration")
    eval_ns = data.get("eval_duration")

    return {
        "text": message.get("content", ""),
        "wall_s": wall_s,
        "total_s": ns_to_s(data.get("total_duration")),
        "load_s": ns_to_s(data.get("load_duration")),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_eval_s": ns_to_s(prompt_ns),
        "eval_s": ns_to_s(eval_ns),
        "prompt_tok_s": tok_per_s(prompt_tokens, prompt_ns),
        "gen_tok_s": tok_per_s(output_tokens, eval_ns),
        "done_reason": data.get("done_reason"),
    }


def extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    m = JSON_RE.search(text)
    if m:
        candidates.append(m.group(1))
    for raw in candidates:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    return None


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or "", flags=re.UNICODE))


def trigram_repeat_ratio(text: str) -> float | None:
    words = re.findall(r"\w+", text or "", flags=re.UNICODE)
    if len(words) < 6:
        return None
    grams = list(zip(words, words[1:], words[2:]))
    return 1.0 - len(set(grams)) / len(grams)


def score_p1(text: str) -> dict[str, Any]:
    wc = word_count(text)
    lowered = (text or "").lower()
    invented_contacts = bool(re.search(r"\+7|@|www\.|http", lowered))
    has_discount = "7%" in text or "7 %" in text or "семь процентов" in lowered
    has_delay = "5" in text and ("дн" in lowered or "рабоч" in lowered)
    return {
        "auto_ok": 120 <= wc <= 180 and has_discount and has_delay and not invented_contacts,
        "word_count": wc,
        "length_ok": 120 <= wc <= 180,
        "has_required_facts": has_discount and has_delay,
        "possible_hallucination": invented_contacts,
        "accuracy": None,
    }


def score_p2(text: str) -> dict[str, Any]:
    parsed = extract_json(text)
    gold = PROMPTS["P2"]["gold"]
    pred: dict[int, str] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "id" in item and "label" in item:
                pred[int(item["id"])] = str(item["label"]).strip()
    correct = sum(pred.get(i) == label for i, label in gold.items())
    valid_label = all(v in LABELS for v in pred.values()) if pred else False
    acc = correct / len(gold) if gold else None
    return {
        "auto_ok": acc == 1.0 and valid_label and set(pred) == set(gold),
        "accuracy": acc,
        "json_ok": isinstance(parsed, list),
        "valid_labels": valid_label,
        "pred": pred,
        "possible_hallucination": bool(pred) and not valid_label,
    }


def normalize(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip().lower().replace(",", ".")
    return value


def score_p3(text: str) -> dict[str, Any]:
    parsed = extract_json(text)
    gold = PROMPTS["P3"]["gold"]
    if not isinstance(parsed, dict):
        return {
            "auto_ok": False,
            "accuracy": 0.0,
            "json_ok": False,
            "field_hits": {},
            "possible_hallucination": False,
        }
    hits = {}
    for key, expected in gold.items():
        got = parsed.get(key)
        if key == "name":
            hits[key] = expected.lower() in str(got or "").lower()
        elif isinstance(expected, (int, float)):
            try:
                hits[key] = float(str(got).replace(" ", "").replace("₽", "")) == float(expected)
            except (TypeError, ValueError):
                hits[key] = False
        else:
            hits[key] = normalize(got) == normalize(expected) or normalize(expected) in normalize(str(got))
    country = parsed.get("country")
    country_ok = country in (None, "", "null") or str(country).lower() in {"не указана", "нет", "unknown"}
    summary = str(parsed.get("summary") or "")
    sentences = [s for s in re.split(r"[.!?]+\s*", summary) if s.strip()]
    acc = sum(hits.values()) / len(hits)
    return {
        "auto_ok": acc == 1.0 and country_ok and 1 <= len(sentences) <= 3,
        "accuracy": acc,
        "json_ok": True,
        "field_hits": hits,
        "country_ok": country_ok,
        "summary_sentences": len(sentences),
        "possible_hallucination": not country_ok,
    }


SCORERS = {"P1": score_p1, "P2": score_p2, "P3": score_p3}


def resolve_models() -> list[str]:
    print("Нативный API:", requests.get(OLLAMA_HOST, timeout=10).text.strip())
    ollama_get("/v1/models")
    local = ollama_get("/api/tags")
    discovered = [m["name"] for m in local.get("models", [])]
    if not discovered:
        raise RuntimeError("В Ollama нет локальных моделей")
    models = MODELS if MODELS is not None else discovered[:3]
    missing = [m for m in models if m not in discovered]
    if missing:
        raise RuntimeError(f"Модели не найдены в Ollama: {missing}. Доступны: {discovered}")
    print("Модели:", models)
    return models


def run_all(models: list[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    n_total = len(models) * len(PROMPTS) * len(MODES)
    n_done = 0
    for model in models:
        for pid, spec in PROMPTS.items():
            for mode_id, mode in MODES.items():
                n_done += 1
                print(f"[{n_done}/{n_total}] {model} | {pid} | {mode_id}", flush=True)
                result = chat_ollama(model, spec["text"], mode["options"])
                runs.append(
                    {
                        "model": model,
                        "prompt_id": pid,
                        "task": spec["task"],
                        "mode_id": mode_id,
                        "mode_label": mode["label"],
                        "options": mode["options"],
                        "response": result["text"],
                        "wall_s": result["wall_s"],
                        "total_s": result["total_s"],
                        "load_s": result["load_s"],
                        "prompt_tokens": result["prompt_tokens"],
                        "output_tokens": result["output_tokens"],
                        "prompt_eval_s": result["prompt_eval_s"],
                        "eval_s": result["eval_s"],
                        "prompt_tok_s": result["prompt_tok_s"],
                        "gen_tok_s": result["gen_tok_s"],
                        "done_reason": result["done_reason"],
                    }
                )
                speed = f"{result['gen_tok_s']:.1f} tok/s" if result["gen_tok_s"] else "n/a"
                print(
                    f"    wall={result['wall_s']:.2f}s  in={result['prompt_tokens']} tok  "
                    f"out={result['output_tokens']} tok  gen={speed}",
                    flush=True,
                )
    return runs


def save_responses(runs: list[dict[str, Any]]) -> None:
    md_lines = ["# Ответы моделей\n"]
    for row in runs:
        md_lines.append(f"## {row['model']} · {row['prompt_id']} · {row['mode_id']} ({row['mode_label']})\n")
        md_lines.append(
            f"- wall: `{row['wall_s']:.2f}s` · "
            f"prompt: `{row['prompt_tokens']}` tok · "
            f"output: `{row['output_tokens']}` tok · "
            f"gen: `{row['gen_tok_s']}` tok/s\n"
        )
        md_lines.append("```\n" + (row["response"] or "") + "\n```\n")
    (RESULTS_DIR / "responses.md").write_text("\n".join(md_lines), encoding="utf-8")


def save_prompts() -> None:
    lines = ["# Тестовые запросы\n"]
    for pid, spec in PROMPTS.items():
        lines.append(f"## {pid}. {spec['title']}\n")
        lines.append(spec["text"] + "\n")
    (RESULTS_DIR / "prompts.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(runs: list[dict[str, Any]]) -> pd.DataFrame:
    analyzed: list[dict[str, Any]] = []
    for row in runs:
        score = SCORERS[row["prompt_id"]](row["response"])
        analyzed.append(
            {
                **{k: v for k, v in row.items() if k not in {"response", "options"}},
                "options": json.dumps(row["options"], ensure_ascii=False),
                "chars": len(row["response"] or ""),
                "words": word_count(row["response"]),
                "repeat_3gram": trigram_repeat_ratio(row["response"]),
                "truncated": row["done_reason"] == "length",
                "auto_ok": score.get("auto_ok"),
                "accuracy": score.get("accuracy"),
                "possible_hallucination": bool(score.get("possible_hallucination")),
                "score_detail": {k: v for k, v in score.items() if k != "pred"},
                "human_adequacy_1to5": None,
            }
        )
    return pd.DataFrame(analyzed)


def print_summary(df: pd.DataFrame) -> None:
    by_prompt = (
        df.groupby(["prompt_id", "mode_label"], dropna=False)
        .agg(
            output_tokens=("output_tokens", "mean"),
            wall_s=("wall_s", "mean"),
            gen_tok_s=("gen_tok_s", "mean"),
            prompt_tok_s=("prompt_tok_s", "mean"),
            accuracy=("accuracy", "mean"),
            auto_ok=("auto_ok", "mean"),
            repeat_3gram=("repeat_3gram", "mean"),
            truncated=("truncated", "mean"),
            hallucinations=("possible_hallucination", "mean"),
        )
        .reset_index()
    )
    by_model = (
        df.groupby(["model", "mode_label"], dropna=False)
        .agg(
            output_tokens=("output_tokens", "mean"),
            wall_s=("wall_s", "mean"),
            gen_tok_s=("gen_tok_s", "mean"),
            accuracy=("accuracy", "mean"),
            auto_ok=("auto_ok", "mean"),
        )
        .reset_index()
    )
    print(by_prompt.round(3).to_string(index=False))
    print(by_model.round(3).to_string(index=False))


def save_charts(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    columns = ["model", "mode_label"]
    df.pivot_table(index="prompt_id", columns=columns, values="output_tokens").plot(
        kind="bar", ax=axes[0], legend=False, title="Длина ответа, tok"
    )
    df.pivot_table(index="prompt_id", columns=columns, values="wall_s").plot(
        kind="bar", ax=axes[1], legend=False, title="Время отклика, s"
    )
    df.pivot_table(index="prompt_id", columns=columns, values="gen_tok_s").plot(
        kind="bar", ax=axes[2], title="Скорость генерации, tok/s"
    )
    axes[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "charts.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    models = resolve_models()
    runs = run_all(models)
    save_responses(runs)
    save_prompts()
    df = analyze(runs)
    print_summary(df)
    save_charts(df)


if __name__ == "__main__":
    main()
