import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations

from config import RUNS_PATH, SUMMARY_PATH
from prompts import PROMPTS

PROMPTS_BY_ID = {prompt["id"]: prompt for prompt in PROMPTS}
SAMPLE_ANSWERS = 3
SAMPLE_SEED = 0


def load_runs(path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Нет файла {path}. Сначала запустите src/run_experiments.py."
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path} пуст. Сначала выполните прогоны.")
    return rows


def extract_json(text: str):
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    start_positions = [pos for pos in (candidate.find("["), candidate.find("{")) if pos >= 0]
    if not start_positions:
        return None
    blob = candidate[min(start_positions):]
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(blob)
    except json.JSONDecodeError:
        return None
    return value


def normalize(value) -> str:
    text = str(value).lower().replace("ё", "е")
    return re.sub(r"\s+", "", text)


def score_p1(text: str, expect: dict) -> dict:
    missing = [item for item in expect["contains"] if item not in text]
    percents = [float(item.replace(",", ".")) for item in re.findall(r"(\d+(?:[.,]\d+)?)\s*%", text)]
    extra_percent = any(percent != 10 for percent in percents)
    codes = set(re.findall(r"\b[A-Z][A-Z0-9]{3,}\b", text))
    extra_codes = sorted(codes - {"SORRY10"})
    return {
        "ok": not missing and not extra_percent and not extra_codes,
        "missing": missing,
        "extra_percent": extra_percent,
        "extra_codes": extra_codes,
        "detail": _p1_detail(missing, extra_percent, extra_codes),
    }


def _p1_detail(missing, extra_percent, extra_codes) -> str:
    parts = []
    if missing:
        parts.append("нет: " + ", ".join(missing))
    if extra_percent:
        parts.append("другая скидка в процентах")
    if extra_codes:
        parts.append("другие коды: " + ", ".join(extra_codes))
    return "; ".join(parts) if parts else "факты на месте"


def score_p2(text: str, expect: dict) -> dict:
    parsed = extract_json(text)
    labels = parsed if isinstance(parsed, list) else None
    expected = expect["labels"]
    ok = labels == expected
    return {
        "ok": ok,
        "parsed": labels,
        "detail": "совпало" if ok else f"получено {labels!r}, ожидалось {expected!r}",
    }


def score_p3(text: str, expect: dict) -> dict:
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "matched": 0,
            "total": len(expect["fields"]),
            "has_summary": False,
            "detail": "JSON-объект не найден",
        }
    matched = []
    missed = []
    for key, expected in expect["fields"].items():
        actual = parsed.get(key, "")
        if normalize(expected) in normalize(actual):
            matched.append(key)
        else:
            missed.append(key)
    summary = parsed.get("summary")
    has_summary = isinstance(summary, str) and summary.strip() != ""
    total = len(expect["fields"])
    return {
        "ok": len(matched) == total and has_summary,
        "matched": len(matched),
        "total": total,
        "has_summary": has_summary,
        "detail": (
            f"поля {len(matched)}/{total}"
            + ("" if not missed else ", нет: " + ", ".join(missed))
            + ("" if has_summary else "; нет summary")
        ),
    }


def score_row(row: dict) -> dict:
    prompt = PROMPTS_BY_ID.get(row["prompt_id"])
    if prompt is None:
        return {"ok": False, "detail": "неизвестный промпт"}
    text = row.get("response_text") or ""
    if prompt["id"] == "P1":
        return score_p1(text, prompt["expect"])
    if prompt["id"] == "P2":
        return score_p2(text, prompt["expect"])
    if prompt["id"] == "P3":
        return score_p3(text, prompt["expect"])
    return {"ok": False, "detail": "нет проверки"}


def fmt_num(value, digits: int) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def mean(values: list) -> float | None:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def group_key(row: dict) -> tuple:
    return (row["model"], row["prompt_id"], row["mode"])


def finish_tally(reasons: list) -> str:
    counts = Counter(reason or "—" for reason in reasons)
    return ", ".join(f"{reason}×{count}" for reason, count in counts.most_common())


def stability(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["prompt_id"], row["mode"])].append(row["response_text"])
    report = []
    for key in sorted(groups):
        texts = groups[key]
        pair_count = 0
        match_count = 0
        for left, right in combinations(texts, 2):
            pair_count += 1
            if left == right:
                match_count += 1
        report.append(
            {
                "model": key[0],
                "prompt_id": key[1],
                "mode": key[2],
                "repeats": len(texts),
                "pairs": pair_count,
                "matches": match_count,
            }
        )
    return report


def sample_answers(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    rng = random.Random(SAMPLE_SEED)
    chosen = []
    for key in sorted(grouped):
        group = grouped[key]
        if len(group) <= SAMPLE_ANSWERS:
            picked = group
        else:
            picked = rng.sample(group, SAMPLE_ANSWERS)
        chosen.extend(sorted(picked, key=lambda row: row["repeat"]))
    return chosen


def render(rows: list[dict]) -> str:
    scored = [(row, score_row(row)) for row in rows]
    grouped = defaultdict(list)
    for row, score in scored:
        grouped[group_key(row)].append((row, score))

    lines = ["# Сводка прогонов", ""]
    lines.append("Числа в таблице — средние по повторам одной связки модель × промпт × режим.")
    lines.append("")
    lines.append(
        "| Модель | Промпт | Режим | N | Ввод | Вывод | Секунды | Ток/с | Символы | Успех | finish |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key in sorted(grouped):
        pairs = grouped[key]
        usages = [row.get("usage") or {} for row, _score in pairs]
        finishes = [row.get("finish_reason") for row, _score in pairs]
        ok_count = sum(1 for _row, score in pairs if score["ok"])
        lines.append(
            "| {model} | {prompt_id} | {mode} | {count} | {prompt_tokens} | {completion_tokens} | {latency} | {tps} | {chars} | {ok} | {finish} |".format(
                model=key[0],
                prompt_id=key[1],
                mode=key[2],
                count=len(pairs),
                prompt_tokens=fmt_num(mean(usage.get("prompt_tokens") for usage in usages), 1),
                completion_tokens=fmt_num(mean(usage.get("completion_tokens") for usage in usages), 1),
                latency=fmt_num(mean(row.get("latency_s") for row, _score in pairs), 2),
                tps=fmt_num(mean(row.get("tokens_per_second") for row, _score in pairs), 1),
                chars=fmt_num(mean(len(row.get("response_text") or "") for row, _score in pairs), 0),
                ok=f"{ok_count}/{len(pairs)}",
                finish=finish_tally(finishes),
            )
        )

    lines.extend(["", "## Стабильность повторов", ""])
    lines.append("Доля пар повторов с дословно одинаковым текстом внутри одного режима.")
    lines.append("")
    lines.append("| Модель | Промпт | Режим | Совпавшие пары |")
    lines.append("| --- | --- | --- | --- |")
    for item in stability(rows):
        lines.append(
            f"| {item['model']} | {item['prompt_id']} | {item['mode']} | {item['matches']}/{item['pairs']} |"
        )

    lines.extend(["", "## Ответы", ""])
    lines.append(
        f"По каждой связке показаны {SAMPLE_ANSWERS} случайных повтора. "
        f"Выборка фиксирована, seed={SAMPLE_SEED}."
    )
    lines.append("")
    for row in sample_answers(rows):
        lines.append(
            f"### {row['model']} / {row['prompt_id']} / {row['mode']} / #{row['repeat']}"
        )
        lines.append("")
        lines.append(row.get("response_text") or "")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    rows = load_runs(RUNS_PATH)
    summary = render(rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Сохранено в {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
