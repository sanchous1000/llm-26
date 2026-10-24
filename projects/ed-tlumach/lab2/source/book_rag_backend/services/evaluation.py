"""Шаг 5: прогон вопросов через retrieval, без расчёта метрик — метрики считаются отдельно
(вручную или, например, Claude по присланному дампу), чтобы видеть, что реально нашлось по
каждому вопросу, а не доверять агрегированным цифрам вслепую.
"""
import json
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for q in qs:
        if not q.get("question") or not q.get("gold"):
            raise ValueError(f"В вопросе нет question или gold: {q}")
    return qs


async def retrieval_dump(rag, questions: list[dict], k: int = 10) -> list[dict]:
    """Прогоняет вопросы через retrieval и возвращает всё найденное (source/page/score/сниппет
    на каждый из топ-k чанков), вместе с gold из самого вопроса — для ручной/внешней проверки."""
    dump = []
    for q in questions:
        hits = await rag.retrieve(q["question"], k=k)
        retrieved = [
            {"rank": i + 1, "score": round(h["score"], 4), "source": h["metadata"].get("source"),
             "page": h["metadata"].get("page"), "snippet": " ".join(h["content"].split())[:300]}
            for i, h in enumerate(hits)
        ]
        dump.append({"id": q.get("id"), "question": q["question"], "gold": q.get("gold", []), "retrieved": retrieved})
    return dump
