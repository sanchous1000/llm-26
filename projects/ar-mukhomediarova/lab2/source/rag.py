"""Шаг 6. RAG-пайплайн: векторизация запроса → поиск → сборка промпта → LLM → ответ с цитатами."""
from __future__ import annotations

import re
import time

from openai import OpenAI

from retriever import Retriever

SYSTEM_PROMPT = """Ты — ассистент по материалам научно-исследовательской работы (НИР) студентки ИТМО:
техническое задание, отчёты по семестрам, аналитический обзор, план НИР, репозиторий с кодом
и статьи-источники.

Правила:
1. Отвечай только на основе фрагментов из раздела «Контекст». Не добавляй фактов от себя.
2. После каждого утверждения ставь ссылку на фрагмент в квадратных скобках: [1], [2][3].
3. Если в контексте нет ответа, ответь ровно: «В документах НИР нет ответа на этот вопрос.»
4. Если фрагменты противоречат друг другу (например, разные версии отчёта), скажи об этом и укажи оба источника.
5. Отвечай по-русски, даже если фрагмент на английском. Числа, названия моделей и команды переноси точно."""


def locator(hit: dict) -> str:
    """Человекочитаемое место в источнике: раздел и страница/лист."""
    parts = []
    if hit.get("heading_path"):
        parts.append(" > ".join(hit["heading_path"][-2:]))
    ps, pe = hit.get("page_start"), hit.get("page_end")
    if ps:
        approx = "≈" if hit.get("page_mode") == "approx" else ""
        parts.append(f"стр. {approx}{ps}" + (f"–{pe}" if pe and pe != ps else ""))
    return ", ".join(parts)


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        loc = locator(h)
        blocks.append(f"[{i}] Источник: {h['title']}" + (f" | {loc}" if loc else "") + f"\n{h['text']}")
    return "\n\n".join(blocks)


def snippet(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


class RAGEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.retriever = Retriever(cfg)
        llm = cfg["llm"]
        self.llm = OpenAI(base_url=llm["base_url"], api_key="ollama", timeout=600)

    def ask(self, question: str, top_k: int | None = None, mode: str | None = None) -> dict:
        llm = self.cfg["llm"]
        k = top_k or llm.get("context_chunks", 5)
        t0 = time.perf_counter()
        hits, timings = self.retriever.search(question, top_k=k, mode=mode)
        prompt = f"Контекст:\n\n{build_context(hits)}\n\nВопрос: {question}\n\nОтвет (со ссылками [n]):"

        t1 = time.perf_counter()
        resp = self.llm.chat.completions.create(
            model=llm["model"],
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=llm.get("temperature", 0.1),
            max_tokens=llm.get("max_tokens", 700),
        )
        answer = (resp.choices[0].message.content or "").strip()
        timings["llm_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # ссылки вида [1], [1][3] и [1, 3]
        nums = [n for grp in re.findall(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", answer) for n in re.split(r"[,;]", grp)]
        used = sorted({int(n) for n in nums if 1 <= int(n) <= len(hits)})
        citations = [{
            "n": i,
            "cited": i in used,
            "source": h["title"],
            "path": h.get("url") or h.get("path"),
            "section": " > ".join(h.get("heading_path") or []),
            "page": (h.get("page_start") if h.get("page_mode") != "none" else None),
            "page_mode": h.get("page_mode"),
            "locator": locator(h),
            "snippet": snippet(h["text"], llm.get("snippet_chars", 300)),
            "score": h["score"],
            "chunk_id": h["chunk_id"],
        } for i, h in enumerate(hits, 1)]
        usage = resp.usage
        return {
            "question": question,
            "answer": answer,
            "citations": [c for c in citations if c["cited"]] or citations,
            "retrieved": citations,
            "mode": mode or self.cfg["search"]["mode"],
            "collection": self.retriever.collection,
            "model": llm["model"],
            "timings": timings,
            "tokens": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens} if usage else None,
        }


def format_answer(res: dict, show_context: bool = False) -> str:
    lines = [res["answer"], "", "Источники:"]
    for c in (res["retrieved"] if show_context else res["citations"]):
        mark = "*" if c["cited"] else " "
        lines.append(f" {mark}[{c['n']}] {c['source']}" + (f" — {c['locator']}" if c["locator"] else ""))
        lines.append(f"      «{c['snippet']}»")
    t = res["timings"]
    lines.append(f"\n(поиск {t['embed_ms'] + t['search_ms']:.0f} мс, LLM {t['llm_ms'] / 1000:.1f} с, "
                 f"режим {res['mode']}, {res['model']})")
    return "\n".join(lines)
