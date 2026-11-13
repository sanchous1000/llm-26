"""Эталонная разметка: сопоставление чанков с «местами» в корпусе и метрики retrieval."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from settings import LAB_DIR, nfc

QUESTIONS = LAB_DIR / "eval" / "questions.yaml"


def norm(text: str) -> str:
    """Регистр, NFC, «0,733» ≡ «0.733», «45 581» ≡ «45581», без markdown-экранирования."""
    t = nfc(str(text)).lower().replace("\\", "").replace("ё", "е")
    t = re.sub(r"(?<=\d)[   ](?=\d{3}\b)", "", t)
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)
    return re.sub(r"\s+", " ", t)


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        qs = yaml.safe_load(f)["questions"]
    for q in qs:  # группа: один спец → список из одного
        q["relevant"] = [g if isinstance(g, list) else [g] for g in q["relevant"]]
    return qs


def spec_matches(chunk: dict, spec: dict) -> bool:
    if "doc" in spec:
        target = norm(spec["doc"])
        if target not in norm(chunk.get("title", "")) and target not in norm(chunk.get("path", "")):
            return False
    if "section" in spec:
        target = norm(spec["section"])
        if not any(target in norm(s) for s in (chunk.get("sections") or [])):
            return False
    if "contains" in spec:
        text = norm(chunk.get("text", ""))
        if not any(norm(c) in text for c in spec["contains"]):
            return False
    if "pages" in spec:
        a, b = spec["pages"]
        ps, pe = chunk.get("page_start"), chunk.get("page_end")
        if ps is None or pe < a or ps > b:
            return False
    return True


def groups_hit(chunk: dict, question: dict) -> set[int]:
    return {gi for gi, group in enumerate(question["relevant"]) if any(spec_matches(chunk, s) for s in group)}


def retrieval_metrics(hits: list[dict], question: dict, ks=(5, 10)) -> dict:
    """Recall@k — доля групп релевантности, покрытых top-k; Precision@k — доля релевантных чанков
    в top-k; MRR — 1/ранг первого релевантного чанка; Hit@k — найден ли хоть один."""
    n_groups = len(question["relevant"])
    per_hit = [groups_hit(h, question) for h in hits]
    out = {}
    for k in ks:
        top = per_hit[:k]
        covered = set().union(*top) if top else set()
        out[f"recall@{k}"] = len(covered) / n_groups
        out[f"precision@{k}"] = sum(1 for g in top if g) / k
        out[f"hit@{k}"] = float(bool(covered))
    first = next((r for r, g in enumerate(per_hit, 1) if g), None)
    out["mrr"] = 1 / first if first else 0.0
    out["first_relevant_rank"] = first
    return out


def answerable(question: dict, chunks: list[dict]) -> bool:
    """Есть ли в корпусе хоть один чанк для каждой группы (иначе вопрос пропускается)."""
    return all(any(any(spec_matches(c, s) for s in g) for c in chunks) for g in question["relevant"])


def key_fact_recall(answer: str, facts: list[str]) -> float:
    if not facts:
        return float("nan")
    a = norm(answer)
    return sum(any(norm(alt) in a for alt in f.split("|")) for f in facts) / len(facts)
