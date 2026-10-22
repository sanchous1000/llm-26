"""Parsing and verification of the model's candidate questions."""

import json
import re
from dataclasses import asdict, dataclass, field

from source.evaluation.documents import heading_path, page_at
from source.evaluation.locate import locate

MIN_EVIDENCE_CHARS = 150
MAX_EVIDENCE_CHARS = 2500
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_LEAKY_RE = re.compile(
    r"\b(this paper|the paper|the authors?|this work|the above|the text|"
    r"(?:section|theorem|lemma|figure|table|algorithm|equation|eq\.|proposition|corollary)\s+\d)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Candidate:
    """One generated question and the verification of its evidence."""

    part: int
    question: str
    question_ru: str
    type: str
    answer: str
    status: str  # ok | evidence_not_found | evidence_size | invalid
    reason: str = ""
    evidence: dict = field(default_factory=dict)
    flags: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["flags"] = list(self.flags)
        return data


def parse_answer(text: str) -> list[dict]:
    """Extract the JSON array from a model answer (tolerates fences/prose)."""
    cleaned = _FENCE_RE.sub("", text.strip())
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON array in the answer")
    items = json.loads(cleaned[start : end + 1])
    if not isinstance(items, list):
        raise ValueError("the JSON value is not an array")
    return [item for item in items if isinstance(item, dict)]


def verify(items: list[dict], body: str, part: int) -> list[Candidate]:
    """Check each item and attach the located evidence passage."""
    return [_verify_one(item, body, part) for item in items]


def _verify_one(item: dict, body: str, part: int) -> Candidate:
    fields = {
        key: str(item.get(key, "")).strip()
        for key in ("question", "question_ru", "type", "answer", "evidence")
    }
    base = {"part": part, **{k: v for k, v in fields.items() if k != "evidence"}}
    if not fields["question"] or not fields["evidence"]:
        return Candidate(
            **base, status="invalid", reason="missing question or evidence"
        )
    found = locate(body, fields["evidence"])
    if found is None:
        return Candidate(
            **base, status="evidence_not_found", reason="quote not in document"
        )
    size = len(found.text)
    if not MIN_EVIDENCE_CHARS <= size <= MAX_EVIDENCE_CHARS:
        return Candidate(**base, status="evidence_size", reason=f"{size} chars")
    evidence = {
        "text": found.text,
        "start": found.start,
        "end": found.end,
        "match_score": round(found.score, 1),
        "section_path": heading_path(body, found.start),
        "page": page_at(body, found.start),
    }
    flags = ("leaky_question",) if _LEAKY_RE.search(fields["question"]) else ()
    return Candidate(**base, status="ok", evidence=evidence, flags=flags)
