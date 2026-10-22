"""Compact text view of candidates (for choosing the best ones)."""

import json
from pathlib import Path

_SNIPPET = 260
_COMPACT_ANSWER = 170  # answer chars shown when evidence is hidden (snippet == 0)


def render(path: Path, snippet: int = _SNIPPET) -> str:
    """Human-readable listing of one document's verified candidates."""
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"=== {data['key']} | {data['title']} | {data['body_chars']} chars"]
    for index, candidate in enumerate(data["candidates"]):
        if candidate["status"] != "ok":
            short = candidate["question"][:80]
            lines.append(f"  [{index}] DROPPED {candidate['status']}: {short}")
            continue
        evidence = candidate["evidence"]
        flags = f" FLAGS={','.join(candidate['flags'])}" if candidate["flags"] else ""
        place = " > ".join(evidence["section_path"][1:]) or "-"
        page = f" p.{evidence['page']}" if evidence.get("page") else ""
        text = " ".join(evidence["text"].split())
        size = len(evidence["text"])
        answer = candidate["answer"]
        lines += [
            f"  [{index}] ({candidate['type']}){flags} @ {place}{page} [{size} ch]",
            f"      Q: {candidate['question']}",
            f"      A: {answer if snippet else answer[:_COMPACT_ANSWER]}",
        ]
        if snippet:
            lines.append(
                f"      E: {text[:snippet]}{'...' if len(text) > snippet else ''}"
            )
    return "\n".join(lines)
