"""Build the final question set from a hand-made selection of candidates."""

import json
import os
from pathlib import Path

SELECTION_HELP = (
    '{"<key>": [0, 3, {"i": 2, "question": "edited text"}]} - indices into the '
    "candidates list of eval/candidates/<key>.json"
)


def export(selection: dict, candidates_dir: Path, output: Path) -> list[dict]:
    """Write ``questions.json`` with the selected, verified questions."""
    questions: list[dict] = []
    for key in sorted(selection):
        data = json.loads((candidates_dir / f"{key}.json").read_text(encoding="utf-8"))
        for position, choice in enumerate(selection[key], start=1):
            questions.append(_question(data, choice, f"{key}-q{position}"))
    payload = {
        "count": len(questions),
        "documents": len(selection),
        "questions": questions,
    }
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    return questions


def _question(data: dict, choice: int | dict, question_id: str) -> dict:
    index, override = (choice, {}) if isinstance(choice, int) else (choice["i"], choice)
    candidate = data["candidates"][index]
    if candidate["status"] != "ok":
        raise ValueError(f"{data['key']}[{index}] is not a verified candidate")
    evidence = candidate["evidence"]
    return {
        "id": question_id,
        "doc_key": data["key"],
        "doc_title": data["title"],
        "type": candidate["type"],
        "question": override.get("question", candidate["question"]),
        "question_ru": override.get("question_ru", candidate["question_ru"]),
        "reference_answer": override.get("answer", candidate["answer"]),
        "evidence": {
            "text": evidence["text"],
            "start": evidence["start"],
            "end": evidence["end"],
            "section_path": evidence["section_path"],
            "page": evidence.get("page"),
        },
        "doc_body_sha256": data["body_sha256"],
    }
