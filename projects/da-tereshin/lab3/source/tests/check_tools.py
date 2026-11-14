"""Проверяет получение сцены, поиск теории и сохранение плана подготовки в файл."""

import json
from pathlib import Path
from uuid import uuid4

from ..tools.study_tools import StudyTools


project = Path(__file__).resolve().parents[2]
store = (
    project.parent / "lab2/experiments/2026-11-05_21-15-53_474914_chunking"
    / "recursive_700_100/vector_store"
)
session_id = uuid4().hex
tools = StudyTools(
    project / "docs", store, project / "artifacts/plans" / session_id
)

scene = tools.get_scene("S01")
print(json.dumps(scene, ensure_ascii=False, indent=2))

results = tools.search_theory(
    "Ролан Барт: закрепление и связывание как функции языкового сообщения"
)
for row in results:
    print(row["rank"], row["source"], row["page"], row["squared_l2"])
    print(row["text"])

saved = tools.save_study_plan("B01", [
    "Повторить различие закрепления и связывания у Барта.",
    "Разобрать S01 с надписью и без неё.",
    "Разобрать S02 и сравнить обе сцены.",
])
actual = json.loads(Path(saved["path"]).read_text(encoding="utf-8"))
assert actual == {"ticket_id": "B01", "points": saved["points"]}
assert tools.get_scene("S01") == scene
print(json.dumps(saved, ensure_ascii=False, indent=2))
