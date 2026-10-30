"""Task set of the agent evaluation: loading, validation, checks and Langfuse export."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from source import config as project

DEFAULT_TASKS_PATH = project.DEFAULT_EVAL_DIR / "agent" / "tasks.yaml"
TOOLS = frozenset(
    {
        "search_papers",
        "get_chunk_context",
        "list_documents",
        "get_document_outline",
        "calculator",
    }
)
TYPES = ("single", "dialog")
DIFFICULTIES = ("easy", "medium", "hard")

FactGroup = tuple[str, ...]  # alternatives: any one of them must appear


class DatasetError(ValueError):
    """The task file is malformed; the message names the task and the field."""


@dataclass(frozen=True)
class Turn:
    user: str
    expected_output: str
    must_include: tuple[FactGroup, ...] = ()
    depends_on_history: bool = False
    memory_probe: bool = False
    expected_tools: tuple[str, ...] = ()
    reference_docs: tuple[str, ...] = ()
    check: str | None = None  # "exact": must_include alone decides success


@dataclass(frozen=True)
class Task:
    id: str
    type: str
    difficulty: str
    category: str
    source: str
    language: str
    turns: tuple[Turn, ...]
    long_dialog: bool = False

    @property
    def is_single(self) -> bool:
        return self.type == "single"


def load_tasks(path: Path = DEFAULT_TASKS_PATH) -> tuple[Task, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
        raise DatasetError(f"{path}: expected a mapping with a 'tasks' list")
    return parse_tasks(data["tasks"])


def parse_tasks(raw: Iterable[Mapping[str, Any]]) -> tuple[Task, ...]:
    tasks = tuple(_parse_task(item) for item in raw)
    counts = Counter(task.id for task in tasks)
    duplicates = sorted(tid for tid, n in counts.items() if n > 1)
    if duplicates:
        raise DatasetError(f"duplicate task ids: {', '.join(duplicates)}")
    return tasks


def _parse_task(raw: Mapping[str, Any]) -> Task:
    tid = str(raw.get("id") or "")
    _require(tid, "?", "every task needs an id")
    _require(raw.get("type") in TYPES, tid, f"type must be one of {TYPES}")
    _require(
        raw.get("difficulty") in DIFFICULTIES,
        tid,
        f"difficulty must be one of {DIFFICULTIES}",
    )
    turns = tuple(_parse_turn(t, tid, i) for i, t in enumerate(raw.get("turns") or []))
    if raw["type"] == "single":
        _require(len(turns) == 1, tid, "a single task has exactly one turn")
    else:
        _require(len(turns) >= 2, tid, "a dialog has at least two turns")
        _require(not turns[0].depends_on_history, tid, "the first turn has no history")
    return Task(
        id=tid,
        type=raw["type"],
        difficulty=raw["difficulty"],
        category=str(raw.get("category", "")),
        source=str(raw.get("source", "own")),
        language=str(raw.get("language", "en")),
        turns=turns,
        long_dialog=bool(raw.get("long_dialog", False)),
    )


def _parse_turn(raw: Mapping[str, Any], tid: str, index: int) -> Turn:
    where = f"{tid} turn {index}"
    user = str(raw.get("user") or "").strip()
    _require(user, where, "user message is empty")
    expected = str(raw.get("expected_output") or "").strip()
    _require(expected, where, "expected_output is empty")
    tools = tuple(raw.get("expected_tools") or ())
    unknown = sorted(set(tools) - TOOLS)
    _require(not unknown, where, f"unknown tool(s): {', '.join(unknown)}")
    depends = bool(raw.get("depends_on_history", False))
    probe = bool(raw.get("memory_probe", False))
    _require(depends or not probe, where, "memory_probe needs depends_on_history")
    facts = _parse_facts(raw.get("must_include") or (), where)
    check = raw.get("check")
    _require(check in (None, "exact"), where, "check must be 'exact'")
    _require(check is None or facts, where, "check=exact needs must_include")
    return Turn(
        user=user,
        expected_output=expected,
        must_include=facts,
        depends_on_history=depends,
        memory_probe=probe,
        expected_tools=tools,
        reference_docs=tuple(raw.get("reference_docs") or ()),
        check=check,
    )


def _parse_facts(raw: Iterable[Any], where: str) -> tuple[FactGroup, ...]:
    groups = tuple(
        (item,) if isinstance(item, str) else tuple(str(alt) for alt in item)
        for item in raw
    )
    _require(
        all(group and all(group) for group in groups),
        where,
        "must_include groups must be non-empty",
    )
    return groups


def _require(condition: object, where: str, message: str) -> None:
    if not condition:
        raise DatasetError(f"{where}: {message}")


def missing_facts(answer: str, groups: Iterable[FactGroup]) -> list[FactGroup]:
    """The fact groups of which no alternative occurs in ``answer``."""
    text = answer.casefold()
    return [g for g in groups if not any(alt.casefold() in text for alt in g)]


def unknown_doc_keys(tasks: Iterable[Task], known: set[str]) -> dict[str, list[str]]:
    """Reference documents that the corpus does not contain, per task."""
    result = {}
    for task in tasks:
        refs = {doc for turn in task.turns for doc in turn.reference_docs}
        if missing := sorted(refs - known):
            result[task.id] = missing
    return result


def to_langfuse_item(task: Task) -> dict[str, Any]:
    """A Langfuse dataset item: user turns as input, per-turn expectations."""
    return {
        "id": task.id,
        "input": {"turns": [turn.user for turn in task.turns]},
        "expected_output": [
            {
                "turn": i,
                "expected_output": turn.expected_output,
                "must_include": [list(group) for group in turn.must_include],
                "check": turn.check,
            }
            for i, turn in enumerate(task.turns)
        ],
        "metadata": {
            "type": task.type,
            "difficulty": task.difficulty,
            "category": task.category,
            "source": task.source,
            "language": task.language,
            "long_dialog": task.long_dialog,
            "history_turns": [
                i for i, t in enumerate(task.turns) if t.depends_on_history
            ],
            "memory_probe_turns": [
                i for i, t in enumerate(task.turns) if t.memory_probe
            ],
            "expected_tools": [list(turn.expected_tools) for turn in task.turns],
            "reference_docs": [list(turn.reference_docs) for turn in task.turns],
        },
    }


def summarise(tasks: Iterable[Task]) -> dict[str, Any]:
    tasks = tuple(tasks)
    turns = [turn for task in tasks for turn in task.turns]
    return {
        "tasks": len(tasks),
        "turns": len(turns),
        "by_type": dict(Counter(task.type for task in tasks)),
        "by_difficulty": dict(Counter(task.difficulty for task in tasks)),
        "by_category": dict(Counter(task.category for task in tasks)),
        "by_language": dict(Counter(task.language for task in tasks)),
        "long_dialogs": sum(task.long_dialog for task in tasks),
        "history_turns": sum(turn.depends_on_history for turn in turns),
        "memory_probes": sum(turn.memory_probe for turn in turns),
        "exact_checks": sum(turn.check == "exact" for turn in turns),
    }
