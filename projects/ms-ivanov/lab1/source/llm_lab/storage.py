"""Сохранение результатов прогонов на диск.

Раскладка (на каждую модель — своя директория):

    results/<model_slug>/
        manifest.json     — параметры запуска эксперимента
        runs.jsonl        — все прогоны построчно (полный промпт + параметры + ответ)
        answers/          — те же ответы в читаемом виде, по файлу на прогон
        report.md         — отчёт по модели, генерируется после прогона
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_lab.paths import RESULTS_DIR

RUNS_FILENAME = "runs.jsonl"
MANIFEST_FILENAME = "manifest.json"
REPORT_FILENAME = "report.md"
ANSWERS_DIRNAME = "answers"
COMPARISON_FILENAME = "comparison.md"


def model_slug(model_id: str) -> str:
    """Безопасное для файловой системы имя модели."""
    slug = re.sub(r"[^\w.\-]+", "_", (model_id or "unknown").strip())
    return slug.strip("_") or "unknown"


def default_results_root() -> Path:
    """<lab1>/results."""
    return RESULTS_DIR


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ModelPaths:
    """Пути результатов одной модели."""

    root: Path
    model_id: str

    @property
    def slug(self) -> str:
        return model_slug(self.model_id)

    @property
    def model_dir(self) -> Path:
        return self.root / self.slug

    @property
    def runs_file(self) -> Path:
        return self.model_dir / RUNS_FILENAME

    @property
    def manifest_file(self) -> Path:
        return self.model_dir / MANIFEST_FILENAME

    @property
    def report_file(self) -> Path:
        return self.model_dir / REPORT_FILENAME

    @property
    def answers_dir(self) -> Path:
        return self.model_dir / ANSWERS_DIRNAME

    def prepare(self, reset: bool) -> None:
        """Создать директории; при reset — очистить прошлые результаты."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.answers_dir.mkdir(parents=True, exist_ok=True)
        if reset:
            if self.runs_file.exists():
                self.runs_file.unlink()
            for stale in self.answers_dir.glob("*.md"):
                stale.unlink()


def append_run(paths: ModelPaths, record: dict[str, Any]) -> None:
    """Дописать один прогон в JSONL (полная запись, без потерь)."""
    with paths.runs_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_runs(runs_file: Path, records: tuple[dict[str, Any], ...]) -> None:
    """Атомарно перезаписать JSONL: сначала во временный файл, затем replace.

    Прерывание посреди записи не оставит полуфайл на месте результатов.
    """
    if not records:
        raise ValueError("Отказ записывать пустой набор прогонов")
    runs_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = runs_file.with_suffix(runs_file.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(runs_file)


def rewrite_answers(paths: "ModelPaths", records: tuple[dict[str, Any], ...]) -> int:
    """Пересобрать answers/ целиком из записей прогонов."""
    paths.answers_dir.mkdir(parents=True, exist_ok=True)
    for stale in paths.answers_dir.glob("*.md"):
        stale.unlink()
    for record in records:
        write_answer(paths, record)
    return len(records)


def load_runs(runs_file: Path) -> tuple[dict[str, Any], ...]:
    """Прочитать JSONL с прогонами; битые строки пропускаются с предупреждением."""
    if not runs_file.exists():
        raise FileNotFoundError(f"Файл с прогонами не найден: {runs_file}")
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(runs_file.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            print(f"[warn] {runs_file.name}:{line_no} — строка пропущена ({exc.msg})")
    if not records:
        raise ValueError(f"В файле {runs_file} нет валидных записей")
    return tuple(records)


def write_manifest(paths: ModelPaths, manifest: dict[str, Any]) -> None:
    paths.manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def answer_filename(prompt_id: str, preset_key: str, repeat: int) -> str:
    return f"{prompt_id}__{preset_key}__r{repeat}.md"


def write_answer(paths: ModelPaths, record: dict[str, Any]) -> Path:
    """Человекочитаемая карточка прогона: промпт + параметры + ответ + метрики."""
    path = paths.answers_dir / answer_filename(
        record["prompt_id"], record["preset_key"], record["repeat"]
    )
    metrics = record.get("metrics", {})
    checks = record.get("checks", [])

    lines = [
        f"# {record['run_id']}",
        "",
        "| Поле | Значение |",
        "| --- | --- |",
        f"| Модель | `{record['model']}` |",
        f"| Промпт | `{record['prompt_id']}` — {record['prompt_title']} ({record['prompt_type']}) |",
        f"| Пресет | `{record['preset_key']}` — {record['preset_label']} ({record['mode']}) |",
        f"| Параметры | `{record['params_desc']}` |",
        f"| Повтор | {record['repeat']} из {record.get('repeats_total', '?')} |",
        f"| Время | {record['timestamp']} |",
        f"| Задержка, с | {metrics.get('latency_s')} |",
        f"| TTFT (первый токен), с | {_or_dash(metrics.get('ttft_s'))} |",
        f"| До начала ответа, с | {_or_dash(metrics.get('ttfa_s'))} |",
        f"| Токены (вход/выход) | {metrics.get('prompt_tokens')} / {metrics.get('completion_tokens')} |",
        f"| из них рассуждения / ответ (оценка) | "
        f"{_or_dash(metrics.get('reasoning_tokens_est'))} / {_or_dash(metrics.get('answer_tokens_est'))} |",
        f"| Скорость, ток/с | {_or_dash(metrics.get('output_tps'))} ({metrics.get('tps_method', '—')}) |",
        f"| finish_reason | {metrics.get('finish_reason')} |",
        f"| Балл качества | {record.get('quality_score')} |",
        "",
        "## Промпт",
        "",
        "### system",
        "",
        "```text",
        record.get("system") or "(нет)",
        "```",
        "",
        "### user",
        "",
        "```text",
        record.get("user", ""),
        "```",
        "",
        "## Ответ модели",
        "",
        "```text",
        record.get("answer") or "(пусто)",
        "```",
        "",
    ]

    if record.get("reasoning"):
        lines += [
            "<details><summary>Блок рассуждений (reasoning)</summary>",
            "",
            "```text",
            record["reasoning"],
            "```",
            "",
            "</details>",
            "",
        ]

    if record.get("error"):
        lines += ["## Ошибка", "", f"`{record['error']}`", ""]

    if checks:
        lines += ["## Автопроверки", "", "| Проверка | Балл | Статус | Детали |", "| --- | --- | --- | --- |"]
        for check in checks:
            status = "✅" if check["passed"] else "❌"
            detail = str(check["detail"]).replace("|", "\\|")
            lines.append(f"| {check['name']} | {check['score']:.2f} | {status} | {detail} |")
        lines.append("")

    write_text(path, "\n".join(lines))
    return path


def _or_dash(value: Any) -> str:
    return "—" if value is None else str(value)


def iter_model_dirs(root: Path) -> Iterator[Path]:
    """Директории моделей с непустым runs.jsonl."""
    if not root.exists():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / RUNS_FILENAME).exists():
            yield child


def resolve_model_paths(root: Path, model_id: str | None) -> tuple[ModelPaths, ...]:
    """Пути одной модели или (model_id=None) всех моделей, у которых есть runs.jsonl."""
    if model_id is not None:
        return (ModelPaths(root=root, model_id=model_id),)
    return tuple(
        ModelPaths(root=root, model_id=load_runs(model_dir / RUNS_FILENAME)[0]["model"])
        for model_dir in iter_model_dirs(root)
    )
