"""Оркестрация эксперимента: промпт × пресет × повтор."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from llm_lab.evaluation import checks as checks_module
from llm_lab.evaluation.text import char_count, max_repeated_ngram_ratio, word_count
from llm_lab.experiment.config import ParamPreset
from llm_lab.experiment.prompts import PromptCase
from llm_lab.inference.client import ChatResponse
from llm_lab.inference.params import EndpointConfig
from llm_lab.storage import ModelPaths, append_run, utc_now_iso, write_answer, write_manifest

DEFAULT_MAX_RETRIES = 1

# Тексты ответа и рассуждений хранятся на уровне записи (answer / reasoning);
# в metrics они только удваивали размер runs.jsonl.
DUPLICATED_TEXT_FIELDS = ("content", "reasoning")


class ChatBackend(Protocol):
    """Минимальный контракт бэкенда (реальный клиент или мок для самотеста)."""

    def chat(self, model: str, messages: list[dict[str, str]], params: Any, stream: bool) -> ChatResponse:
        ...


@dataclass(frozen=True)
class ExperimentConfig:
    """Неизменяемое описание одного запуска эксперимента."""

    model: str
    endpoint: EndpointConfig
    presets: tuple[ParamPreset, ...]
    prompts: tuple[PromptCase, ...]
    repeats: int = 3
    stream: bool = False
    safety_max_tokens: int | None = None
    include_extra_params: bool = True
    pause_s: float = 0.0
    reset: bool = False
    profile: str = "custom"

    @property
    def total_runs(self) -> int:
        return len(self.presets) * len(self.prompts) * self.repeats

    def validate(self) -> None:
        if not self.model:
            raise ValueError("Не указана модель (--model)")
        if self.repeats < 1:
            raise ValueError("--repeats должен быть >= 1")
        if not self.presets:
            raise ValueError("Не выбран ни один пресет параметров")
        if not self.prompts:
            raise ValueError("Не выбран ни один промпт")


def build_manifest(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "model": cfg.model,
        "started_at": utc_now_iso(),
        "endpoint": cfg.endpoint.base_url,
        "profile": cfg.profile,
        "repeats": cfg.repeats,
        "stream": cfg.stream,
        "safety_max_tokens": cfg.safety_max_tokens,
        "include_extra_params": cfg.include_extra_params,
        "total_runs_planned": cfg.total_runs,
        "presets": [
            {
                "key": p.key,
                "label": p.label,
                "mode": p.mode,
                "params": p.params.to_payload(include_extra=cfg.include_extra_params),
                "params_desc": p.params.describe(),
                "rationale": p.rationale,
            }
            for p in cfg.presets
        ],
        "prompts": [
            {
                "id": c.id,
                "type": c.task_type,
                "title": c.title,
                "system": c.system,
                "user": c.user,
                "reference": c.reference,
                "notes": c.notes,
                "checks": [check.name for check in c.checks],
            }
            for c in cfg.prompts
        ],
    }


def _build_record(
    cfg: ExperimentConfig,
    case: PromptCase,
    preset: ParamPreset,
    repeat: int,
    response: ChatResponse,
    check_results: tuple[checks_module.CheckResult, ...],
    attempts: int,
) -> dict[str, Any]:
    answer = response.content
    params = preset.params.with_max_tokens_floor(cfg.safety_max_tokens)
    return {
        "run_id": f"{case.id}__{preset.key}__r{repeat}",
        "timestamp": utc_now_iso(),
        "model": cfg.model,
        "endpoint": cfg.endpoint.base_url,
        "prompt_id": case.id,
        "prompt_type": case.task_type,
        "prompt_title": case.title,
        "system": case.system,
        "user": case.user,
        "reference": case.reference,
        "preset_key": preset.key,
        "preset_label": preset.label,
        "mode": preset.mode,
        "params": params.to_payload(include_extra=cfg.include_extra_params),
        "params_desc": params.describe(),
        "stream": cfg.stream,
        "repeat": repeat,
        "repeats_total": cfg.repeats,
        "attempts": attempts,
        "answer": answer,
        "reasoning": response.reasoning,
        "error": response.error,
        "metrics": {
            **{k: v for k, v in response.as_dict().items() if k not in DUPLICATED_TEXT_FIELDS},
            "words": word_count(answer),
            "chars": char_count(answer),
            "reasoning_chars": char_count(response.reasoning),
            "repeat_ngram_ratio": round(max_repeated_ngram_ratio(answer), 4),
        },
        "checks": [result.as_dict() for result in check_results],
        "checks_version": checks_module.CHECKS_VERSION,
        "quality_score": round(checks_module.aggregate_score(check_results), 4),
    }


def _call_with_retry(
    backend: ChatBackend,
    cfg: ExperimentConfig,
    case: PromptCase,
    preset: ParamPreset,
) -> tuple[ChatResponse, int]:
    """Запрос с одной повторной попыткой при ошибке транспорта."""
    params = preset.params.with_max_tokens_floor(cfg.safety_max_tokens)
    response = backend.chat(cfg.model, case.to_messages(), params, cfg.stream)
    attempts = 1
    while not response.ok and attempts <= DEFAULT_MAX_RETRIES:
        print(f"      [retry {attempts}] {response.error}")
        time.sleep(2.0)
        response = backend.chat(cfg.model, case.to_messages(), params, cfg.stream)
        attempts += 1
    return response, attempts


def run_experiment(backend: ChatBackend, cfg: ExperimentConfig, paths: ModelPaths) -> tuple[dict[str, Any], ...]:
    """Полный прогон. Результаты пишутся на диск инкрементально."""
    cfg.validate()
    paths.prepare(reset=cfg.reset)
    write_manifest(paths, build_manifest(cfg))

    records: list[dict[str, Any]] = []
    done = 0
    total = cfg.total_runs
    started = time.perf_counter()

    try:
        for case in cfg.prompts:
            print(f"\n=== {case.id} — {case.title} ===")
            for preset in cfg.presets:
                print(f"  -> {preset.key}: {preset.params.describe()}")
                for repeat in range(1, cfg.repeats + 1):
                    response, attempts = _call_with_retry(backend, cfg, case, preset)
                    check_results = (
                        checks_module.evaluate_all(case.checks, response.content)
                        if response.ok
                        else ()
                    )
                    record = _build_record(cfg, case, preset, repeat, response, check_results, attempts)
                    append_run(paths, record)
                    write_answer(paths, record)
                    records.append(record)
                    done += 1
                    _print_progress(done, total, record)
                    if cfg.pause_s > 0:
                        time.sleep(cfg.pause_s)
    except KeyboardInterrupt:
        print("\n[!] Прервано пользователем — сохранены уже выполненные прогоны.")

    elapsed = time.perf_counter() - started
    print(f"\nГотово: {done}/{total} прогонов за {elapsed:.1f} с. Данные: {paths.model_dir}")
    return tuple(records)


def _print_progress(done: int, total: int, record: dict[str, Any]) -> None:
    metrics = record["metrics"]
    if record["error"]:
        print(f"     [{done}/{total}] r{record['repeat']} ОШИБКА: {record['error'][:120]}")
        return
    print(
        f"     [{done}/{total}] r{record['repeat']} "
        f"score={record['quality_score']:.2f} "
        f"out={metrics['completion_tokens']}т "
        f"{metrics['latency_s']}с "
        f"{metrics['output_tps'] or '—'} ток/с "
        f"{'ОБРЕЗАН' if metrics['truncated'] else ''}"
    )
