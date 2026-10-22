"""Команда run: прогон тестов одной модели за запуск.

Примеры:
    python source/main.py run --list-models
    python source/main.py run --model qwen3.5-35b-a3b
    python source/main.py run --model google/gemma-4-12b-qat --profile full --repeats 3
    python source/main.py run --model <id> --report-only
    python source/main.py run --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from llm_lab.experiment.config import (
    ALL_PRESETS,
    DEFAULT_BUDGET_MAX_TOKENS,
    DEFAULT_PROFILE,
    DEFAULT_REPEATS,
    PROFILES,
    resolve_presets,
    with_budget,
)
from llm_lab.experiment.prompts import ALL_PROMPTS, resolve_prompts
from llm_lab.experiment.runner import ExperimentConfig, run_experiment
from llm_lab.inference.client import LLMRequestError, LMStudioClient
from llm_lab.inference.params import ConfigError, EndpointConfig
from llm_lab.reporting.model_report import generate_report
from llm_lab.storage import ModelPaths, default_results_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Прогон тестов LLM через OpenAI-совместимый REST (LM Studio и др.)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", help="ID модели на сервере инференса")
    parser.add_argument("--base-url", help="Базовый URL API (по умолчанию http://localhost:1234/v1)")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES),
                        help="набор пресетов гиперпараметров")
    parser.add_argument("--presets", nargs="*", default=[],
                        help=f"явный список пресетов вместо профиля: {', '.join(p.key for p in ALL_PRESETS)}")
    parser.add_argument("--prompts", nargs="*", default=[],
                        help=f"подмножество промптов: {', '.join(c.id for c in ALL_PROMPTS)}")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="повторов на каждую конфигурацию (для оценки стабильности)")
    parser.add_argument("--stream", action="store_true",
                        help="использовать SSE-стриминг и измерять TTFT")
    parser.add_argument("--budget-max-tokens", type=int, default=None,
                        help=f"лимит длины в пресете G_budget — единственном, где он есть "
                             f"(по умолчанию {DEFAULT_BUDGET_MAX_TOKENS})")
    parser.add_argument("--safety-max-tokens", type=int, default=None,
                        help="страховочный лимит для ВСЕХ остальных пресетов; по умолчанию выключен, "
                             "включайте только если модель уходит в бесконечную генерацию")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="не отправлять system prompt: reasoning-модели с собственным шаблоном "
                             "(напр. Ministral reasoning) включают рассуждения только со штатным system prompt")
    parser.add_argument("--no-extra-params", action="store_true",
                        help="не отправлять top_k/repeat_penalty/min_p (для серверов, их не принимающих)")
    parser.add_argument("--pause", type=float, default=0.0, help="пауза между запросами, с")
    parser.add_argument("--results", type=Path, default=default_results_root(), help="директория результатов")
    parser.add_argument("--append", action="store_true",
                        help="дописать к прошлым результатам вместо перезаписи")
    parser.add_argument("--force", action="store_true",
                        help="не проверять наличие модели в списке сервера")
    parser.add_argument("--no-report", action="store_true", help="не строить отчёт после прогона")
    parser.add_argument("--report-only", action="store_true",
                        help="только пересобрать отчёт из сохранённого runs.jsonl")
    parser.add_argument("--list-models", action="store_true", help="показать модели на сервере и выйти")
    parser.add_argument("--self-test", action="store_true",
                        help="прогнать пайплайн на мок-бэкенде без реальной модели")
    return parser


def cmd_list_models(client: LMStudioClient) -> int:
    try:
        models = client.list_models()
    except LLMRequestError as exc:
        print(f"[error] {exc}")
        print(f"        Проверьте, что сервер запущен: {client.base_url}")
        return 1
    if not models:
        print("Сервер не вернул ни одной модели.")
        return 1
    print(f"Модели на {client.base_url}:")
    for model in models:
        state = f"  [{model.state}]" if model.state != "unknown" else ""
        kind = f"  ({model.kind})" if model.kind != "unknown" else ""
        print(f"  - {model.id}{kind}{state}")
    print("\nЗагружайте модели по одной; --model принимает ID из этого списка.")
    return 0


def preflight(client: LMStudioClient, model: str, force: bool) -> bool:
    """Проверка, что запрошенная модель известна серверу."""
    try:
        models = client.list_models()
    except LLMRequestError as exc:
        print(f"[error] Сервер недоступен: {exc}")
        return force
    known = {m.id for m in models}
    if model not in known:
        print(f"[error] Модель {model!r} не найдена на сервере.")
        print("        Доступны: " + (", ".join(sorted(known)) or "(пусто)"))
        if not force:
            print("        Повторите с --force, если ID корректен и модель будет загружена по требованию.")
            return False
        return True
    info = next(m for m in models if m.id == model)
    if info.state == "not-loaded":
        print(f"[info] Модель {model} ещё не загружена в память — LM Studio загрузит её при первом запросе"
              " (первый прогон будет дольше остальных).")
    return True


def build_config(args: argparse.Namespace, endpoint: EndpointConfig) -> ExperimentConfig:
    presets = with_budget(resolve_presets(args.profile, tuple(args.presets)), args.budget_max_tokens)
    prompts = resolve_prompts(tuple(args.prompts))
    if args.no_system_prompt:
        prompts = tuple(replace(case, system=None) for case in prompts)
    return ExperimentConfig(
        model=args.model,
        endpoint=endpoint,
        presets=presets,
        prompts=prompts,
        repeats=args.repeats,
        stream=args.stream,
        safety_max_tokens=args.safety_max_tokens,
        include_extra_params=not args.no_extra_params,
        pause_s=args.pause,
        reset=not args.append,
        profile="custom" if args.presets else args.profile,
    )


def print_plan(cfg: ExperimentConfig, paths: ModelPaths) -> None:
    print("=" * 72)
    print(f"Модель:     {cfg.model}")
    print(f"Эндпоинт:   {cfg.endpoint.base_url}")
    print(f"Промпты:    {', '.join(c.id for c in cfg.prompts)}")
    print(f"Пресеты:    {', '.join(p.key for p in cfg.presets)}  (профиль: {cfg.profile})")
    print(f"Повторов:   {cfg.repeats}  →  всего запросов: {cfg.total_runs}")
    print(f"Результаты: {paths.model_dir}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        endpoint = EndpointConfig.from_env().with_base_url(args.base_url)
    except ConfigError as exc:
        print(f"[error] {exc}")
        return 2

    client = LMStudioClient(endpoint, include_extra_params=not args.no_extra_params)

    if args.list_models:
        return cmd_list_models(client)

    if args.self_test:
        return run_self_test(args, endpoint)

    if not args.model:
        print("[error] Укажите --model (список моделей: --list-models)")
        return 2

    paths = ModelPaths(root=args.results, model_id=args.model)

    if args.report_only:
        return finish_report(paths)

    try:
        cfg = build_config(args, endpoint)
        cfg.validate()
    except (ConfigError, ValueError) as exc:
        print(f"[error] {exc}")
        return 2

    if not preflight(client, cfg.model, args.force):
        return 1

    print_plan(cfg, paths)
    run_experiment(client, cfg, paths)

    if args.no_report:
        return 0
    return finish_report(paths)


def finish_report(paths: ModelPaths) -> int:
    try:
        report_path = generate_report(paths)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] Отчёт не построен: {exc}")
        return 1
    print(f"Отчёт по модели: {report_path}")
    print("Сводка по всем моделям: python source/main.py compare")
    return 0


def run_self_test(args: argparse.Namespace, endpoint: EndpointConfig) -> int:
    """Проверка всей цепочки на мок-ответах (без сервера инференса)."""
    from llm_lab.inference.mock import MockBackend

    results_root = args.results / "_selftest"
    model_id = "mock/self-test"
    paths = ModelPaths(root=results_root, model_id=model_id)
    cfg = ExperimentConfig(
        model=model_id,
        endpoint=endpoint,
        presets=with_budget(resolve_presets("full", ()), args.budget_max_tokens),
        prompts=resolve_prompts(()),
        repeats=max(2, args.repeats),
        stream=False,
        reset=True,
        profile="full (self-test)",
    )
    print_plan(cfg, paths)
    run_experiment(MockBackend(), cfg, paths)
    status = finish_report(paths)
    print(f"\n[self-test] Это синтетические данные для проверки кода, не результаты модели: {results_root}")
    return status
