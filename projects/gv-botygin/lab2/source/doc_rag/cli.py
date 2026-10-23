"""Общие аргументы командной строки для скриптов: переопределение параметров нарезки."""
import argparse

from doc_rag.config import Config


def add_splitter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunk-size", type=int, help="размер чанка в токенах")
    parser.add_argument("--chunk-overlap", type=int, help="перекрытие в токенах")
    parser.add_argument(
        "--headers-in-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="добавлять путь заголовков в текст чанка",
    )


def apply_splitter_overrides(config: Config, args: argparse.Namespace) -> Config:
    data = config.model_dump()
    if args.chunk_size is not None:
        data["splitter"]["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        data["splitter"]["chunk_overlap"] = args.chunk_overlap
    if args.headers_in_text is not None:
        data["splitter"]["include_headers_in_text"] = args.headers_in_text
    return Config.model_validate(data)  # заново валидируем (overlap < chunk_size и т.д.)
