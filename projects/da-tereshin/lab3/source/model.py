"""Создаёт клиент OpenAI и загружает системный промпт и схемы инструментов из конфигурации."""

import json
import logging
from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from openai import APITimeoutError

logger = logging.getLogger(__name__)


def load_model(project: Path, callbacks=None):
    config = json.loads((project / "config.json").read_text(encoding="utf-8"))
    options = config["model"]
    paths = config["paths"]
    schemas = json.loads((project / paths["tool_schemas"]).read_text(encoding="utf-8"))
    system = SystemMessage(
        content=(project / paths["system_prompt"]).read_text(encoding="utf-8")
    )
    model = ChatOpenAI(
        model=options["name"],
        max_completion_tokens=options["max_output_tokens"],
        timeout=options["request_timeout_seconds"],
        max_retries=options["max_retries"],
        use_responses_api=False,
        **({"callbacks": callbacks} if callbacks else {}),
    )
    return model, schemas, system


def load_summarizer(options, callbacks=None):
    """Создаёт отдельный клиент без инструментов и возвращает функцию суммаризации."""
    model = ChatOpenAI(
        model=options["name"],
        max_completion_tokens=options["max_output_tokens"],
        timeout=options["request_timeout_seconds"],
        max_retries=options["max_retries"],
        use_responses_api=False,
        **({"callbacks": callbacks} if callbacks else {}),
    )

    def summarize(messages):
        try:
            response = model.invoke(messages)
        except APITimeoutError as error:
            logger.warning("Истёк таймаут API суммаризатора; результат не получен.")
            raise RuntimeError(
                "Истёк таймаут API суммаризатора после предусмотренных повторов SDK"
            ) from error

        reason = response.response_metadata.get("finish_reason")
        logger.info(
            "Суммаризация: finish_reason=%s, usage=%s", reason, response.usage_metadata
        )
        if reason != "stop":
            raise RuntimeError(f"Суммаризация не завершена: finish_reason={reason}")
        if response.tool_calls or response.invalid_tool_calls:
            raise RuntimeError("Суммаризатор вернул вызов инструмента")
        if response.additional_kwargs.get("refusal"):
            raise RuntimeError("Суммаризатор отказался формировать резюме")
        text = response.content
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Суммаризатор не вернул текст резюме")
        return text.strip()

    return summarize
