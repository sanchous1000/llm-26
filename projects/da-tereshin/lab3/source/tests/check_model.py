"""Проверяет через OpenAI API формирование вызова get_scene для S01 без выполнения инструмента."""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage
from openai import APIConnectionError, APIStatusError, APITimeoutError

from ..model import load_model


def main():
    project = Path(__file__).resolve().parents[2]
    model, schemas, system = load_model(project)
    # Принудительный выбор проверяет интеграцию, а не умение выбрать инструмент.
    caller = model.bind_tools(
        schemas,
        strict=True,
        tool_choice="get_scene",
        parallel_tool_calls=False,
    )
    try:
        response = caller.invoke([
            system,
            HumanMessage(content="Получи карточку сцены S01."),
        ])
    except APITimeoutError:
        raise SystemExit("Истёк таймаут запроса после разрешённых повторов.")
    except APIConnectionError:
        raise SystemExit("Не удалось подключиться к OpenAI API.")
    except APIStatusError as error:
        raise SystemExit(
            f"OpenAI API: HTTP {error.status_code}; request_id={error.request_id}. "
            "Проверь доступ к модели, ключ, квоту и параметры запроса."
        )

    print("tool_calls:", json.dumps(response.tool_calls, ensure_ascii=False, indent=2))
    print("usage:", json.dumps(response.usage_metadata, ensure_ascii=False, indent=2))
    print("finish_reason:", response.response_metadata.get("finish_reason"))
    assert not response.invalid_tool_calls, response.invalid_tool_calls
    assert len(response.tool_calls) == 1, "Модель не вернула один вызов инструмента"
    call = response.tool_calls[0]
    assert call["name"] == "get_scene"
    assert call["args"] == {"scene_id": "S01"}
    print("OK: модель вернула get_scene(S01). Инструмент не выполнялся.")


if __name__ == "__main__":
    main()
