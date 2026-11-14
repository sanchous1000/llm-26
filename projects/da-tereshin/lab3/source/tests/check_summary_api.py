"""Проверяет одним запросом к API сокращение синтетической истории и повторное использование резюме."""

import json
from pathlib import Path
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..context import TokenCounter, prepare_context
from ..model import load_summarizer


def main():
    project = Path(__file__).resolve().parents[2]
    settings = json.loads((project / "config.json").read_text(encoding="utf-8"))
    options = settings["context"]["summary"]
    summarize = Mock(wraps=load_summarizer(options["model"]))
    prompt = (project / settings["paths"]["summary_prompt"]).read_text(encoding="utf-8")
    history = [
        HumanMessage(content="Готовлю билет B01. Нужны три пункта, источники и пример S01."),
        AIMessage(content=(
            "Цель — подготовка билета B01 по кинотеории. Пользователь просил три пункта. "
            "Источник: Барт, Риторика образа, PDF 10. Закрепление ограничивает смыслы изображения. "
            "Разобрать пример сцены S01; план ещё не сохранён. Не выдумывать цитаты.\n"
        ) * 30),
        HumanMessage(content="Сохрани ограничение по источникам."),
        AIMessage(content="Используем только материалы курса; неподтверждённые цитаты не добавляем."),
        HumanMessage(content="Продолжим подготовку B01."),
    ]
    original = [m.model_dump() for m in history]
    common = dict(
        history=history, system=SystemMessage(content="Помогай готовиться к экзамену."),
        count=TokenCounter([]), budget=1000, strategies=["summary"],
        summarize=summarize, summary_prompt=prompt,
        summary_input_budget=options["input_budget_tokens"],
        keep_recent_turns=options["keep_recent_turns"],
        max_summary_calls=options["max_calls"],
    )
    messages, stats = prepare_context(**common)
    print("Контекст:", json.dumps({k: v for k, v in stats.items() if k != "summary_state"},
                                  ensure_ascii=False, indent=2))
    print("Резюме:", stats["summary_state"]["summary"])
    assert stats["fits"], "Резюме не уложило контекст в тестовый бюджет"
    assert stats["summary_state"]["summary_end"] == 2, "Резюме не обновилось"
    assert messages[-3:] == history[-3:], "Свежий или текущий ход изменён"
    assert [m.model_dump() for m in history] == original, "Полная история изменена"
    cached, again = prepare_context(**common, **stats["summary_state"])
    summarize.assert_called_once()
    assert cached == messages and again["fits"], "Резюме не использовано повторно"
    print("OK: реальная суммаризация и повторное использование резюме без второго API-вызова.")


if __name__ == "__main__":
    main()