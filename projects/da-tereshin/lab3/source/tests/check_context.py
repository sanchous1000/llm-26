"""Проверяет соблюдение бюджета контекста и сохранность истории и результатов при сокращении."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..context import TokenCounter, prepare_context, drop_tool_calls


def main():
    system = SystemMessage(content="Следуй задаче пользователя.")
    history = [
        HumanMessage(content="Найди теорию."),
        AIMessage(content="", tool_calls=[{
            "name": "search_theory",
            "args": {"query": "длинный поисковый запрос " * 200},
            "id": "call-1", "type": "tool_call",
        }]),
        ToolMessage(content="Барт, PDF 10: закрепление ограничивает смыслы.",
                    tool_call_id="call-1", name="search_theory"),
    ]
    original = [message.model_dump() for message in history]
    count = TokenCounter([])
    full_size = count([system, *history])
    packed = drop_tool_calls(history)
    packed_size = count([system, *packed])
    assert packed_size < full_size
    assert history[-1].content in packed[-1].content
    assert not any(isinstance(message, ToolMessage) for message in packed)
    assert not any(getattr(message, "tool_calls", []) for message in packed)
    assert drop_tool_calls(history[:-1]) == history[:-1]

    full, stats = prepare_context(history, system, count, full_size)
    assert stats["method"] == "full" and stats["fits"]
    assert full[1:] == history
    _, stats = prepare_context(history, system, count, packed_size)
    assert stats["method"] == "drop_tool_calls" and stats["fits"]
    _, stats = prepare_context(history, system, count, packed_size - 1)
    assert not stats["fits"]
    assert [message.model_dump() for message in history] == original
    print("OK: границы бюджета, сохранность истории и результатов проверены.")


if __name__ == "__main__":
    main()
