"""Описывает структуру состояния графа с историей диалога, контекстом и счётчиком итераций."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    history: Annotated[list[AnyMessage], add_messages]
    # Отдельно собираемый список для следующего запроса модели.
    context: list[AnyMessage]
    summary: str  # резюме старых ходов
    summary_end: int  # число сообщений исходной history, покрытых резюме
    summary_calls: int  # попытки суммаризации в текущей реплике
    iterations: int  # для лимита вызовов модели внутри текущей реплики
    context_stats: dict
    context_error: str
    trace_id: str  # стабильный ID трейса реплики, включая resume
