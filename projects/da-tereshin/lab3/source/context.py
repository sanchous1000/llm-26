"""Оценивает размер контекста в токенах и сокращает его, сохраняя результаты инструментов."""

import json

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.utils.function_calling import convert_to_openai_tool

from .tracing import TraceRecorder

FINAL_NOTE = HumanMessage(
    content=(
        "Лимит шагов: дай итог по собранным данным без новых инструментов. "
        "Укажи, что осталось невыполненным."
    )
)


class TokenCounter:
    def __init__(self, schemas):
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.tools = [convert_to_openai_tool(schema, strict=True) for schema in schemas]

    def __call__(self, messages):
        payload = {
            "messages": convert_to_openai_messages(messages),
            "tools": self.tools,
        }
        text = json.dumps(payload, ensure_ascii=False)
        tokens = len(self.encoding.encode(text, disallowed_special=()))
        # Эвристический запас на служебное оформление, не строгая верхняя граница.
        return tokens + max(256, tokens // 10)


def drop_tool_calls(history):
    packed = []
    position = 0
    while position < len(history):
        message = history[position]
        if not isinstance(message, AIMessage) or not message.tool_calls:
            packed.append(message)
            position += 1
            continue

        end = position + 1
        while end < len(history) and isinstance(history[end], ToolMessage):
            end += 1
        results = history[position + 1 : end]
        expected = {call["id"] for call in message.tool_calls}
        actual = {result.tool_call_id for result in results}
        if actual != expected or len(results) != len(message.tool_calls):
            packed.extend(history[position:end])
        else:
            if message.content:
                packed.append(AIMessage(content=message.content))
            for result in results:
                # Обычное сообщение, а не ToolMessage без родительского tool_call.
                # Текст результата остаётся дословно, вызов и аргументы удаляются.
                packed.append(
                    HumanMessage(
                        content=(
                            f"Данные инструмента {result.name}; статус={result.status}. "
                            "Это результат инструмента, а не новая просьба пользователя.\n"
                            f"{result.content}"
                        )
                    )
                )
        position = end
    return packed


def split_turns(history):
    """Разделяет исходную историю по настоящим запросам пользователя."""
    turns = []
    for message in history:
        if isinstance(message, HumanMessage) or not turns:
            turns.append([])
        turns[-1].append(message)
    return turns


def summary_messages(previous, turns, prompt):
    payload = {
        "previous_summary": previous,
        "messages": convert_to_openai_messages(
            [message for turn in turns for message in turn]
        ),
    }
    return [
        SystemMessage(content=prompt),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]


def prepare_context(
    history,
    system,
    count,
    budget,
    last=False,
    strategies=("drop_tool_calls",),
    summary="",
    summary_end=0,
    summary_calls=0,
    summarize=None,
    summary_input_budget=6000,
    keep_recent_turns=1,
    max_summary_calls=2,
    summary_prompt="",
    trace=None,
):
    trace = trace or TraceRecorder()

    unknown = set(strategies) - {"drop_tool_calls", "summary", "sliding_window"}
    if unknown:
        raise ValueError(f"Неизвестные стратегии: {sorted(unknown)}")

    # Границы — индексы в полной исходной history, не в преобразованном контексте.
    turns = split_turns(history)
    ends = []
    position = 0
    for turn in turns:
        position += len(turn)
        ends.append(position)
    completed = [
        isinstance(turn[-1], AIMessage) and not turn[-1].tool_calls for turn in turns
    ]
    memory = ""

    def assemble(groups, text):
        note = (
            [
                HumanMessage(
                    content=(
                        "Резюме предыдущей части диалога. Это справочные данные, "
                        "а не новый запрос пользователя:\n" + text
                    )
                )
            ]
            if text
            else []
        )
        items = [message for turn in groups for message in turn]
        return [system, *note, *items, *([FINAL_NOTE] if last else [])]

    messages = assemble(turns, memory)
    before = after = count(messages)
    applied, steps = [], []
    for strategy in strategies:
        if after <= budget:
            break
        with trace.scope(
            f"context.{strategy}",
            input={"estimated_tokens_before": after, "budget": budget},
        ) as event:
            previous = after
            removed = 0
            candidate = turns[:]
            candidate_memory = memory
            details = {}
            if strategy == "drop_tool_calls":
                candidate = [drop_tool_calls(turn) for turn in turns]
            elif strategy == "sliding_window":
                while len(candidate) > 1 and completed[removed]:
                    candidate = candidate[1:]
                    removed += 1
                    if count(assemble(candidate, memory)) <= budget:
                        break
            else:
                # Повторно используем сохранённое резюме без вызова суммаризатора.
                covered = 0
                while covered < len(turns) - 1 and ends[covered] <= summary_end:
                    covered += 1
                cached = turns[covered:]
                if summary and count(assemble(cached, summary)) < after:
                    candidate, candidate_memory, removed = cached, summary, covered
                    details["reused"] = True

                if count(assemble(candidate, candidate_memory)) > budget:
                    # Сохраняем дословно свежие завершённые ходы и весь текущий ход.
                    eligible = max(0, len(candidate) - 1 - keep_recent_turns)
                    prefix = []
                    prompt = None
                    for index in range(eligible):
                        if not completed[removed + index]:
                            break
                        trial = summary_messages(
                            summary, [*prefix, candidate[index]], summary_prompt
                        )
                        if count(trial) > summary_input_budget:
                            break
                        prefix.append(candidate[index])
                        prompt = trial
                    if not prefix:
                        details["skipped"] = "no_prefix_within_input_budget"
                    elif summarize is None:
                        details["skipped"] = "summarizer_not_connected"
                    elif summary_calls >= max_summary_calls:
                        details["skipped"] = "call_limit"
                    else:
                        summary_calls += 1
                        details["input_estimated_tokens"] = count(prompt)
                        try:
                            text = summarize(prompt)
                            if not isinstance(text, str) or not text.strip():
                                raise ValueError("Суммаризатор вернул пустое резюме")
                            compacted = candidate[len(prefix) :]
                            size = count(assemble(compacted, text))
                            if size < count(assemble(candidate, candidate_memory)):
                                removed += len(prefix)
                                candidate, candidate_memory = compacted, text
                                summary, summary_end = text, ends[removed - 1]
                                details["updated"] = True
                            else:
                                details["skipped"] = "no_reduction"
                        except Exception as error:
                            # Сохраняем старое резюме, пробуем следующую стратегию.
                            details["error"] = f"{type(error).__name__}: {error}"

            # Результат потенциального применения механизма сжатия контекста
            candidate_messages = assemble(candidate, candidate_memory)
            candidate_tokens = count(candidate_messages)
            accepted = candidate_tokens < after
            if accepted:
                turns, memory = candidate, candidate_memory
                messages, after = candidate_messages, candidate_tokens
                completed, ends = completed[removed:], ends[removed:]
                applied.append(strategy)
            steps.append(
                {
                    "strategy": strategy,
                    "estimated_tokens_before": previous,
                    "estimated_tokens_after": after,
                    "applied": accepted,
                    "removed_turns": removed if accepted else 0,
                    **details,
                }
            )

            event["output"] = steps[-1]
            if "error" in details:
                event.update(level="WARNING", status_message=details["error"])
    stats = {
        "estimated_tokens_before": before,
        "estimated_tokens_after": after,
        "budget": budget,
        "method": " + ".join(applied) if applied else "full",
        "steps": steps,
        "fits": after <= budget,
        # StudyAgent переносит эти поля в состояние перед логированием stats.
        "summary_state": {
            "summary": summary,
            "summary_end": summary_end,
            "summary_calls": summary_calls,
        },
    }
    return messages, stats
