"""Собирает граф агента с вызовами модели, выполнением инструментов и контролем бюджета контекста."""

import json
import logging

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from .context import TokenCounter, prepare_context
from .state import AgentState
from .tracing import TraceRecorder

logger = logging.getLogger(__name__)


class StudyAgent:
    def __init__(
        self,
        model,
        schemas,
        system,
        tools,
        max_iterations=5,
        input_budget_tokens=12000,
        context_strategies=("drop_tool_calls",),
        summarize=None,
        summary_settings=None,
        trace=None,
    ):
        self.trace = trace or TraceRecorder()
        self.context_strategies = tuple(context_strategies)
        self.system = system
        self.tools = tools
        self.max_iterations = max_iterations
        self.input_budget_tokens = input_budget_tokens
        self.count_tokens = TokenCounter(schemas)
        self.automatic = model.bind_tools(
            schemas, strict=True, tool_choice="auto", parallel_tool_calls=False
        )
        self.final = model.bind_tools(
            schemas, strict=True, tool_choice="none", parallel_tool_calls=False
        )
        self.functions = {
            "search_theory": tools.search_theory,
            "get_scene": tools.get_scene,
            "save_study_plan": tools.save_study_plan,
        }
        self.summarize = summarize
        self.summary_settings = summary_settings or {}

    def prepare_context(self, state: AgentState):
        messages, stats = prepare_context(
            state["history"],
            self.system,
            self.count_tokens,
            self.input_budget_tokens,
            last=state["iterations"] >= self.max_iterations - 1,
            strategies=self.context_strategies,
            summary=state.get("summary", ""),
            summary_end=state.get("summary_end", 0),
            summary_calls=state.get("summary_calls", 0),
            summarize=self.summarize,
            trace=self.trace,
            **self.summary_settings,
        )
        summary_state = stats.pop("summary_state")

        logger.info("Контекст: %s", stats)
        error = (
            ""
            if stats["fits"]
            else (
                "Не удалось продолжить: контекст превышает заданный бюджет токенов "
                "даже после применения настроенных стратегий сокращения. "
                "Запрос и история сохранены; на этом шаге основная модель не вызывалась. "
                "Попробуйте сократить запрос, начать новую сессию или увеличить "
                "context.input_budget_tokens в config.json."
            )
        )
        return {
            "context": messages,
            "context_stats": stats,
            "context_error": error,
            **summary_state,
        }

    def stop_context(self, state: AgentState):
        logger.warning("%s", state["context_error"])
        return {"history": [AIMessage(content=state["context_error"])]}

    def call_model(self, state: AgentState):
        last = state["iterations"] >= self.max_iterations - 1
        response = (self.final if last else self.automatic).invoke(state["context"])
        if response.invalid_tool_calls:
            raise RuntimeError("Модель вернула некорректные tool_calls")
        if response.response_metadata.get("finish_reason") == "length":
            raise RuntimeError("Ответ модели оборван лимитом выходных токенов")
        if last and response.tool_calls:
            raise RuntimeError("Модель нарушила запрет tool calling на последнем шаге")
        return {"history": [response], "iterations": state["iterations"] + 1}

    def execute_tools(self, state: AgentState):
        messages = []
        for call in state["history"][-1].tool_calls:
            try:
                with self.trace.scope(
                    call["name"],
                    as_type="tool",
                    input=call["args"],
                    metadata={"tool_call_id": call["id"]},
                ) as event:
                    name, arguments = call["name"], call["args"]
                    if name == "save_study_plan":
                        # Только разрешённые аргументы; operation_id модель не задаёт.
                        result = self.tools.save_study_plan(
                            ticket_id=arguments["ticket_id"],
                            points=arguments["points"],
                            operation_id=call["id"],
                        )
                    else:
                        result = self.functions[name](**arguments)
                    content = json.dumps(result, ensure_ascii=False)
                    status = "success"
                    event["output"] = result
            except Exception as error:
                # Граница выполнения инструмента: ошибку возвращаем модели.
                logger.exception("Ошибка инструмента %s", call["name"])
                content = json.dumps(
                    {"error": type(error).__name__, "message": str(error)},
                    ensure_ascii=False,
                )
                status = "error"
            messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=call["id"],
                    name=call["name"],
                    status=status,
                )
            )
        return {"history": messages}

    def route(self, state: AgentState):
        return "tools" if state["history"][-1].tool_calls else END

    def build_graph(self, checkpointer):
        builder = StateGraph(AgentState)
        builder.add_node(
            "prepare",
            self.trace.node(self.prepare_context, "prepare", self.max_iterations),
        )
        builder.add_node(
            "model", self.trace.node(self.call_model, "model", self.max_iterations)
        )
        builder.add_node(
            "tools", self.trace.node(self.execute_tools, "tools", self.max_iterations)
        )
        builder.add_node(
            "stop_context",
            self.trace.node(self.stop_context, "stop_context", self.max_iterations),
        )

        builder.add_edge(START, "prepare")
        builder.add_conditional_edges(
            "prepare",
            lambda state: "stop_context" if state["context_error"] else "model",
            {"stop_context": "stop_context", "model": "model"},
        )
        builder.add_edge("stop_context", END)
        builder.add_conditional_edges("model", self.route, {"tools": "tools", END: END})
        builder.add_edge("tools", "prepare")
        return builder.compile(checkpointer=checkpointer)
