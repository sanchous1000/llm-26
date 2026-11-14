"""Проверяет сохранение состояния графа в SQLite и продолжение после учебного сбоя в отдельном запуске."""

import argparse
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from ..state import AgentState


def build_graph(checkpointer, fail: bool):
    def prepare(state: AgentState):
        return {"context": list(state["history"]), "iterations": 0}

    def respond(state: AgentState):
        if fail:
            raise RuntimeError("Учебный сбой после сохранения prepare")
        return {
            "history": [AIMessage(content="Состояние восстановлено.")],
            "iterations": state["iterations"] + 1,
        }

    builder = StateGraph(AgentState)
    builder.add_node("prepare", prepare)
    builder.add_node("respond", respond)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["start", "resume"])
    parser.add_argument("--thread-id", required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    settings = json.loads((project / "config.json").read_text(encoding="utf-8"))
    database = project / settings["paths"]["checkpoints"]
    database.parent.mkdir(parents=True, exist_ok=True)
    config = {"configurable": {"thread_id": args.thread_id}}

    with SqliteSaver.from_conn_string(str(database)) as saver:
        graph = build_graph(saver, fail=args.stage == "start")

        if args.stage == "start":
            assert not graph.get_state(config).values, "Для start возьми новый thread-id"
            initial = {
                "history": [HumanMessage(content="Отвечай не более чем в 180 словах.")],
                "summary": "",
            }
            try:
                graph.invoke(initial, config, durability="sync")
            except RuntimeError as error:
                print(error)
            snapshot = graph.get_state(config)
            assert snapshot.next == ("respond",)
            assert snapshot.values["context"][0].content == initial["history"][0].content
            print("Checkpoint сохранён. Заверши этот запуск и выполни resume.")
        else:
            snapshot = graph.get_state(config)
            assert snapshot.next == ("respond",), "Сначала выполни start с тем же thread-id"
            original = snapshot.values["history"][0].content
            result = graph.invoke(None, config, durability="sync")
            assert result["history"][0].content == original
            assert len(result["history"]) == 2
            assert result["iterations"] == 1
            assert not graph.get_state(config).next
            for message in result["history"]:
                print(message.type, message.content)
            print("OK: история восстановлена, граф завершён.")


if __name__ == "__main__":
    main()
