"""Запускает реплику агента или продолжает незавершённое выполнение сессии из SQLite."""

import argparse
import json
import logging
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import APIConnectionError, APIStatusError, APITimeoutError

from .agent import StudyAgent
from .deadline import run_with_timeout
from .model import load_model, load_summarizer
from .tracing import make_recorder

RESUME_NOTE = (
    "Продолжайте с --resume и тем же --session: повторится незавершённый узел, "
    "а завершённые узлы с записанным checkpoint повторяться не будут. "
    "Уже выполненные действия внутри прерванного узла могут повториться; "
    "внешние эффекты не откатываются. Если выполнение не дошло до первого "
    "checkpoint новой реплики, повторите --message."
)


def invoke_turn(graph, data, config):
    """Выполняет граф и объясняет причину остановки и сохранённую точку продолжения."""
    try:
        return graph.invoke(data, config, durability="sync")
    except APITimeoutError:
        message = (
            "Истёк таймаут запроса к OpenAI API после предусмотренных повторов SDK. "
            "Ответ основной модели не получен."
        )
    except APIConnectionError:
        message = "Не удалось подключиться к OpenAI API; ответ модели не получен."
    except APIStatusError as error:
        message = (
            f"OpenAI API вернул HTTP {error.status_code}; request_id={error.request_id}. "
            "Проверьте доступ к модели, ключ, квоту и параметры запроса."
        )
    except Exception:
        logging.exception("Ошибка выполнения графа")
        message = (
            "Выполнение графа остановлено из-за ошибки; подробности приведены в логе."
        )

    print(message)
    pending = graph.get_state(config).next
    descriptions = {
        "prepare": "подготовка контекста; суммаризация может выполниться повторно",
        "model": "вызов основной модели с сохранённым контекстом",
        "tools": "весь узел инструментов, включая ещё не сохранённые результаты",
        "stop_context": "формирование сообщения о превышении бюджета контекста",
    }
    if pending:
        print(
            "При --resume повторится: "
            + "; ".join(descriptions[node] for node in pending)
            + "."
        )
        print(
            "Завершённые узлы с записанным checkpoint не повторяются. "
            "Действие инструмента, выполненное до записи checkpoint его узла, "
            "может повториться; внешние эффекты не откатываются."
        )
    else:
        print("В checkpoint нет незавершённых узлов; --resume не требуется.")
    raise SystemExit(1)


def _run_turn(args, trace):
    logging.basicConfig(level=logging.WARNING)

    project = Path(__file__).resolve().parents[1]
    settings = json.loads((project / "config.json").read_text(encoding="utf-8"))
    paths = settings["paths"]
    database = project / paths["checkpoints"]
    database.parent.mkdir(parents=True, exist_ok=True)
    # ID сессии превращаем в безопасное имя каталога.
    output = project / paths["plans"] / uuid5(NAMESPACE_URL, args.session).hex
    config = {
        "configurable": {"thread_id": args.session},
        "recursion_limit": 4 * settings["agent"]["max_iterations"] + 5,
    }

    try:
        from .tools.study_tools import StudyTools

        model, schemas, system = load_model(project, callbacks=trace.callbacks())
        tools = StudyTools(
            project / paths["docs"],
            project / paths["vector_store"],
            output,
            trace=trace,
        )
        summarize = None
        summary_settings = {}
        if "summary" in settings["context"]["strategies"]:
            options = settings["context"]["summary"]
            summarize = load_summarizer(options["model"], callbacks=trace.callbacks())
            summary_settings = {
                "summary_prompt": (project / paths["summary_prompt"]).read_text(
                    encoding="utf-8"
                ),
                "summary_input_budget": options["input_budget_tokens"],
                "keep_recent_turns": options["keep_recent_turns"],
                "max_summary_calls": options["max_calls"],
            }

        with SqliteSaver.from_conn_string(str(database)) as saver:
            agent = StudyAgent(
                model,
                schemas,
                system,
                tools,
                max_iterations=settings["agent"]["max_iterations"],
                input_budget_tokens=settings["context"]["input_budget_tokens"],
                context_strategies=settings["context"]["strategies"],
                summarize=summarize,
                summary_settings=summary_settings,
                trace=trace,
            )
            graph = agent.build_graph(saver)
            snapshot = graph.get_state(config)
            if args.resume:
                if not snapshot.next:
                    print("В этой сессии нет незавершённого выполнения.")
                    return
                data = None
            else:
                if snapshot.next:
                    print("Сначала продолжи незавершённую реплику через --resume.")
                    return
                data = {
                    "history": [HumanMessage(content=args.message)],
                    "iterations": 0,
                    "summary_calls": 0,
                    "trace_id": args.trace_id,
                }
            result = invoke_turn(graph, data, config)
            print(result["history"][-1].content)
            print(f"Вызовов основной модели: {result['iterations']}")
            print(f"Попыток суммаризации: {result.get('summary_calls', 0)}")
            print("Контекст:", json.dumps(result["context_stats"], ensure_ascii=False))
            return result
    except Exception:
        logging.exception("Выполнение остановлено. Причина — в traceback ниже/выше.")
        print(RESUME_NOTE)
        raise SystemExit(1)


def run_turn(args):
    project = Path(__file__).resolve().parents[1]
    settings = json.loads((project / "config.json").read_text(encoding="utf-8"))
    trace = make_recorder(project, settings)
    try:
        with trace.scope("agent-attempt", trace_context=args.trace_context) as event:
            with trace.attributes(args.session, settings):
                result = _run_turn(args, trace)
                event["output"] = {"completed": result is not None}
    finally:
        trace.close()


def saved_state(database, session):
    if not database.exists():
        return {}
    with SqliteSaver.from_conn_string(str(database)) as saver:
        checkpoint = saver.get_tuple({"configurable": {"thread_id": session}})
        return checkpoint.checkpoint["channel_values"] if checkpoint else {}


def traced_run(args, project, settings):
    database = project / settings["paths"]["checkpoints"]
    previous = saved_state(database, args.session) if args.resume else {}
    args.trace_id = previous.get("trace_id") or uuid4().hex
    trace = make_recorder(project, settings)
    try:
        with trace.scope(
            "user-turn",
            trace_context={"trace_id": args.trace_id},
            input={"message": args.message, "resume": args.resume},
            metadata={
                "config": settings,
                "restored_trace": bool(previous.get("trace_id")),
            },
        ) as event:
            with trace.attributes(args.session, settings):
                args.trace_context = (
                    {
                        "trace_id": args.trace_id,
                        "parent_span_id": trace.client.get_current_observation_id(),
                    }
                    if trace.client
                    else None
                )
                try:
                    exit_code = run_with_timeout(
                        run_turn,
                        settings["agent"]["turn_timeout_seconds"],
                        args=(args,),
                    )
                except TimeoutError as error:
                    event.update(
                        level="WARNING",
                        status_message="turn_timeout",
                        output={"status": "timeout"},
                    )
                    print(error)
                    print(RESUME_NOTE)
                    return 2
                state = saved_state(database, args.session)
                history = state.get("history", [])
                current = state.get("trace_id") == args.trace_id
                event["output"] = {
                    "status": "ok" if exit_code == 0 else "error",
                    "answer": (
                        history[-1].content
                        if exit_code == 0 and current and history
                        else None
                    ),
                }
                if exit_code:
                    event.update(
                        level="ERROR", status_message=f"worker_exit_{exit_code}"
                    )
                elif current and state.get("context_error"):
                    event.update(level="WARNING", status_message="context_limit")
                return exit_code
    finally:
        trace.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--message")
    action.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    settings = json.loads((project / "config.json").read_text(encoding="utf-8"))

    raise SystemExit(traced_run(args, project, settings))


if __name__ == "__main__":
    main()
