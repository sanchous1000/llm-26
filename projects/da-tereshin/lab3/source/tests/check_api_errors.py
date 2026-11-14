"""Проверяет сообщения ошибок API, точку продолжения и fallback суммаризации без сети."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import APIConnectionError, APIStatusError, APITimeoutError

from ..context import prepare_context
from ..model import load_summarizer
from ..run_agent import invoke_turn
from . import check_agent


def request():
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


class APIErrorTests(unittest.TestCase):
    def test_timeout_reports_model_checkpoint_and_resume_does_not_repeat_tool(self):
        fixture = check_agent.AgentTests()
        agent = fixture.make_agent([fixture.tool_call(), APITimeoutError(request=request())])
        config = {"configurable": {"thread_id": "api-timeout"}}
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "state.sqlite")
            with SqliteSaver.from_conn_string(database) as saver:
                graph = agent.build_graph(saver)
                with patch("builtins.print") as output:
                    with self.assertRaises(SystemExit) as error:
                        invoke_turn(graph, fixture.initial(), config)
                self.assertEqual(error.exception.code, 1)
                text = "\n".join(call.args[0] for call in output.call_args_list)
                self.assertIn("Истёк таймаут", text)
                self.assertIn("вызов основной модели с сохранённым контекстом", text)
                self.assertNotIn("Traceback", text)
                self.assertEqual(graph.get_state(config).next, ("model",))
                self.assertEqual(graph.get_state(config).values["iterations"], 1)
                fixture.tools.get_scene.assert_called_once()
            agent = fixture.make_agent([AIMessage(content="Продолжено")])
            with SqliteSaver.from_conn_string(database) as saver:
                result = invoke_turn(agent.build_graph(saver), None, config)
            fixture.tools.get_scene.assert_not_called()
            self.assertEqual(result["history"][-1].content, "Продолжено")
            self.assertEqual(result["iterations"], 2)

    def test_connection_and_http_errors_have_clear_messages(self):
        response = httpx.Response(429, request=request(), headers={"x-request-id": "test-id"})
        cases = [
            (APIConnectionError(request=request()), "Не удалось подключиться"),
            (APIStatusError("rate limit", response=response, body=None), "HTTP 429"),
        ]
        for exception, expected in cases:
            with self.subTest(exception=exception):
                graph = Mock()
                graph.invoke.side_effect = exception
                graph.get_state.return_value.next = ("model",)
                with patch("builtins.print") as output:
                    with self.assertRaises(SystemExit):
                        invoke_turn(graph, None, {})
                self.assertIn(expected, output.call_args_list[0].args[0])

    def test_pending_node_descriptions_and_missing_pending_node(self):
        for pending, expected in [
            (("prepare",), "суммаризация может выполниться повторно"),
            (("tools",), "весь узел инструментов"),
            (("stop_context",), "сообщения о превышении бюджета"),
            ((), "--resume не требуется"),
        ]:
            with self.subTest(pending=pending):
                graph = Mock()
                graph.invoke.side_effect = APIConnectionError(request=request())
                graph.get_state.return_value.next = pending
                with patch("builtins.print") as output:
                    with self.assertRaises(SystemExit):
                        invoke_turn(graph, None, {})
                text = "\n".join(call.args[0] for call in output.call_args_list)
                self.assertIn(expected, text)

    def test_unexpected_exception_is_logged(self):
        graph = Mock()
        graph.invoke.side_effect = ValueError("test error")
        graph.get_state.return_value.next = ("model",)
        with self.assertLogs(level="ERROR"), patch("builtins.print") as output:
            with self.assertRaises(SystemExit):
                invoke_turn(graph, None, {})
        self.assertTrue(any("из-за ошибки" in str(call.args[0]) for call in output.call_args_list))

    def test_summary_api_timeout_preserves_memory_and_uses_window(self):
        client = Mock()
        client.invoke.side_effect = APITimeoutError(request=request())
        with patch("source.model.ChatOpenAI", return_value=client):
            summarize = load_summarizer({
                "name": "test-model", "max_output_tokens": 2048,
                "request_timeout_seconds": 20, "max_retries": 1,
            })
        history = [HumanMessage(content="Старый запрос"),
                   AIMessage(content="Старые факты " * 200),
                   HumanMessage(content="Текущий запрос")]
        count = lambda messages: sum(len(message.model_dump_json()) for message in messages)
        with self.assertLogs("source.model", level="WARNING"):
            messages, stats = prepare_context(
                history, SystemMessage(content="Помогай"), count, 1000,
                strategies=["summary", "sliding_window"], summarize=summarize,
                summary="Прежнее резюме", summary_end=0,
                summary_prompt="Составь резюме", keep_recent_turns=0,
                summary_input_budget=10000,
            )
        self.assertTrue(stats["fits"])
        self.assertEqual(messages[-1], history[-1])
        self.assertEqual(stats["summary_state"]["summary"], "Прежнее резюме")
        self.assertEqual(stats["summary_state"]["summary_end"], 0)
        self.assertIn("таймаут API суммаризатора", stats["steps"][0]["error"])
        self.assertEqual(stats["summary_state"]["summary_calls"], 1)


if __name__ == "__main__":
    unittest.main()
