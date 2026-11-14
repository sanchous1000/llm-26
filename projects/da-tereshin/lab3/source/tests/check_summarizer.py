"""Проверяет настройки клиента и обработку ответов суммаризатора без API."""

import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..context import prepare_context
from ..model import load_summarizer


class SummarizerTests(unittest.TestCase):
    def setUp(self):
        self.options = {
            "name": "test-model", "max_output_tokens": 2048,
            "request_timeout_seconds": 20, "max_retries": 1,
        }
        self.client = Mock()
        with patch("source.model.ChatOpenAI", return_value=self.client) as factory:
            self.summarize = load_summarizer(self.options)
        factory.assert_called_once_with(
            model="test-model", max_completion_tokens=2048,
            timeout=20, max_retries=1, use_responses_api=False,
        )
        self.client.bind_tools.assert_not_called()

    def test_success_passes_messages_and_returns_text(self):
        self.client.invoke.return_value = AIMessage(
            content="  Факты S01  ", response_metadata={"finish_reason": "stop"})
        messages = [SystemMessage(content="Составь резюме"), HumanMessage(content="Данные")]
        self.assertEqual(self.summarize(messages), "Факты S01")
        self.client.invoke.assert_called_once_with(messages)

    def test_rejects_incomplete_empty_refused_and_tool_responses(self):
        responses = [
            AIMessage(content="Оборвано", response_metadata={"finish_reason": "length"}),
            AIMessage(content="", response_metadata={"finish_reason": "stop"}),
            AIMessage(content="  ", response_metadata={"finish_reason": "stop"}),
            AIMessage(content="Непроверенный ответ"),
            AIMessage(content="Отказ", response_metadata={"finish_reason": "content_filter"}),
            AIMessage(content="Отказ", response_metadata={"finish_reason": "stop"},
                      additional_kwargs={"refusal": "Отказ"}),
            AIMessage(content="", response_metadata={"finish_reason": "stop"}, tool_calls=[{
                "id": "c", "name": "get_scene", "args": {}, "type": "tool_call",
            }]),
            AIMessage(content="", response_metadata={"finish_reason": "stop"}, invalid_tool_calls=[{
                "id": "c", "name": "get_scene", "args": "{", "error": "bad JSON",
                "type": "invalid_tool_call",
            }]),
            AIMessage(content=[{"type": "text", "text": "Блок вместо строки"}],
                      response_metadata={"finish_reason": "stop"}),
        ]
        for response in responses:
            with self.subTest(response=response):
                self.client.invoke.return_value = response
                with self.assertRaises(RuntimeError):
                    self.summarize([])

    def test_api_error_is_not_retried_by_adapter(self):
        self.client.invoke.side_effect = TimeoutError("timeout")
        with self.assertRaises(TimeoutError):
            self.summarize([])
        self.client.invoke.assert_called_once()

    def test_context_falls_back_after_adapter_error(self):
        self.client.invoke.return_value = AIMessage(
            content="Оборвано", response_metadata={"finish_reason": "length"})
        history = [HumanMessage(content="Старый вопрос"),
                   AIMessage(content="Старые факты " * 200),
                   HumanMessage(content="Текущий вопрос")]
        count = lambda messages: sum(len(m.model_dump_json()) for m in messages)
        system = SystemMessage(content="Система")
        messages, stats = prepare_context(
            history, system, count, count([system, history[-1]]),
            strategies=["summary", "sliding_window"], summarize=self.summarize,
            summary_prompt="Составь резюме", summary_input_budget=10000,
            keep_recent_turns=0,
        )
        self.assertTrue(stats["fits"])
        self.assertIn("length", stats["steps"][0]["error"])
        self.assertEqual(stats["summary_state"]["summary_calls"], 1)
        self.assertEqual(stats["summary_state"]["summary_end"], 0)
        self.assertEqual(messages, [system, history[-1]])


if __name__ == "__main__":
    unittest.main()