"""Проверяет резюме, бюджет суммаризации и восстановление состояния без API."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from ..context import prepare_context
from . import check_agent


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.system = SystemMessage(content="Помогай")
        self.old = [HumanMessage(content="Старый запрос"), AIMessage(content="Факты " * 500)]
        self.recent = [HumanMessage(content="Свежий запрос"), AIMessage(content="Ответ " * 80)]
        self.current = [HumanMessage(content="Текущий запрос")]
        self.history = self.old + self.recent + self.current
        self.summarizer = Mock(return_value="Краткие факты со ссылкой S01")

    def count(self, messages):
        return sum(len(m.model_dump_json()) for m in messages)

    def prepare(self, history=None, budget=2500, **settings):
        options = dict(strategies=("summary",), summarize=self.summarizer,
                       summary_input_budget=10000, summary_prompt="Составь резюме.")
        options.update(settings)
        return prepare_context(self.history if history is None else history,
                               self.system, self.count, budget, **options)

    def test_summary_keeps_recent_and_current_and_does_not_mutate_history(self):
        original = [m.model_dump() for m in self.history]
        messages, stats = self.prepare()
        self.assertTrue(stats["fits"])
        self.assertEqual(messages[-3:], self.recent + self.current)
        self.assertEqual(stats["summary_state"]["summary_end"], 2)
        self.assertEqual(stats["summary_state"]["summary_calls"], 1)
        self.assertEqual([m.model_dump() for m in self.history], original)
        self.assertEqual(self.summarizer.call_args.args[0][0].content, "Составь резюме.")
        payload = json.loads(self.summarizer.call_args.args[0][-1].content)
        self.assertEqual(len(payload["messages"]), 2)

    def test_under_budget_uses_full_history_even_with_cached_summary(self):
        messages, stats = self.prepare(budget=100000, summary="Старое", summary_end=2)
        self.assertEqual(messages, [self.system, *self.history])
        self.summarizer.assert_not_called()
        self.assertEqual(stats["summary_state"]["summary"], "Старое")

    def test_cached_summary_needs_no_new_call(self):
        _, first = self.prepare()
        messages, stats = self.prepare(**first["summary_state"])
        self.assertEqual(self.summarizer.call_count, 1)
        self.assertTrue(stats["steps"][0]["reused"])
        self.assertTrue(stats["fits"])
        self.assertEqual(messages[-3:], self.recent + self.current)

    def test_update_passes_previous_summary_and_only_new_turns(self):
        _, first = self.prepare()
        history = self.history + [AIMessage(content="Дополнение " * 500),
                                  HumanMessage(content="Ещё вопрос")]
        self.summarizer.return_value = "Обновлённые факты"
        _, stats = self.prepare(history=history, **first["summary_state"])
        self.assertEqual(self.summarizer.call_args.args[0][0].content, "Составь резюме.")
        payload = json.loads(self.summarizer.call_args.args[0][-1].content)
        self.assertEqual(payload["previous_summary"], "Краткие факты со ссылкой S01")
        self.assertEqual(payload["messages"][0]["content"], "Свежий запрос")
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(stats["summary_state"]["summary_end"], 4)

    def test_failure_preserves_summary_and_falls_back_to_window(self):
        self.summarizer.side_effect = RuntimeError("offline")
        messages, stats = self.prepare(
            budget=1000, summary="Старое", summary_end=0,
            strategies=("summary", "sliding_window"))
        self.assertTrue(stats["fits"])
        self.assertEqual(stats["summary_state"]["summary"], "Старое")
        self.assertEqual(stats["summary_state"]["summary_end"], 0)
        self.assertIn("offline", stats["steps"][0]["error"])
        self.assertEqual(messages[-1], self.current[0])

    def test_empty_or_long_response_does_not_replace_memory(self):
        for response in ("", "Слишком длинно " * 1000):
            with self.subTest(response=response[:20]):
                self.summarizer.return_value = response
                _, stats = self.prepare()
                self.assertEqual(stats["summary_state"]["summary"], "")
                self.assertEqual(stats["summary_state"]["summary_end"], 0)
                self.assertFalse(stats["steps"][0]["applied"])

    def test_input_budget_and_call_limit(self):
        for options in ({"summary_input_budget": 1}, {"summary_calls": 2}):
            with self.subTest(options=options):
                _, stats = self.prepare(**options)
                self.assertFalse(stats["steps"][0]["applied"])
        self.summarizer.assert_not_called()

    def test_current_turn_and_unfinished_old_turn_are_never_summarized(self):
        _, stats = self.prepare(history=self.current, budget=1)
        self.assertFalse(stats["fits"])
        broken = [HumanMessage(content="Вопрос"), AIMessage(content="", tool_calls=[{
            "name": "get_scene", "args": {"scene_id": "S01"}, "id": "c",
            "type": "tool_call",
        }]), ToolMessage(content="Данные", tool_call_id="c")]
        self.prepare(history=broken + self.recent + self.current, budget=1)
        self.summarizer.assert_not_called()

    def test_pipeline_uses_original_boundaries_after_drop_tool_calls(self):
        old = [HumanMessage(content="Найди"), AIMessage(content="", tool_calls=[{
            "name": "search_theory", "args": {"query": "Поиск " * 200},
            "id": "c", "type": "tool_call",
        }]), ToolMessage(content="Источник " * 500, tool_call_id="c", name="search_theory"),
            AIMessage(content="Вывод")]
        _, stats = self.prepare(history=old + self.recent + self.current,
                                strategies=("drop_tool_calls", "summary", "sliding_window"))
        self.assertEqual(stats["summary_state"]["summary_end"], 4)
        self.assertEqual(self.summarizer.call_args.args[0][0].content, "Составь резюме.")
        payload = json.loads(self.summarizer.call_args.args[0][-1].content)
        self.assertTrue(any("Источник" in m["content"] for m in payload["messages"]))
        self.assertTrue(stats["fits"])


class SummaryCheckpointTests(unittest.TestCase):
    def test_resume_uses_saved_summary_without_second_summary_call(self):
        fixture = check_agent.AgentTests()
        summarizer = Mock(return_value="Факты S01")
        history = [HumanMessage(content="Старый запрос"),
                   AIMessage(content="Длинный ответ " * 2000),
                   HumanMessage(content="Текущий запрос")]
        config = {"configurable": {"thread_id": "summary-resume"}}
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "state.sqlite")
            agent = fixture.make_agent([RuntimeError("API offline")],
                context_strategies=["summary"], summarize=summarizer,
                input_budget_tokens=1000,
                summary_settings={
                    "keep_recent_turns": 0, "summary_input_budget": 20000,
                    "summary_prompt": "Составь резюме.",
                })
            with SqliteSaver.from_conn_string(database) as saver:
                graph = agent.build_graph(saver)
                with self.assertRaisesRegex(RuntimeError, "API offline"):
                    graph.invoke({"history": history, "iterations": 0}, config, durability="sync")
                state = graph.get_state(config)
                self.assertEqual(state.next, ("model",))
                self.assertEqual(state.values["summary_end"], 2)
                self.assertEqual(state.values["summary_calls"], 1)
            agent = fixture.make_agent([AIMessage(content="Готово")],
                context_strategies=["summary"], summarize=summarizer)
            with SqliteSaver.from_conn_string(database) as saver:
                result = agent.build_graph(saver).invoke(None, config, durability="sync")
            summarizer.assert_called_once()
            self.assertEqual(result["summary"], "Факты S01")
            self.assertEqual(result["summary_end"], 2)
            self.assertEqual(result["iterations"], 1)
            self.assertEqual(len(result["history"]), 4)


if __name__ == "__main__":
    unittest.main()