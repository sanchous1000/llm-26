"""Проверяет окно по токенам, порядок стратегий и сохранность текущего хода."""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..context import FINAL_NOTE, prepare_context
from . import check_agent


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.system = SystemMessage(content="Система")
        self.old = [HumanMessage(content="Старый запрос"), AIMessage(content="Ответ" * 30)]
        self.current = [
            HumanMessage(content="Текущий запрос"),
            AIMessage(content="", tool_calls=[{
                "name": "get_scene", "args": {"scene_id": "S01" * 100},
                "id": "c1", "type": "tool_call",
            }]),
            ToolMessage(content="Результат" * 20, tool_call_id="c1", name="get_scene"),
        ]
        self.history = self.old + self.current

    def count(self, messages):
        # Детерминированная мера для проверки алгоритма; TokenCounter проверен отдельно.
        return sum(len(m.model_dump_json()) for m in messages)

    def prepare(self, history, budget, strategies=("sliding_window",), last=False):
        return prepare_context(history, self.system, self.count, budget,
                               strategies=strategies, last=last)

    def test_exact_boundary_keeps_full_history(self):
        full = [self.system, *self.history]
        messages, stats = self.prepare(self.history, self.count(full))
        self.assertEqual(messages, full)
        self.assertEqual(stats["steps"], [])

    def test_window_preserves_current_tool_cycle_and_history(self):
        snapshot = [m.model_dump() for m in self.history]
        expected = [self.system, *self.current]
        messages, stats = self.prepare(self.history, self.count(expected))
        self.assertEqual(messages, expected)
        self.assertTrue(stats["fits"])
        self.assertEqual(stats["steps"][0]["removed_turns"], 1)
        self.assertEqual([m.model_dump() for m in self.history], snapshot)

    def test_window_keeps_recent_completed_turn_that_fits(self):
        recent = [HumanMessage(content="Недавний"), AIMessage(content="Ответ")]
        expected = [self.system, *recent, *self.current]
        messages, stats = self.prepare(self.old + recent + self.current, self.count(expected))
        self.assertEqual(messages, expected)
        self.assertEqual(stats["steps"][0]["removed_turns"], 1)

    def test_oversized_current_turn_and_final_note_survive(self):
        expected = [self.system, *self.current, FINAL_NOTE]
        messages, stats = self.prepare(self.history, self.count(expected) - 1, last=True)
        self.assertEqual(messages, expected)
        self.assertFalse(stats["fits"])

    def test_unfinished_old_turn_is_not_deleted(self):
        history = self.current + [HumanMessage(content="Ещё вопрос")]
        messages, stats = self.prepare(history, 1)
        self.assertEqual(messages, [self.system, *history])
        self.assertFalse(stats["fits"])

    def test_pipeline_does_not_treat_tool_result_as_new_turn(self):
        packed, _ = self.prepare(self.current, 1, ("drop_tool_calls",))
        budget = self.count(packed)
        messages, stats = self.prepare(
            self.history, budget, ("drop_tool_calls", "sliding_window"))
        self.assertEqual(messages, packed)
        self.assertTrue(stats["fits"])
        self.assertEqual([s["strategy"] for s in stats["steps"]],
                         ["drop_tool_calls", "sliding_window"])
        self.assertTrue(all(s["applied"] for s in stats["steps"]))
        self.assertIn(self.current[-1].content, messages[-1].content)

    def test_pipeline_stops_when_first_strategy_fits(self):
        packed, _ = self.prepare(self.history, 1, ("drop_tool_calls",))
        messages, stats = self.prepare(
            self.history, self.count(packed), ("drop_tool_calls", "sliding_window"))
        self.assertEqual(messages, packed)
        self.assertEqual(len(stats["steps"]), 1)

    def test_empty_strategies_and_unknown_name(self):
        messages, stats = self.prepare(self.history, 1, ())
        self.assertEqual(messages, [self.system, *self.history])
        self.assertFalse(stats["fits"])
        with self.assertRaises(ValueError):
            self.prepare(self.history, 100000, ("typo",))


class WindowAgentTests(unittest.TestCase):
    def test_graph_uses_configured_window(self):
        fixture = check_agent.AgentTests()
        agent = fixture.make_agent([AIMessage(content="Готово")],
                                context_strategies=["sliding_window"])
        current = HumanMessage(content="Текущий запрос")
        agent.input_budget_tokens = agent.count_tokens([agent.system, current])
        result = agent.build_graph(None).invoke({
            "history": [HumanMessage(content="Старый"),
                        AIMessage(content="Длинный ответ " * 200), current],
            "iterations": 0,
        })
        self.assertEqual(fixture.automatic.invoke.call_args.args[0], [agent.system, current])
        self.assertEqual(len(result["history"]), 4)
        self.assertTrue(result["context_stats"]["fits"])


if __name__ == "__main__":
    unittest.main()