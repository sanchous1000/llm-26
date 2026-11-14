"""Проверяет цикл агента, ограничения и восстановление без обращения к API и загрузки эмбеддингов."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from ..agent import StudyAgent


class AgentTests(unittest.TestCase):
    def make_agent(self, responses, **settings):
        self.automatic = Mock()
        self.automatic.invoke.side_effect = responses
        self.final = Mock()
        self.final.invoke.return_value = AIMessage(content="Лимит достигнут.")
        model = Mock()
        model.bind_tools.side_effect = [self.automatic, self.final]
        self.tools = Mock()
        self.tools.get_scene.return_value = {"id": "S01"}
        return StudyAgent(
            model, [], SystemMessage(content="Помогай пользователю."),
            self.tools, **settings,
        )

    def tool_call(self):
        return AIMessage(content="", tool_calls=[{
            "name": "get_scene", "args": {"scene_id": "S01"},
            "id": "call-1", "type": "tool_call",
        }])

    def initial(self):
        return {"history": [HumanMessage(content="Получи S01.")], "iterations": 0}

    def test_tool_cycle(self):
        agent = self.make_agent([self.tool_call(), AIMessage(content="Готово.")])
        result = agent.build_graph(None).invoke(self.initial())
        self.tools.get_scene.assert_called_once_with(scene_id="S01")
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["history"][-2].status, "success")
        self.assertEqual(result["history"][-1].content, "Готово.")

    def test_tool_error_reaches_model(self):
        agent = self.make_agent([self.tool_call(), AIMessage(content="Уточни ID.")])
        self.tools.get_scene.side_effect = KeyError("S01")
        with self.assertLogs("source.agent", level="ERROR"):
            result = agent.build_graph(None).invoke(self.initial())
        self.assertEqual(result["history"][-2].status, "error")
        self.assertIn("KeyError", self.automatic.invoke.call_args.args[0][-1].content)
        self.assertEqual(result["history"][-1].content, "Уточни ID.")

    def test_last_iteration_disables_tools(self):
        agent = self.make_agent([self.tool_call()], max_iterations=2)
        result = agent.build_graph(None).invoke(self.initial())
        self.assertEqual(self.automatic.invoke.call_count, 1)
        self.assertEqual(self.final.invoke.call_count, 1)
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["history"][-1].content, "Лимит достигнут.")

    def test_over_budget_skips_model(self):
        agent = self.make_agent([], input_budget_tokens=1)
        with self.assertLogs("source.agent", level="WARNING"):
            result = agent.build_graph(None).invoke(self.initial())
        self.automatic.invoke.assert_not_called()
        self.final.invoke.assert_not_called()
        self.assertFalse(result["context_stats"]["fits"])
        self.assertEqual(result["iterations"], 0)

    def test_resume_with_new_agent_and_database_connection(self):
        config = {"configurable": {"thread_id": "test-resume"}}
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "checkpoints.sqlite")
            agent = self.make_agent([self.tool_call(), RuntimeError("API unavailable")])
            with SqliteSaver.from_conn_string(database) as saver:
                graph = agent.build_graph(saver)
                with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                    graph.invoke(self.initial(), config, durability="sync")
                self.assertEqual(graph.get_state(config).next, ("model",))
            agent = self.make_agent([AIMessage(content="Продолжено.")])
            with SqliteSaver.from_conn_string(database) as saver:
                graph = agent.build_graph(saver)
                result = graph.invoke(None, config, durability="sync")
                self.assertFalse(graph.get_state(config).next)
            self.tools.get_scene.assert_not_called()
            self.assertEqual(result["history"][0].content, "Получи S01.")
            self.assertEqual(result["history"][-2].status, "success")
            self.assertEqual(result["history"][-1].content, "Продолжено.")
            self.assertEqual(result["iterations"], 2)


if __name__ == "__main__":
    unittest.main()
