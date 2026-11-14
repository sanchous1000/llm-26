"""Проверяет остановку процесса и восстановление графа после общего таймаута без API."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from ..deadline import run_with_timeout
from . import check_agent


def complete(path):
    Path(path).write_text("done")


def fail():
    raise SystemExit(7)


def stalled_turn(database, marker, stage):
    fixture = check_agent.AgentTests()

    def stall(*args, **kwargs):
        Path(marker).write_text(json.dumps({"pid": os.getpid(), "stage": stage}))
        time.sleep(30)
        raise AssertionError("Зависший вызов не был остановлен")

    if stage == "summary":
        agent = fixture.make_agent(
            [AIMessage(content="Готово")], input_budget_tokens=1000,
            context_strategies=["summary"], summarize=stall,
            summary_settings={
                "keep_recent_turns": 0, "summary_input_budget": 10000,
                "summary_prompt": "Составь резюме",
            },
        )
        data = {
            "history": [HumanMessage(content="Старый запрос"),
                        AIMessage(content="Много фактов " * 1000),
                        HumanMessage(content="Продолжай")],
            "iterations": 0, "summary_calls": 0,
        }
    else:
        agent = fixture.make_agent([])
        fixture.automatic.invoke.side_effect = [fixture.tool_call(), AIMessage(content="Готово")]
        if stage == "model":
            def respond(messages):
                if len(messages) == 2:
                    return fixture.tool_call()
                return stall()
            fixture.automatic.invoke.side_effect = respond
        else:
            fixture.tools.get_scene.side_effect = stall
        data = fixture.initial()
    config = {"configurable": {"thread_id": "deadline"}}
    with SqliteSaver.from_conn_string(database) as saver:
        agent.build_graph(saver).invoke(data, config, durability="sync")


class DeadlineTests(unittest.TestCase):
    def test_success_and_child_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "result")
            self.assertEqual(run_with_timeout(complete, 5, args=(path,)), 0)
            self.assertEqual(Path(path).read_text(), "done")
        self.assertEqual(run_with_timeout(fail, 5), 7)

    def check_resume(self, stage):
        config = {"configurable": {"thread_id": "deadline"}}
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "state.sqlite")
            marker = Path(directory) / "started.json"
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                run_with_timeout(stalled_turn, 2, args=(database, str(marker), stage))
            self.assertLess(time.monotonic() - started, 5)
            self.assertTrue(marker.exists(), "Дочерний процесс не дошёл до проверяемого вызова")
            pid = json.loads(marker.read_text())["pid"]
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

            fixture = check_agent.AgentTests()
            summary = Mock(return_value="Краткие факты")
            agent = fixture.make_agent(
                [AIMessage(content="Продолжено")], summarize=summary,
                context_strategies=["summary"], input_budget_tokens=1000,
                summary_settings={
                    "keep_recent_turns": 0, "summary_input_budget": 10000,
                    "summary_prompt": "Составь резюме",
                },
            )
            with SqliteSaver.from_conn_string(database) as saver:
                graph = agent.build_graph(saver)
                snapshot = graph.get_state(config)
                expected = {"model": "model", "tools": "tools", "summary": "prepare"}[stage]
                self.assertEqual(snapshot.next, (expected,))
                result = graph.invoke(None, config, durability="sync")
                self.assertFalse(graph.get_state(config).next)
            self.assertEqual(result["history"][-1].content, "Продолжено")
            if stage == "model":
                fixture.tools.get_scene.assert_not_called()
                self.assertEqual(result["iterations"], 2)
            elif stage == "tools":
                fixture.tools.get_scene.assert_called_once_with(scene_id="S01")
                self.assertEqual(result["iterations"], 2)
            else:
                summary.assert_called_once()
                self.assertEqual(result["summary_end"], 2)
                self.assertEqual(result["summary_calls"], 1)

    def test_model_timeout_and_resume_without_repeated_tool(self):
        self.check_resume("model")

    def test_tool_timeout_and_resume_from_tool(self):
        self.check_resume("tools")

    def test_summary_timeout_and_resume_from_prepare(self):
        self.check_resume("summary")

    def test_cli_reports_timeout_and_passes_session_to_worker(self):
        from .. import run_agent

        with patch.dict(os.environ, {"LANGFUSE_TRACING_ENABLED": "false"}), \
             patch("sys.argv", ["run_agent", "--session", "test", "--resume"]), \
             patch.object(run_agent, "run_with_timeout", side_effect=TimeoutError("Истёк таймаут")) as run, \
             patch("builtins.print") as output:
            with self.assertRaises(SystemExit) as error:
                run_agent.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIs(run.call_args.args[0], run_agent.run_turn)
        self.assertTrue(run.call_args.kwargs["args"][0].resume)
        self.assertEqual(run.call_args.kwargs["args"][0].session, "test")
        self.assertIn("--resume", output.call_args_list[-1].args[0])


if __name__ == "__main__":
    unittest.main()
