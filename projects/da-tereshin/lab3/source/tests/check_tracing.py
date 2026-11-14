"""Проверяет дерево наблюдений, ошибки и сокращение контекста через экспорт в память без сети."""

import json
import unittest
from uuid import uuid4
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ..agent import StudyAgent
from ..context import prepare_context
from ..tracing import TraceRecorder
from ..model import load_summarizer


class TestModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class TestTools:
    def __init__(self, trace):
        self.trace = trace

    def search_theory(self, query):
        with self.trace.scope("retrieval", as_type="retriever", input={"query": query},
                              metadata={"vector_store": "synthetic", "k": 1}) as event:
            result = [{"chunk_id": "test-1", "source": "synthetic", "page": 1,
                       "squared_l2": 0.1, "text": "Тестовые данные"}]
            event["output"] = result
            return result

    def get_scene(self, scene_id):
        raise KeyError(scene_id)

    def save_study_plan(self, **kwargs):
        return {"status": "synthetic"}


def test_agent(trace, error=False, sleep=None):
    call = AIMessage(content="", tool_calls=[{
        "name": "get_scene" if error else "search_theory",
        "args": {"scene_id": "missing"} if error else {"query": "test"},
        "id": "call-test", "type": "tool_call",
    }], usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
    answer = AIMessage(content="Тест завершён", usage_metadata={
        "input_tokens": 200, "output_tokens": 20, "total_tokens": 220,
    })
    model = TestModel(responses=[call, answer], callbacks=trace.callbacks(), sleep=sleep,
                      metadata={"ls_model_name": "synthetic-model"})
    return StudyAgent(model, [], SystemMessage(content="Тест"), TestTools(trace), trace=trace)


class TracingTests(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        public = "pk-lf-test-" + uuid4().hex
        client = Langfuse(public_key=public, secret_key="test", base_url="http://127.0.0.1:9",
                          tracer_provider=self.provider, span_exporter=self.exporter)
        self.trace = TraceRecorder(client, public_key=public)

    def tearDown(self):
        self.trace.close()

    def spans(self):
        self.trace.client.flush()
        return self.exporter.get_finished_spans()

    def test_tree_generation_usage_and_retriever(self):
        with self.trace.scope("test-root"):
            result = test_agent(self.trace).build_graph(None).invoke({
                "history": [HumanMessage(content="Тест")], "iterations": 0,
            })
        spans = self.spans()
        by_name = {span.name: span for span in spans}
        for name in ("test-root", "iteration-1", "iteration-2", "prepare", "model",
                     "tools", "search_theory", "retrieval"):
            self.assertIn(name, by_name)
        generations = [span for span in spans
                       if span.attributes.get("langfuse.observation.type") == "generation"]
        self.assertEqual(len(generations), 2)
        self.assertTrue(all("langfuse.observation.usage_details" in span.attributes for span in generations))
        self.assertEqual(by_name["retrieval"].parent.span_id, by_name["search_theory"].context.span_id)
        self.assertEqual(len({span.context.trace_id for span in spans}), 1)
        self.assertEqual(result["history"][-1].content, "Тест завершён")

    def test_tool_error_is_recorded_before_conversion_to_tool_message(self):
        with self.trace.scope("test-root"), self.assertLogs("source.agent", level="ERROR"):
            result = test_agent(self.trace, error=True).build_graph(None).invoke({
                "history": [HumanMessage(content="Тест")], "iterations": 0,
            })
        tool = next(span for span in self.spans() if span.name == "get_scene")
        self.assertEqual(tool.attributes["langfuse.observation.level"], "ERROR")
        self.assertEqual(result["history"][-2].status, "error")

    def test_context_strategy_has_before_after_and_real_scope(self):
        history = [HumanMessage(content="Старый"), AIMessage(content="Факты " * 100),
                   HumanMessage(content="Текущий")]
        count = lambda messages: sum(len(message.model_dump_json()) for message in messages)
        _, stats = prepare_context(history, SystemMessage(content="Тест"), count, 600,
                                   strategies=["sliding_window"], trace=self.trace)
        span = next(span for span in self.spans() if span.name == "context.sliding_window")
        output = json.loads(span.attributes["langfuse.observation.output"])
        self.assertTrue(stats["fits"])
        self.assertEqual(output["removed_turns"], 1)
        self.assertLess(output["estimated_tokens_after"], output["estimated_tokens_before"])

    def test_summary_error_is_warning(self):
        history = [HumanMessage(content="Старый"), AIMessage(content="Факты " * 100),
                   HumanMessage(content="Текущий")]
        def fail(messages):
            raise TimeoutError("synthetic")
        count = lambda messages: sum(len(message.model_dump_json()) for message in messages)
        prepare_context(history, SystemMessage(content="Тест"), count, 600,
                        strategies=["summary", "sliding_window"], trace=self.trace,
                        summarize=fail, keep_recent_turns=0, summary_input_budget=10000)
        span = next(span for span in self.spans() if span.name == "context.summary")
        self.assertEqual(span.attributes["langfuse.observation.level"], "WARNING")

    def test_summary_generation_is_nested_under_strategy(self):
        def factory(**options):
            return TestModel(
                responses=[AIMessage(content="Краткие факты", response_metadata={"finish_reason": "stop"},
                                     usage_metadata={"input_tokens": 100, "output_tokens": 5, "total_tokens": 105})],
                callbacks=options["callbacks"], metadata={"ls_model_name": "synthetic-model"},
            )
        with patch("source.model.ChatOpenAI", side_effect=factory):
            summarize = load_summarizer({"name": "synthetic-model", "max_output_tokens": 100,
                                         "request_timeout_seconds": 5, "max_retries": 0},
                                        callbacks=self.trace.callbacks())
        history = [HumanMessage(content="Старый"), AIMessage(content="Факты " * 100),
                   HumanMessage(content="Текущий")]
        count = lambda messages: sum(len(message.model_dump_json()) for message in messages)
        _, stats = prepare_context(history, SystemMessage(content="Тест"), count, 1000,
                                   strategies=["summary"], trace=self.trace, summarize=summarize,
                                   keep_recent_turns=0, summary_input_budget=10000,
                                   summary_prompt="Составь резюме")
        spans = self.spans()
        strategy = next(span for span in spans if span.name == "context.summary")
        generation = next(span for span in spans
                          if span.attributes.get("langfuse.observation.type") == "generation")
        self.assertEqual(generation.parent.span_id, strategy.context.span_id)
        self.assertTrue(stats["fits"])


if __name__ == "__main__":
    unittest.main()
