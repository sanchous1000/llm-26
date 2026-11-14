"""Создаёт наблюдения Langfuse и связывает узлы графа с итерациями и сессией."""

import os
from contextlib import contextmanager, nullcontext

from dotenv import load_dotenv
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler


class TraceRecorder:
    def __init__(self, client=None, public_key=None):
        self.client = client
        self.public_key = public_key
        self.iteration = None
        self.iteration_number = None

    def callbacks(self):
        return [CallbackHandler(public_key=self.public_key)] if self.client else []

    def attributes(self, session, settings):
        if not self.client:
            return nullcontext()
        return propagate_attributes(
            session_id=session,
            user_id=settings["tracing"]["user_id"],
            trace_name="study-turn",
            tags=[
                "lab3",
                settings["tracing"]["version"],
                settings["model"]["name"],
                *settings["context"]["strategies"],
            ],
        )

    @contextmanager
    def scope(self, name, as_type="span", **fields):
        changes = {}
        if not self.client:
            yield changes
            return
        with self.client.start_as_current_observation(
            name=name, as_type=as_type, **fields
        ) as span:
            try:
                yield changes
            except BaseException as error:
                changes.update(
                    level="ERROR",
                    status_message=type(error).__name__,
                    output={"error": type(error).__name__},
                )
                raise
            finally:
                span.update(**changes)

    def end_iteration(self):
        if self.iteration:
            self.iteration.end()
            self.iteration = None

    def node(self, function, name, max_iterations):
        def run(state):
            if not self.client:
                return function(state)
            index = state["iterations"] + (0 if name == "tools" else 1)
            if self.iteration is None or self.iteration_number != index:
                self.end_iteration()
                self.iteration = self.client.start_observation(
                    name=f"iteration-{index}",
                    as_type="chain",
                    metadata={"iteration": index},
                )
                self.iteration_number = index
            parent = {
                "trace_id": self.iteration.trace_id,
                "parent_span_id": self.iteration.id,
            }
            try:
                with self.scope(
                    name,
                    as_type="chain",
                    trace_context=parent,
                    input={"iteration": index},
                ) as event:
                    result = function(state)
                    event["output"] = {
                        key: value for key, value in result.items() if key != "context"
                    }
                    if name == "stop_context" or (
                        name == "model" and index >= max_iterations
                    ):
                        event.update(
                            level="WARNING",
                            status_message=(
                                "context_limit"
                                if name == "stop_context"
                                else "iteration_limit"
                            ),
                        )
                if name in ("tools", "stop_context") or (
                    name == "model" and not result["history"][-1].tool_calls
                ):
                    self.end_iteration()
                return result
            except BaseException:
                self.end_iteration()
                raise
            finally:
                self.client.flush()

        return run

    def close(self):
        self.end_iteration()
        if self.client:
            self.client.shutdown()


def make_recorder(project, settings):
    load_dotenv(project / ".env")
    if (
        not settings.get("tracing", {}).get("enabled", False)
        or os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "false"
    ):
        return TraceRecorder()
    return TraceRecorder(
        Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            base_url=os.environ["LANGFUSE_BASE_URL"],
            timeout=5,
            sample_rate=1,
        ),
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    )
