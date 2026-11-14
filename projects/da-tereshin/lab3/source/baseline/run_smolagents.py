"""Запускает готовый ToolCallingAgent на последовательности реплик одного диалога."""

import argparse
import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langfuse import Langfuse, propagate_attributes
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from smolagents import OpenAIModel, Tool, ToolCallingAgent

from ..tracing import TraceRecorder


class StudyOpenAIModel(OpenAIModel):
    def _prepare_completion_kwargs(self, *args, **kwargs):
        options = super()._prepare_completion_kwargs(*args, **kwargs)
        if not options.get("tools"):
            options.pop("parallel_tool_calls", None)
        return options


class StudyToolAdapter(Tool):
    skip_forward_signature_validation = True
    output_type = "string"

    def __init__(self, schema, function, trace):
        self.name = schema["name"]
        self.description = schema["description"]
        self.inputs = schema["parameters"]["properties"]
        self.function = function
        self.trace = trace
        super().__init__()

    def forward(self, **arguments):
        with self.trace.scope(self.name, as_type="tool", input=arguments) as event:
            result = self.function(**arguments)
            event["output"] = result
            return json.dumps(result, ensure_ascii=False)


def build_agent(project, settings, tools):
    schemas = json.loads(
        (project / settings["paths"]["tool_schemas"]).read_text(encoding="utf-8")
    )
    options = settings["model"]
    model = StudyOpenAIModel(
        model_id=options["name"],
        api_key=os.environ["OPENAI_API_KEY"],
        client_kwargs={
            "timeout": options["request_timeout_seconds"],
            "max_retries": options["max_retries"],
        },
        max_completion_tokens=options["max_output_tokens"],
        parallel_tool_calls=False,
    )
    return ToolCallingAgent(
        tools=[
            StudyToolAdapter(schema, getattr(tools, schema["name"]), tools.trace)
            for schema in schemas
        ],
        model=model,
        instructions=(project / settings["paths"]["system_prompt"]).read_text(
            encoding="utf-8"
        ),
        max_steps=settings["agent"]["max_iterations"],
        add_base_tools=False,
        verbosity_level=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--message", action="append", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    load_dotenv(project / ".env")
    settings = json.loads((project / "config.json").read_text(encoding="utf-8"))
    provider = TracerProvider()
    client = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
        tracer_provider=provider,
        timeout=5,
    )
    trace = TraceRecorder(client)
    instrumentor = SmolagentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    try:
        from ..tools.study_tools import StudyTools

        paths = settings["paths"]
        output = (
            project
            / paths["plans"]
            / "smolagents"
            / uuid5(NAMESPACE_URL, args.session).hex
        )
        tools = StudyTools(
            project / paths["docs"],
            project / paths["vector_store"],
            output,
            trace=trace,
        )
        agent = build_agent(project, settings, tools)
        for index, message in enumerate(args.message):
            with trace.scope(
                "smolagents-turn",
                input={"message": message},
                metadata={"config": settings, "agent": "ToolCallingAgent"},
            ) as event:
                with propagate_attributes(
                    session_id=args.session,
                    user_id=settings["tracing"]["user_id"],
                    tags=[
                        "lab3",
                        "ready-agent",
                        "smolagents-1.26.0",
                        settings["model"]["name"],
                    ],
                ):
                    memory_start = len(agent.memory.steps)
                    answer = agent.run(message, reset=index == 0)
                    limit_reached = any(
                        type(getattr(step, "error", None)).__name__
                        == "AgentMaxStepsError"
                        for step in agent.memory.steps[memory_start:]
                    )
                    if limit_reached:
                        event.update(level="WARNING", status_message="iteration_limit")
                    event["output"] = {"answer": str(answer)}
                    print(f"\nРеплика {index + 1}:\n{answer}")
                    print("Trace:", client.get_trace_url())
            client.flush()
    finally:
        instrumentor.uninstrument()
        trace.close()


if __name__ == "__main__":
    main()
