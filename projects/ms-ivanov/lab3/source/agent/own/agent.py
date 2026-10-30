"""The dialogue agent: PydanticAI agent + context manager + Langfuse logging.

One ``DialogSession`` is one conversation (one Langfuse session); ``ask`` runs one turn:

    turn (agent) -> iteration N (chain) -> llm (generation)
                                        -> tool:* (tool) -> retrieval (retriever)
                 -> context-reduction -> reduce:* (span), helper (generation)

Stop conditions of a turn: a final text answer; ``max_steps`` model requests or
``max_tool_calls`` tool calls (then a tool-less answer is forced from what was
gathered); ``turn_timeout_s``; an unrecoverable error. The dialogue always goes on.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from source.agent.config import AgentConfig, Prompts, require_env
from source.agent.own.context import (
    ContextManager,
    TokenCounter,
    render_transcript,
    turn_starts,
)
from source.agent.own.mcp_client import ToolBackend
from source.agent.own.runtime import AgentRuntime
from source.agent.own.tools import AgentDeps, TurnState, build_tools
from source.agent.tracing import STATUS_LEVELS, Tracing

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # logging goes to Langfuse
logger = logging.getLogger(__name__)
TIMEOUT_ANSWER = (
    "I could not finish within the time limit. Please ask again or narrow the question."
)
FINALISE_PROMPT = """\
The research assistant ran out of {reason} while answering the user. Below is the
work done in this turn (the user's question, tool calls and results). Write the best
possible answer to the user's question from this material only, cite the papers, and
say clearly what could not be verified. Answer in the language of the question.

{transcript}"""


@dataclass(frozen=True)
class TurnResult:
    """Outcome of one user turn (also the per-turn record of the evaluation)."""

    answer: str
    status: str  # ok | step_limit | timeout | error
    error: str | None
    latency_s: float
    trace_id: str | None
    observation_id: str | None
    state: TurnState
    context_limit: int
    context_tokens_end: int

    @property
    def reduced(self) -> bool:
        return bool(self.state.reductions)

    def to_dict(self) -> dict[str, Any]:
        state = self.state
        return {
            "answer": self.answer,
            "status": self.status,
            "error": self.error,
            "latency_s": round(self.latency_s, 2),
            "trace_id": self.trace_id,
            "observation_id": self.observation_id,
            "steps": state.steps,
            "llm_calls": state.llm_calls,
            "tool_calls": state.tool_calls,
            "tool_errors": state.tool_errors,
            "tools_used": state.tools_used,
            "retrieved_docs": state.retrieved_docs,
            "input_tokens": state.input_tokens,
            "output_tokens": state.output_tokens,
            "cost_usd": state.cost_usd,
            "context_limit": self.context_limit,
            "context_tokens_max": max(state.context_tokens, default=0),
            "context_tokens_end": self.context_tokens_end,
            "reduced": self.reduced,
            "reductions": state.reductions,
        }


class AgentFactory:
    """Everything shared by the sessions of one configuration."""

    def __init__(self, config: AgentConfig, prompts: Prompts, tracing: Tracing):
        self.config = config
        self.prompts = prompts
        self.tracing = tracing
        self.tools = build_tools(prompts.tools, retries=config.limits.tool_retries)
        self.counter = TokenCounter(config.context.tokenizer)
        self.overhead_tokens = self.counter.text(prompts.system) + self.counter.text(
            json.dumps(prompts.tools)
        )

    def model(self, session_id: str) -> OpenAIChatModel:
        client = AsyncOpenAI(
            base_url=self.config.model.base_url,
            api_key=require_env(self.config.model.api_key_env),
            default_headers={"x-opencode-session": session_id},  # required by opencode
            timeout=self.config.model.request_timeout_s,
            max_retries=2,
        )
        return OpenAIChatModel(
            self.config.model.name, provider=OpenAIProvider(openai_client=client)
        )

    def session(
        self, backend: ToolBackend, session_id: str, user_id: str | None = None
    ) -> "DialogSession":
        return DialogSession(
            self, backend, session_id, user_id or self.config.langfuse.user_id
        )


class DialogSession:
    def __init__(
        self, factory: AgentFactory, backend: ToolBackend, session_id: str, user_id: str
    ):
        self.factory = factory
        self.config = factory.config
        self.backend = backend
        self.session_id = session_id
        self.user_id = user_id
        self.history: list[ModelMessage] = []
        self.turns = 0
        self._deps: AgentDeps | None = None
        model = factory.model(session_id)
        settings = {
            "temperature": self.config.model.temperature,
            "max_tokens": self.config.model.max_output_tokens,
        }
        self.context = ContextManager(
            self.config.context,
            overhead_tokens=factory.overhead_tokens,
            summarise=self._summarise,
            counter=factory.counter,
        )
        self.agent = Agent(
            model,
            instructions=factory.prompts.system,
            deps_type=AgentDeps,
            tools=factory.tools,
            capabilities=[AgentRuntime(self.config.model, context=self.context)],
            retries=self.config.limits.tool_retries,
            tool_timeout=self.config.limits.tool_timeout_s,
            model_settings=settings,
        )
        self.helper = Agent(  # summaries and forced final answers, no tools
            model,
            instructions="You are a precise assistant for a research assistant agent.",
            deps_type=AgentDeps,
            capabilities=[AgentRuntime(self.config.model, label="helper")],
            model_settings=settings,
        )

    async def ask(self, text: str) -> TurnResult:
        """One user turn, logged as one Langfuse trace of the session."""
        self.turns += 1
        deps = AgentDeps(self.backend, self.factory.tracing)
        self._deps = deps
        started = time.monotonic()
        with (
            self.factory.tracing.attributes(
                session_id=self.session_id,
                user_id=self.user_id,
                tags=self.config.tags,
                version=self.config.langfuse.version,
                trace_name=f"turn {self.turns}",
                metadata={"config": self.config.name, "turn": str(self.turns)},
            ),
            self.factory.tracing.observe(
                "turn",
                as_type="agent",
                input=text,
                metadata={
                    "turn": self.turns,
                    "context_limit": self.config.context.limit_tokens,
                    "context_strategy": self.config.context.strategy,
                    "context_tokens_start": self.context.state(self.history).tokens,
                },
            ) as root,
        ):
            trace_id = self.factory.tracing.current_trace_id()
            status, error, answer, messages = await self._run_guarded(text, deps)
            end_tokens = self.context.state(messages).tokens
            root.update(
                output=answer,
                level=STATUS_LEVELS[status],
                status_message=error,
                metadata=self._turn_metadata(status, deps.state, end_tokens),
            )
        self.history = messages
        return TurnResult(
            answer=answer,
            status=status,
            error=error,
            latency_s=time.monotonic() - started,
            trace_id=trace_id,
            observation_id=getattr(root, "id", None),
            state=deps.state,
            context_limit=self.config.context.limit_tokens,
            context_tokens_end=end_tokens,
        )

    def _turn_metadata(
        self, status: str, state: TurnState, end_tokens: int
    ) -> dict[str, Any]:
        return {
            "status": status,
            "steps": state.steps,
            "tool_calls": state.tool_calls,
            "context_tokens_max": max(state.context_tokens, default=0),
            "context_tokens_end": end_tokens,
            "context_limit": self.config.context.limit_tokens,
            "reduced": bool(state.reductions),
            "cost_usd": state.cost_usd,
        }

    async def _run_guarded(self, text: str, deps: AgentDeps):
        runs: list[Any] = []
        limits = self.config.limits
        try:
            async with asyncio.timeout(limits.turn_timeout_s):
                answer, messages = await self._iterate(text, deps, runs)
            return "ok", None, answer, messages
        except UsageLimitExceeded as error:
            status, reason = "step_limit", "steps (tool-call budget)"
            failure: BaseException = error
        except TimeoutError as error:
            status, reason = "timeout", "time"
            failure = error
        except Exception as error:  # noqa: BLE001 - the dialogue must go on
            logger.exception("turn failed")
            status, reason = "error", "luck (an error occurred)"
            failure = error
        partial = self._partial_messages(text, runs)
        if status == "timeout":
            answer = TIMEOUT_ANSWER
        else:
            answer = await self._finalise(partial, reason)
        return (
            status,
            f"{type(failure).__name__}: {failure}"[:500],
            answer,
            [*partial, ModelResponse(parts=[TextPart(content=answer)])],
        )

    async def _iterate(self, text: str, deps: AgentDeps, runs: list[Any]):
        limits = UsageLimits(
            request_limit=self.config.limits.max_steps,
            tool_calls_limit=self.config.limits.max_tool_calls,
        )
        tracing = self.factory.tracing
        async with self.agent.iter(
            text, message_history=self.history, deps=deps, usage_limits=limits
        ) as run:
            runs.append(run)
            node, iteration = run.next_node, 0
            with ExitStack() as stack:
                while not Agent.is_end_node(node):
                    if Agent.is_model_request_node(node):
                        stack.close()
                        iteration += 1
                        stack.enter_context(
                            tracing.observe(f"iteration {iteration}", as_type="chain")
                        )
                    node = await run.next(node)
            return str(run.result.output), run.result.all_messages()

    def _partial_messages(self, text: str, runs: list[Any]) -> list[ModelMessage]:
        """History up to the failure, without a dangling unanswered tool call."""
        messages: list[ModelMessage] = []
        if runs:
            try:
                messages = list(runs[0].all_messages())
            except Exception:  # noqa: BLE001
                messages = []
        # The run recorded this turn if its last user turn is this question (a
        # summary made during the turn may have shrunk the turn count: do not count).
        if not _current_turn_is(messages, text):
            return [*self.history, ModelRequest(parts=[UserPromptPart(content=text)])]
        if isinstance(messages[-1], ModelResponse):  # unanswered calls / cut-off answer
            messages = messages[:-1]
        return messages

    async def _finalise(self, messages: list[ModelMessage], reason: str) -> str:
        starts = turn_starts(messages)
        current = messages[starts[-1] :] if starts else messages
        prompt = FINALISE_PROMPT.format(
            reason=reason, transcript=render_transcript(current, tool_chars=3000)
        )
        try:
            result = await self.helper.run(prompt, deps=self._deps)
            return str(result.output)
        except Exception as error:  # noqa: BLE001
            logger.exception("forced final answer failed")
            return f"I could not complete the answer ({type(error).__name__})."

    async def _summarise(self, transcript: str) -> str:
        prompt = self.factory.prompts.summary.format(
            max_tokens=self.config.context.summary_max_tokens, transcript=transcript
        )
        result = await self.helper.run(
            prompt,
            deps=self._deps,
            # a reasoning model spends tokens on thinking before the summary itself
            model_settings={"max_tokens": self.config.model.max_output_tokens},
        )
        return str(result.output)


def _current_turn_is(messages: list[ModelMessage], text: str) -> bool:
    starts = turn_starts(messages)
    if not starts:
        return False
    return any(
        isinstance(part, UserPromptPart) and part.content == text
        for part in messages[starts[-1]].parts
    )
