"""Context accounting and shortening of the dialogue history.

The history is a list of PydanticAI messages. A *turn* starts at a request that carries
a user prompt; everything up to the next such request (tool calls, tool results, the
answer) belongs to it. Shortening never splits a turn: tool calls keep their results.

Order of reductions when the prompt exceeds the limit:
1. tool results (and reasoning) of earlier turns are cut to a short preview;
2. ``window``: the oldest turns are dropped; ``summary``: they become one summary;
3. tool results inside the current turn are cut, the two latest batches last.
The current user message and the last ``keep_recent_turns`` turns are never removed.
"""

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any

import tiktoken
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from source.agent.config import ContextConfig

logger = logging.getLogger(__name__)
SUMMARY_MARK = "[Summary of the earlier conversation]"
PART_OVERHEAD_TOKENS = 4  # role / separators per message part
KEEP_CURRENT_BATCHES = 2  # latest tool-result batches of the current turn kept intact
Summariser = Callable[[str], Awaitable[str]]


class TokenCounter:
    """Estimates tokens with tiktoken (the provider's tokenizer is not public)."""

    def __init__(self, encoding: str = "o200k_base"):
        self._encoding_name = encoding

    @cached_property
    def _encoding(self) -> tiktoken.Encoding:
        return tiktoken.get_encoding(self._encoding_name)

    def text(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def messages(self, messages: Sequence[ModelMessage]) -> int:
        return sum(
            self.text(part_text(part)) + PART_OVERHEAD_TOKENS
            for message in messages
            for part in message.parts
        )


def part_text(part: Any) -> str:
    """The text a message part contributes to the prompt."""
    if isinstance(part, ToolCallPart):
        args = part.args if isinstance(part.args, str) else json.dumps(part.args)
        return f"{part.tool_name}({args})"
    if isinstance(part, ToolReturnPart | RetryPromptPart):
        content = part.content
        return content if isinstance(content, str) else json.dumps(content, default=str)
    content = getattr(part, "content", "")
    return content if isinstance(content, str) else json.dumps(content, default=str)


def is_summary(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(p, UserPromptPart) and str(p.content).startswith(SUMMARY_MARK)
        for p in message.parts
    )


def turn_starts(messages: Sequence[ModelMessage]) -> list[int]:
    """Indices of the requests that open a user turn (summaries excluded)."""
    return [
        i
        for i, message in enumerate(messages)
        if isinstance(message, ModelRequest)
        and not is_summary(message)
        and any(isinstance(p, UserPromptPart) for p in message.parts)
    ]


@dataclass(frozen=True)
class ContextState:
    tokens: int
    limit: int
    messages: int
    turns: int

    @property
    def over_limit(self) -> bool:
        return self.tokens > self.limit


@dataclass(frozen=True)
class Reduction:
    """One shortening step, as logged to Langfuse."""

    kind: (
        str  # trim_old_tool_results | drop_turns | summarise_turns | trim_current_turn
    )
    tokens_before: int
    tokens_after: int
    removed: tuple[str, ...]
    summary: str | None = None


@dataclass(frozen=True)
class ContextResult:
    messages: list[ModelMessage]
    before: ContextState
    after: ContextState
    reductions: tuple[Reduction, ...]


class ContextManager:
    """Keeps the prompt (fixed overhead + history) under ``config.limit_tokens``."""

    def __init__(
        self,
        config: ContextConfig,
        *,
        overhead_tokens: int = 0,
        summarise: Summariser | None = None,
        counter: TokenCounter | None = None,
    ):
        self.config = config
        self.overhead_tokens = overhead_tokens
        self._summarise = summarise
        self.counter = counter or TokenCounter(config.tokenizer)

    def state(self, messages: Sequence[ModelMessage]) -> ContextState:
        return ContextState(
            tokens=self.overhead_tokens + self.counter.messages(messages),
            limit=self.config.limit_tokens,
            messages=len(messages),
            turns=len(turn_starts(messages)),
        )

    async def process(self, messages: Sequence[ModelMessage]) -> ContextResult:
        before = self.state(messages)
        current = list(messages)
        reductions: list[Reduction] = []
        if before.over_limit and self.config.strategy != "none":
            for step in (self._trim_old, self._drop_or_summarise, self._trim_current):
                if not self.state(current).over_limit:
                    break
                current, reduction = await step(current)
                if reduction is not None:
                    reductions.append(reduction)
        return ContextResult(current, before, self.state(current), tuple(reductions))

    async def _trim_old(self, messages: list[ModelMessage]):
        starts = turn_starts(messages)
        if len(starts) < 2:
            return messages, None
        limit = self.config.old_tool_result_chars
        trimmed, removed = trim_tool_results(messages[: starts[-1]], limit)
        return self._result(
            messages, trimmed + messages[starts[-1] :], "trim_old_tool_results", removed
        )

    async def _drop_or_summarise(self, messages: list[ModelMessage]):
        old, recent = split_old_turns(messages, self.config.keep_recent_turns)
        if not old:
            return messages, None
        failure = None
        if self.config.strategy == "summary" and self._summarise is not None:
            try:
                return await self._summarise_old(messages, old, recent)
            except Exception as error:  # noqa: BLE001 - the turn must go on
                logger.warning("summary failed, dropping old turns instead: %s", error)
                failure = f"summary failed: {type(error).__name__}: {error}"[:500]
        kept = list(old)
        dropped: list[str] = []
        while kept and self.state(kept + recent).over_limit:
            first_turn = _first_turn(kept)
            dropped.append(turn_preview(kept[:first_turn]))
            kept = kept[first_turn:]
        new, reduction = self._result(messages, kept + recent, "drop_turns", dropped)
        if reduction is not None and failure:
            reduction = replace(reduction, summary=failure)
        return new, reduction

    async def _summarise_old(self, messages, old, recent):
        prompt = render_transcript(old)
        summary = (await self._summarise(prompt)).strip()
        summary_message = ModelRequest(
            parts=[UserPromptPart(content=f"{SUMMARY_MARK}\n{summary}")]
        )
        removed = [
            turn_preview(t) for t in split_into_turns(old) if not is_summary(t[0])
        ]
        new = [summary_message, *recent]
        reduction = Reduction(
            kind="summarise_turns",
            tokens_before=self.state(messages).tokens,
            tokens_after=self.state(new).tokens,
            removed=tuple(removed),
            summary=summary,
        )
        return new, reduction

    async def _trim_current(self, messages: list[ModelMessage]):
        starts = turn_starts(messages)
        if not starts:
            return messages, None
        start = starts[-1]
        batches = [
            i
            for i, m in enumerate(messages)
            if i > start
            and isinstance(m, ModelRequest)
            and any(isinstance(p, ToolReturnPart) for p in m.parts)
        ]
        limit = self.config.old_tool_result_chars * 4
        current, removed = messages, []
        for keep in (KEEP_CURRENT_BATCHES, 1):  # protect the latest results if possible
            if len(batches) <= keep:
                continue
            cut = batches[-keep]
            trimmed, dropped = trim_tool_results(current[start:cut], limit)
            current = current[:start] + trimmed + current[cut:]
            removed += dropped
            if not self.state(current).over_limit:
                break
        return self._result(messages, current, "trim_current_turn", removed)

    def _result(self, old, new, kind, removed):
        if not removed:
            return old, None
        reduction = Reduction(
            kind=kind,
            tokens_before=self.state(old).tokens,
            tokens_after=self.state(new).tokens,
            removed=tuple(removed),
        )
        return new, reduction


def split_into_turns(messages: Sequence[ModelMessage]) -> list[list[ModelMessage]]:
    """Consecutive groups: an optional prefix (summary) and then one group per turn."""
    starts = turn_starts(messages)
    bounds = sorted({0, *starts, len(messages)})
    return [
        list(messages[a:b]) for a, b in zip(bounds, bounds[1:], strict=False) if a < b
    ]


def split_old_turns(
    messages: Sequence[ModelMessage], keep_recent: int
) -> tuple[list[ModelMessage], list[ModelMessage]]:
    """Split into (old part, last ``keep_recent`` turns + current turn)."""
    starts = turn_starts(messages)
    keep = keep_recent + 1  # the current turn is always kept
    if len(starts) <= keep:
        return [], list(messages)
    cut = starts[-keep]
    return list(messages[:cut]), list(messages[cut:])


def _first_turn(messages: Sequence[ModelMessage]) -> int:
    """Length of the first group (prefix or turn) of ``messages``."""
    starts = [i for i in turn_starts(messages) if i > 0]
    return starts[0] if starts else len(messages)


def trim_tool_results(
    messages: Sequence[ModelMessage], max_chars: int
) -> tuple[list[ModelMessage], list[str]]:
    """Copies of the messages with long tool results cut and reasoning dropped."""
    removed: list[str] = []
    result: list[ModelMessage] = []
    for message in messages:
        parts = []
        for part in message.parts:
            if isinstance(part, ThinkingPart) and part.content:
                removed.append(f"reasoning ({len(part.content)} chars)")
                continue
            text = part_text(part) if isinstance(part, ToolReturnPart) else ""
            if len(text) > max_chars:
                removed.append(
                    f"{part.tool_name} result: {len(text)} -> {max_chars} chars"
                )
                parts.append(replace(part, content=text[:max_chars] + " [...trimmed]"))
            else:
                parts.append(part)
        result.append(replace(message, parts=parts))
    return result, removed


def turn_preview(turn: Sequence[ModelMessage], limit: int = 120) -> str:
    """The user question of a turn, shortened: what a log says was removed."""
    for message in turn:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                text = " ".join(str(part.content).split())
                return text if len(text) <= limit else text[:limit] + "..."
    return "(no user message)"


def render_transcript(messages: Sequence[ModelMessage], tool_chars: int = 600) -> str:
    """Plain-text transcript for the summariser."""
    lines: list[str] = []
    for message in messages:
        for part in message.parts:
            lines.extend(_render_part(part, tool_chars))
    return "\n".join(lines)


def _render_part(part: Any, tool_chars: int) -> list[str]:
    if isinstance(part, UserPromptPart):
        label = (
            "PREVIOUS SUMMARY" if str(part.content).startswith(SUMMARY_MARK) else "USER"
        )
        return [f"{label}: {part.content}"]
    if isinstance(part, TextPart):
        return [f"ASSISTANT: {part.content}"]
    if isinstance(part, ToolCallPart):
        return [f"TOOL CALL: {part_text(part)}"]
    if isinstance(part, ToolReturnPart):
        return [f"TOOL RESULT ({part.tool_name}): {part_text(part)[:tool_chars]}"]
    if isinstance(part, SystemPromptPart | ThinkingPart | RetryPromptPart):
        return []
    return []
