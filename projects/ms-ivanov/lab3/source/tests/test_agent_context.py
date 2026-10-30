import asyncio

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from source.agent.config import ContextConfig
from source.agent.own.context import (
    SUMMARY_MARK,
    ContextManager,
    TokenCounter,
    is_summary,
    render_transcript,
    split_old_turns,
    turn_starts,
)


class CharCounter(TokenCounter):
    """Deterministic: four characters per token."""

    def text(self, text: str) -> int:
        return len(text) // 4


def turn(question: str, result_chars: int = 0, call_id: str = "c") -> list:
    """A user turn with one tool call (if result_chars) and an answer."""
    messages = [ModelRequest(parts=[UserPromptPart(content=question)])]
    if result_chars:
        messages += [
            ModelResponse(
                parts=[
                    ThinkingPart(content="thinking " * 10),
                    ToolCallPart(
                        "search_papers", {"query": question}, tool_call_id=call_id
                    ),
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "search_papers", "x" * result_chars, tool_call_id=call_id
                    )
                ]
            ),
        ]
    messages.append(ModelResponse(parts=[TextPart(content=f"answer to {question}")]))
    return messages


def manager(strategy="window", limit=2000, keep=1, summarise=None) -> ContextManager:
    config = ContextConfig(
        strategy=strategy,
        limit_tokens=limit,
        keep_recent_turns=keep,
        old_tool_result_chars=100,
    )
    return ContextManager(config, summarise=summarise, counter=CharCounter())


def run(coro):
    return asyncio.run(coro)


def tool_pairs_are_intact(messages) -> bool:
    calls = {
        p.tool_call_id for m in messages for p in m.parts if isinstance(p, ToolCallPart)
    }
    returns = {
        p.tool_call_id
        for m in messages
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    }
    return calls == returns


def test_history_under_the_limit_is_left_alone():
    history = turn("q1", 400) + turn("q2")

    result = run(manager(limit=5000).process(history))

    assert result.messages == history
    assert result.reductions == ()
    assert result.before.tokens == result.after.tokens
    assert result.before.turns == 2


def test_strategy_none_only_measures():
    history = turn("q1", 40_000) + turn("q2")

    result = run(manager(strategy="none").process(history))

    assert result.messages == history
    assert result.after.over_limit


def test_old_tool_results_are_trimmed_first_and_the_current_turn_is_kept():
    history = turn("q1", 9000, "a") + turn("q2", 9000, "b")

    result = run(manager(limit=2600).process(history))

    assert [r.kind for r in result.reductions] == ["trim_old_tool_results"]
    old_return = result.messages[2].parts[0]
    assert len(old_return.content) < 200
    current_return = result.messages[-2].parts[0]
    assert len(current_return.content) == 9000
    assert not result.after.over_limit
    assert history[2].parts[0].content == "x" * 9000  # input not mutated


def test_window_drops_the_oldest_turns_but_keeps_recent_and_current():
    history = turn("q1") + turn("q2") + turn("q3") + turn("q4")
    padded = [
        (
            ModelRequest(parts=[UserPromptPart(content=f"q{i} " + "y" * 3000)])
            if isinstance(m, ModelRequest)
            else m
        )
        for i, m in enumerate(history)
    ]

    result = run(manager(limit=1700, keep=1).process(padded))

    assert result.reductions[-1].kind == "drop_turns"
    assert len(turn_starts(result.messages)) == 2  # one recent + the current turn
    assert result.messages[-1] == padded[-1]
    assert result.reductions[-1].removed  # previews of the dropped questions


def test_summary_replaces_old_turns_with_one_message():
    seen = {}

    async def summarise(transcript: str) -> str:
        seen["transcript"] = transcript
        return "USER FACTS: d = 16"

    history = (
        turn("my dimension is d = 16", 6000, "a")
        + turn("q2", 6000, "b")
        + turn("q3", 6000, "c")
    )
    result = run(
        manager("summary", limit=1600, keep=1, summarise=summarise).process(history)
    )

    kinds = [r.kind for r in result.reductions]
    assert "summarise_turns" in kinds
    assert is_summary(result.messages[0])
    assert result.messages[0].parts[0].content.startswith(SUMMARY_MARK)
    assert "my dimension is d = 16" in seen["transcript"]
    assert len(turn_starts(result.messages)) == 2
    assert tool_pairs_are_intact(result.messages)
    summary_step = next(r for r in result.reductions if r.kind == "summarise_turns")
    assert summary_step.tokens_after < summary_step.tokens_before
    assert summary_step.summary == "USER FACTS: d = 16"


def test_a_previous_summary_is_merged_into_the_next_one():
    calls = []

    async def summarise(transcript: str) -> str:
        calls.append(transcript)
        return f"summary {len(calls)}"

    ctx = manager("summary", limit=1600, keep=0, summarise=summarise)
    history = turn("q1", 7000, "a") + turn("q2", 7000, "b")
    first = run(ctx.process(history)).messages
    second = run(ctx.process(first + turn("q3", 7000, "c"))).messages

    assert is_summary(second[0])
    assert "PREVIOUS SUMMARY" in calls[-1]
    assert len([m for m in second if is_summary(m)]) == 1


def test_a_single_huge_turn_trims_its_own_older_tool_results():
    current = [
        ModelRequest(parts=[UserPromptPart(content="q")]),
        ModelResponse(parts=[ToolCallPart("search_papers", {}, tool_call_id="1")]),
        ModelRequest(
            parts=[ToolReturnPart("search_papers", "z" * 20000, tool_call_id="1")]
        ),
        ModelResponse(parts=[ToolCallPart("search_papers", {}, tool_call_id="2")]),
        ModelRequest(
            parts=[ToolReturnPart("search_papers", "w" * 2000, tool_call_id="2")]
        ),
    ]

    result = run(manager(limit=2000).process(current))

    assert [r.kind for r in result.reductions] == ["trim_current_turn"]
    assert len(result.messages[2].parts[0].content) < 500
    assert result.messages[4].parts[0].content == "w" * 2000  # the latest result stays
    assert tool_pairs_are_intact(result.messages)


def test_the_two_latest_tool_batches_of_the_current_turn_are_kept_when_possible():
    current = [ModelRequest(parts=[UserPromptPart(content="q")])]
    for call_id, size in (("1", 20000), ("2", 3000), ("3", 3000)):
        current += [
            ModelResponse(
                parts=[ToolCallPart("search_papers", {}, tool_call_id=call_id)]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart("search_papers", "z" * size, tool_call_id=call_id)
                ]
            ),
        ]

    result = run(manager(limit=2200).process(current))

    sizes = [len(m.parts[0].content) for m in result.messages[2::2]]
    assert sizes[0] < 2000  # the oldest batch is cut
    assert sizes[1:] == [3000, 3000]  # the two latest stay intact
    assert tool_pairs_are_intact(result.messages)


def test_split_old_turns_keeps_the_requested_number_of_turns():
    history = turn("q1") + turn("q2") + turn("q3")

    old, recent = split_old_turns(history, keep_recent=1)

    assert len(turn_starts(old)) == 1
    assert len(turn_starts(recent)) == 2
    assert split_old_turns(history, keep_recent=5) == ([], history)


def test_transcript_shows_roles_tools_and_hides_reasoning():
    text = render_transcript(turn("what is DTM?", 50, "a"))

    assert "USER: what is DTM?" in text
    assert "TOOL CALL: search_papers(" in text
    assert "TOOL RESULT (search_papers)" in text
    assert "ASSISTANT: answer to what is DTM?" in text
    assert "thinking" not in text


def test_real_tokenizer_counts_something():
    counter = TokenCounter("o200k_base")

    assert 0 < counter.text("Sinkhorn convergence rate") < 10


def test_a_failed_summary_falls_back_to_dropping_turns():
    async def broken(_transcript: str) -> str:
        raise RuntimeError("token limit exceeded before any response")

    history = turn("q1", 6000, "a") + turn("q2", 6000, "b") + turn("q3", 6000, "c")

    result = run(
        manager("summary", limit=1600, keep=1, summarise=broken).process(history)
    )

    kinds = [r.kind for r in result.reductions]
    assert "summarise_turns" not in kinds
    assert "drop_turns" in kinds
    dropped = next(r for r in result.reductions if r.kind == "drop_turns")
    assert dropped.summary.startswith("summary failed: RuntimeError")
    assert len(turn_starts(result.messages)) == 2
    assert tool_pairs_are_intact(result.messages)
