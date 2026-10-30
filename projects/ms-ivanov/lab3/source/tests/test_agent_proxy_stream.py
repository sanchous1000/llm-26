import json

from source.agent.opencode.proxy_stream import (
    StreamAccumulator,
    completion_message,
    new_tool_results,
    retrieved_chunks,
    usage_from,
)


def sse(*chunks) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def test_stream_with_text_tool_calls_and_usage_is_rebuilt_across_split_bytes():
    raw = sse(
        {"model": "m", "choices": [{"delta": {"reasoning_content": "think "}}]},
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {
                                    "name": "papers-rag_search",
                                    "arguments": '{"query":',
                                },
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": ' "dtm"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    )
    stream = StreamAccumulator()
    for i in range(0, len(raw), 7):  # arbitrary network chunking
        stream.feed(raw[i : i + 7])
    stream.close()

    message = stream.message()
    assert message["content"] == "Hello"
    assert message["reasoning"] == "think "
    assert message["tool_calls"] == [
        {"id": "c1", "name": "papers-rag_search", "arguments": '{"query": "dtm"}'}
    ]
    assert stream.finish_reason == "tool_calls"
    assert stream.model == "m"
    assert usage_from(stream.usage) == {
        "input": 10, "output": 3, "cache_read_input_tokens": 0, "total": 13,
    }  # fmt: skip


def test_garbage_lines_are_ignored():
    stream = StreamAccumulator()
    stream.feed(b": keep-alive\n\ndata: {not json}\n\ndata: [DONE]\n")
    stream.close()

    assert stream.message() == {"role": "assistant", "content": ""}


def test_non_streamed_completion_message():
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "x", "function": {"name": "t", "arguments": "{}"}}
                    ],
                }
            }
        ]
    }

    assert completion_message(body)["tool_calls"] == [
        {"id": "x", "name": "t", "arguments": "{}"}
    ]


def test_only_the_latest_tool_results_are_reported():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"id": "old", "function": {"name": "a", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "old", "content": "old result"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "n1", "function": {"name": "papers-rag_search_papers", "arguments": '{"query": "x"}'}},
            {"id": "n2", "function": {"name": "calc_calculator", "arguments": '{"expression": "1+1"}'}},
        ]},
        {"role": "tool", "tool_call_id": "n1", "content": [{"type": "text", "text": '{"results": [{"rank": 1, "score": 0.5, "doc_key": "d"}]}'}]},
        {"role": "tool", "tool_call_id": "n2", "content": '{"value": 2}'},
    ]  # fmt: skip

    exchanges = new_tool_results(messages)

    assert [e.call_id for e in exchanges] == ["n1", "n2"]
    assert exchanges[0].arguments == {"query": "x"}
    assert retrieved_chunks(exchanges[0].result)[0]["doc_key"] == "d"
    assert retrieved_chunks(exchanges[1].result) is None
    assert new_tool_results([{"role": "user", "content": "q"}]) == []


def test_truncated_search_output_still_yields_the_documents():
    cut = '{"results": [{"rank": 1, "doc_key": "a"}, {"rank": 2, "doc_key": "b", "text": "lon'

    assert [c["doc_key"] for c in retrieved_chunks(cut)] == ["a", "b"]
