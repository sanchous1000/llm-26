"""LLM proxy parts that need no network: SSE accumulation and tool extraction."""

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamAccumulator:
    """Rebuilds a chat completion from OpenAI streaming chunks (``data:`` lines)."""

    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    _buffer: str = ""

    def feed(self, data: bytes) -> None:
        self._buffer += data.decode("utf-8", errors="replace")
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            self._line(line.strip())

    def close(self) -> None:
        if self._buffer.strip():
            self._line(self._buffer.strip())
        self._buffer = ""

    def _line(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            return
        self.add_chunk(chunk)

    def add_chunk(self, chunk: dict[str, Any]) -> None:
        self.model = chunk.get("model") or self.model
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.content.append(delta["content"])
            if delta.get("reasoning_content"):
                self.reasoning.append(delta["reasoning_content"])
            for call in delta.get("tool_calls") or []:
                self._add_call(call)
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def _add_call(self, call: dict[str, Any]) -> None:
        slot = self.tool_calls.setdefault(
            call.get("index", len(self.tool_calls)),
            {"id": None, "name": "", "arguments": ""},
        )
        slot["id"] = call.get("id") or slot["id"]
        function = call.get("function") or {}
        slot["name"] += function.get("name") or ""
        slot["arguments"] += function.get("arguments") or ""

    def message(self) -> dict[str, Any]:
        """The assistant message in the same shape as a non-streamed response."""
        result: dict[str, Any] = {"role": "assistant", "content": "".join(self.content)}
        if self.reasoning:
            result["reasoning"] = "".join(self.reasoning)
        if self.tool_calls:
            result["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        return result


def completion_message(body: dict[str, Any]) -> dict[str, Any]:
    """The assistant message of a non-streamed chat completion."""
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    if message.get("reasoning_content"):
        result["reasoning"] = message["reasoning_content"]
    if message.get("tool_calls"):
        result["tool_calls"] = [
            {
                "id": c.get("id"),
                "name": (c.get("function") or {}).get("name"),
                "arguments": (c.get("function") or {}).get("arguments"),
            }
            for c in message["tool_calls"]
        ]
    return result


@dataclass(frozen=True)
class ToolExchange:
    """One tool call of the agent with the result it got back."""

    call_id: str
    name: str
    arguments: Any
    result: str


def new_tool_results(messages: list[dict[str, Any]]) -> list[ToolExchange]:
    """Tool results added since the last assistant message (what the agent just ran)."""
    last_assistant = max(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1
    )
    if last_assistant < 0:
        return []
    calls = {
        call.get("id"): call.get("function") or {}
        for call in messages[last_assistant].get("tool_calls") or []
    }
    exchanges = []
    for message in messages[last_assistant + 1 :]:
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id") or ""
        function = calls.get(call_id, {})
        exchanges.append(
            ToolExchange(
                call_id=call_id,
                name=function.get("name") or message.get("name") or "unknown",
                arguments=_maybe_json(function.get("arguments")),
                result=_text(message.get("content")),
            )
        )
    return exchanges


_DOC_KEY = re.compile(r'"doc_key"\s*:\s*"([^"]+)"')
CHUNK_FIELDS = ("rank", "score", "chunk_id", "doc_key", "section", "citation")
_CODE_CALL = re.compile(r"""tools(?:\.([\w-]+)|\[["']([\w-]+)["']\])\.(\w+)\s*\(""")


def retrieved_chunks(result: str) -> list[dict[str, Any]] | None:
    """Ranked chunks of search_papers results found anywhere in a tool output.

    Handles a plain result, results nested in an object (OpenCode's code tool returns
    ``{"a": {...}, "b": {...}}``) and JSON printed after ``Logs:``. None if absent.
    """
    hits: list[dict[str, Any]] = []
    for data in _json_values(result):
        _collect_hits(data, hits)
    if not hits:  # output cut by the agent (truncated JSON): read the keys directly
        hits = [{"doc_key": key} for key in _DOC_KEY.findall(result)]
    if not hits:
        return None
    return [{k: hit.get(k) for k in CHUNK_FIELDS} for hit in hits]


def code_tool_calls(code: str) -> list[str]:
    """``server.tool`` names called in OpenCode's code-mode tool (``execute``)."""
    return [
        f"{dotted or bracketed}.{tool}"
        for dotted, bracketed, tool in _CODE_CALL.findall(code or "")
    ]


def _collect_hits(data: Any, hits: list[dict[str, Any]]) -> None:
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list) and any(
            isinstance(h, dict) and "doc_key" in h for h in results
        ):
            hits.extend(h for h in results if isinstance(h, dict))
            return
        for value in data.values():
            _collect_hits(value, hits)
    elif isinstance(data, list):
        for value in data:
            _collect_hits(value, hits)


def _json_values(text: str) -> list[Any]:
    values = []
    for piece in (text, *text.split("Logs:")[1:]):
        parsed = _maybe_json(piece.strip())
        if not isinstance(parsed, str):
            values.append(parsed)
    return values


def usage_from(raw: dict[str, Any]) -> dict[str, int]:
    """OpenAI usage -> Langfuse usage_details."""
    details = raw.get("prompt_tokens_details") or {}
    prompt = int(raw.get("prompt_tokens") or 0)
    completion = int(raw.get("completion_tokens") or 0)
    return {
        "input": prompt,
        "output": completion,
        "cache_read_input_tokens": int(details.get("cached_tokens") or 0),
        "total": prompt + completion,
    }


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # [{"type": "text", "text": ...}]
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return json.dumps(content, ensure_ascii=False, default=str)


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
