"""Running OpenCode (the ready-made agent) on the same tasks, logged through the proxy.

Per dialogue: an empty working directory with an ``opencode.json`` that defines
- provider ``lab3``: the LLM proxy (OpenAI-compatible) with the same model as our agent;
- MCP servers ``papers-rag`` (same server, own token) and ``calc`` (same calculator);
- agent ``papers``: our system prompt, the same step limit, built-in tools denied, and
  the model context limit of our agent, so OpenCode compacts at the same budget.

Per turn: a Langfuse ``turn`` observation is opened here, its trace/observation ids are
written into the provider headers, and ``opencode run --session <id>`` is executed; the
proxy logs the model and tool calls under that turn.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from source import config as project
from source.agent.config import AgentConfig, Prompts
from source.agent.opencode.proxy_server import running_proxy
from source.agent.opencode.proxy_stream import code_tool_calls, retrieved_chunks
from source.agent.tracing import STATUS_LEVELS, Tracing

logger = logging.getLogger(__name__)
PROVIDER_PACKAGE = "@ai-sdk/openai-compatible"


def opencode_config(
    config: AgentConfig, prompts: Prompts, headers: dict[str, str]
) -> dict[str, Any]:
    """The ``opencode.json`` of a dialogue (headers carry the Langfuse binding)."""
    oc = config.opencode
    model_ref = f"{oc.provider_id}/{config.model.name}"
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {oc.provider_id: _provider(config, headers)},
        "small_model": model_ref,
        "compaction": {"auto": True, "buffer": 0},
        "mcp": {"servers": _mcp_servers(config)},
        "agents": {
            oc.agent: {
                "model": model_ref,
                "system": prompts.system,
                "description": "Research assistant over the papers-rag corpus",
                "mode": "primary",
                "steps": config.limits.max_steps,
                "permissions": [
                    {"action": action, "resource": "*", "effect": "deny"}
                    for action in oc.deny_builtin_tools
                ],
            }
        },
    }


def _provider(config: AgentConfig, headers: dict[str, str]) -> dict[str, Any]:
    """The LLM proxy as an OpenAI-compatible provider with our model."""
    model = config.model
    return {
        "npm": PROVIDER_PACKAGE,
        "name": "lab3 LLM proxy",
        "options": {
            "baseURL": f"http://{config.proxy.host}:{config.proxy.port}/v1",
            "apiKey": "proxy-adds-the-real-key",
            "headers": headers,
        },
        "models": {
            model.name: {
                "name": model.name,
                "tool_call": True,
                "temperature": True,
                # OpenCode compacts when input + reserved output (+ buffer) exceed
                # the window: this gives the same input budget as our agent
                "limit": {
                    "context": config.context.limit_tokens + model.max_output_tokens,
                    "output": model.max_output_tokens,
                },
            }
        },
    }


def _mcp_servers(config: AgentConfig) -> dict[str, Any]:
    """The same papers-rag server (own token) and the same calculator as ours."""
    return {
        config.mcp.name: {
            "type": "remote",
            "url": config.mcp.url,
            "headers": {"Authorization": f"Bearer {{env:{config.mcp.token_env}}}"},
        },
        "calc": {
            "type": "local",
            "command": [sys.executable, "-m", "source.agent.opencode.calc_mcp"],
            "environment": {"PYTHONPATH": str(project.PROJECT_ROOT)},
        },
    }


@dataclass
class RunOutcome:
    """What one ``opencode run`` produced (parsed from its JSON events)."""

    answer: str = ""
    steps: int = 0
    tools_used: list[str] = field(default_factory=list)
    tool_errors: int = 0
    retrieved_docs: list[list[str]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    llm_calls: int = 0
    compactions: int = 0
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    compacted: bool = False


def parse_events(stdout: str) -> RunOutcome:
    """Fold the JSON event stream of ``opencode run --format json``."""
    outcome = RunOutcome()
    texts: dict[str, str] = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):  # stray non-event output is ignored
            _fold(outcome, texts, event)
    outcome.answer = "\n".join(t for t in texts.values() if t).strip()
    return outcome


def _fold(outcome: RunOutcome, texts: dict[str, str], event: dict[str, Any]) -> None:
    kind = event.get("type")
    part = event.get("part") or {}
    if kind == "text":
        texts[part.get("messageID", "") + part.get("id", "")] = part.get("text", "")
    elif kind == "step_start":
        outcome.steps += 1
    elif kind == "tool_use":
        _fold_tool(outcome, part)
    elif kind == "step_finish":
        tokens = part.get("tokens") or {}
        cache = int((tokens.get("cache") or {}).get("read") or 0)
        fresh = int(tokens.get("input") or 0)  # OpenCode reports cached input apart
        output = int(tokens.get("output") or 0) + int(tokens.get("reasoning") or 0)
        outcome.input_tokens += fresh + cache
        outcome.output_tokens += output
        outcome.cached_tokens += cache
    elif kind == "error":
        error = event.get("error") or {}
        outcome.errors.append(str(error.get("message") or error)[:500])
    if part.get("type") == "compaction" or kind == "compaction":
        outcome.compacted = True


def _fold_tool(outcome: RunOutcome, part: dict[str, Any]) -> None:
    """One OpenCode tool call; MCP tools run inside its code tool ``execute``."""
    state = part.get("state") or {}
    calls = _mcp_calls(part.get("tool") or "unknown", state)
    outcome.tools_used.extend(call.rpartition(".")[2] for call, _ in calls)
    failed = state.get("status") == "error" or any(s == "error" for _, s in calls)
    outcome.tool_errors += int(failed)
    if any(call.endswith("search_papers") for call, _ in calls):
        chunks = retrieved_chunks(str(state.get("output") or "")) or []
        outcome.retrieved_docs.append([c["doc_key"] for c in chunks])


def _mcp_calls(tool: str, state: dict[str, Any]) -> list[tuple[str, str]]:
    """(``server.tool``, status) pairs of a tool call."""
    metadata = (state.get("metadata") or {}).get("metadata") or {}
    recorded = metadata.get("toolCalls") if isinstance(metadata, dict) else None
    if recorded:
        return [(c.get("tool", "unknown"), c.get("status", "")) for c in recorded]
    if tool == "execute":
        code = str((state.get("input") or {}).get("code") or "")
        return [(call, state.get("status", "")) for call in code_tool_calls(code)]
    return [(tool, state.get("status", ""))]


def classify(
    outcome: RunOutcome, returncode: int | None, timed_out: bool, max_steps: int
) -> str:
    if timed_out:
        return "timeout"
    if outcome.errors or (returncode and not outcome.answer):
        return "error"
    if not outcome.answer or outcome.steps >= max_steps:
        return "step_limit"  # OpenCode forces a wrap-up text at the step budget
    return "ok"


class OpencodeSession:
    """One dialogue with OpenCode: a working directory and an OpenCode session id."""

    def __init__(
        self,
        config: AgentConfig,
        prompts: Prompts,
        tracing: Tracing,
        *,
        session_id: str,
        workdir: Path,
        user_id: str | None = None,
    ):
        self.config = config
        self.prompts = prompts
        self.tracing = tracing
        self.session_id = session_id
        self.user_id = user_id or config.langfuse.user_id
        self.workdir = workdir
        self.opencode_session = "ses_" + uuid.uuid4().hex[:24]
        self.turns = 0
        self.tags = [t for t in config.tags if not t.startswith("context:")] + [
            "context:opencode-compaction"
        ]
        workdir.mkdir(parents=True, exist_ok=True)

    async def ask(self, text: str) -> dict[str, Any]:
        """One turn: ``opencode run`` inside a Langfuse ``turn`` observation."""
        self.turns += 1
        started = time.monotonic()
        with (
            self.tracing.attributes(
                session_id=self.session_id,
                user_id=self.user_id,
                tags=self.tags,
                version=self.config.langfuse.version,
                trace_name=f"turn {self.turns}",
                metadata={"config": self.config.name, "turn": str(self.turns)},
            ),
            self.tracing.observe("turn", as_type="agent", input=text) as root,
        ):
            self._write_config(self._binding_headers(root))
            outcome, returncode, timed_out = await self._run(text)
            status = classify(
                outcome, returncode, timed_out, self.config.limits.max_steps
            )
            outcome.cost_usd = _cost(outcome, self.config)
            await self._use_proxy_usage(outcome, root.trace_id)
            answer = outcome.answer or _fallback_answer(status, outcome)
            root.update(
                output=answer,
                level=STATUS_LEVELS[status],
                status_message="; ".join(outcome.errors)[:1000] or None,
                metadata={
                    "status": status,
                    "steps": outcome.steps,
                    "tool_calls": len(outcome.tools_used),
                    "compacted": outcome.compacted,
                    "opencode_session": self.opencode_session,
                    "context_limit": self.config.context.limit_tokens,
                },
            )
        return self._record(outcome, answer, status, root, time.monotonic() - started)

    def _binding_headers(self, root: Any) -> dict[str, str]:
        """Provider headers that tell the proxy where to log this turn."""
        return {
            "x-lab3-session": self.session_id,
            "x-lab3-trace": root.trace_id or "",
            "x-lab3-parent": root.id or "",
            "x-lab3-tags": ",".join(self.tags),
            "x-lab3-user": self.user_id,
        }

    def _record(  # noqa: PLR0913 - the fields of one turn record
        self, outcome: RunOutcome, answer: str, status: str, root: Any, latency: float
    ) -> dict[str, Any]:
        """The per-turn record, in the same shape as ``TurnResult.to_dict``."""
        return {
            "answer": answer,
            "status": status,
            "error": "; ".join(outcome.errors) or None,
            "latency_s": round(latency, 2),
            "trace_id": root.trace_id,
            "observation_id": root.id,
            "steps": outcome.steps,
            "llm_calls": outcome.llm_calls,
            "tool_calls": len(outcome.tools_used),
            "tool_errors": outcome.tool_errors,
            "tools_used": outcome.tools_used,
            "retrieved_docs": outcome.retrieved_docs,
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
            "cost_usd": outcome.cost_usd,
            "context_limit": self.config.context.limit_tokens,
            "reduced": outcome.compacted,
            "compactions": outcome.compactions,
        }

    async def _use_proxy_usage(self, outcome: RunOutcome, trace_id: str | None):
        """Token and cost totals seen by the proxy (OpenCode events miss some)."""
        if not trace_id:
            return
        proxy = self.config.proxy
        url = f"http://{proxy.host}:{proxy.port}/usage/{trace_id}"
        try:
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                totals = (await client.get(url)).json()
        except (httpx.HTTPError, ValueError):
            logger.warning("proxy usage unavailable; using OpenCode events")
            return
        if totals.get("llm_calls"):
            outcome.llm_calls = int(totals["llm_calls"])
            outcome.compactions = int(totals.get("compactions", 0))
            outcome.compacted = outcome.compacted or outcome.compactions > 0
            outcome.input_tokens = int(totals["input"])
            outcome.output_tokens = int(totals["output"])
            outcome.cached_tokens = int(totals["cached"])
            outcome.cost_usd = float(totals["cost_usd"])

    def _write_config(self, headers: dict[str, str]) -> None:
        config = opencode_config(self.config, self.prompts, headers)
        path = self.workdir / "opencode.json"
        path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    async def _run(self, text: str) -> tuple[RunOutcome, int | None, bool]:
        oc = self.config.opencode
        command = [
            "opencode", "run", "--standalone", "--auto", "--format", "json",
            "--agent", oc.agent, "--session", self.opencode_session,
            "--model", f"{oc.provider_id}/{self.config.model.name}", text,
        ]  # fmt: skip
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.workdir,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PWD": str(self.workdir)},  # OpenCode trusts PWD
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), oc.run_timeout_s
            )
        except TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return self._parse(stdout), None, True
        if process.returncode:
            logger.warning(
                "opencode exited %s: %s", process.returncode, stderr.decode()[-500:]
            )
        return self._parse(stdout), process.returncode, False

    def _parse(self, stdout: bytes) -> RunOutcome:
        text = stdout.decode("utf-8", "replace")
        (self.workdir / f"turn-{self.turns:02d}.events.jsonl").write_text(text, "utf-8")
        return parse_events(text)


async def run_opencode_dialog(
    config: AgentConfig,
    prompts: Prompts,
    tracing: Tracing,
    turns: list[str],
    *,
    session_id: str,
    workdir: Path,
    on_turn=None,
) -> list[dict[str, Any]]:
    """All turns of one dialogue in one OpenCode session; starts the proxy if needed."""
    session = OpencodeSession(
        config, prompts, tracing, session_id=session_id, workdir=workdir
    )
    results = []
    async with running_proxy(config, tracing):
        for index, text in enumerate(turns):
            result = await session.ask(text)
            results.append(result)
            if on_turn is not None:
                on_turn(index, text, result)
    tracing.flush()
    return results


def _cost(outcome: RunOutcome, config: AgentConfig) -> float:
    """USD at the list prices of our agent (OpenCode reports 0 for custom providers)."""
    model = config.model
    fresh = outcome.input_tokens - outcome.cached_tokens
    return (
        fresh * model.price_input_per_mtok
        + outcome.cached_tokens * model.price_cache_read_per_mtok
        + outcome.output_tokens * model.price_output_per_mtok
    ) / 1e6


def _fallback_answer(status: str, outcome: RunOutcome) -> str:
    if status == "timeout":
        return "(OpenCode did not finish within the time limit)"
    if status == "step_limit":
        return "(OpenCode stopped without a final answer: step limit reached)"
    return f"(OpenCode failed: {'; '.join(outcome.errors)[:300]})"
