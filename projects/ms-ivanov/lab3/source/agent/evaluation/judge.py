"""Grading of the evaluation runs and scores in Langfuse.

Per turn: ``check: exact`` turns are graded by the key facts alone; the others by an LLM
judge (configs/agent/judge.yaml) that compares the answer with the reference and the
key facts. Scores go to Langfuse on three levels: the turn observation, the task
(dataset item trace) and the run (experiment).
"""

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from source import config as project
from source.agent.config import require_env
from source.agent.evaluation.dataset import Task, Turn, missing_facts
from source.agent.evaluation.metrics import summarise, task_success, turn_rows

logger = logging.getLogger(__name__)
JUDGE_CONFIG = project.PROJECT_ROOT / "configs" / "agent" / "judge.yaml"
SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
FAILED_VERDICTS = ("unparsable", "judge error")  # reasons of gradings to redo
RUN_SCORES = (
    "success_rate", "success_single", "success_dialog", "turn_accuracy",
    "history_accuracy", "history_accuracy_before_reduction",
    "history_accuracy_after_reduction", "memory_probe_accuracy", "cost_per_task_usd",
    "latency_per_turn_s", "failure_rate", "step_limit_rate", "tool_correctness",
    "recall@5", "mrr",
)  # fmt: skip
Ask = Callable[[str], Any]  # async prompt -> text


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.0
    max_tokens: int = 4000
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0
    concurrency: int = 4
    prompt: str = "judge_prompt.md"


def load_judge(path: Path = JUDGE_CONFIG) -> tuple[JudgeConfig, str]:
    config = JudgeConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))
    return config, (path.parent / config.prompt).read_text("utf-8")


def judge_prompt(template: str, task: Task, index: int, answer: str) -> str:
    spec = task.turns[index]
    history = "\n".join(f"- {t.user}" for t in task.turns[:index]) or "(none)"
    return template.format(
        history=history,
        question=spec.user,
        reference=spec.expected_output,
        facts=_facts(spec, answer),
        answer=answer or "(empty answer)",
    )


def _facts(spec: Turn, answer: str) -> str:
    missing = set(missing_facts(answer, spec.must_include))
    lines = [
        f"- {' | '.join(group)}: {'missing' if group in missing else 'found'}"
        for group in spec.must_include
    ]
    return "\n".join(lines) or "(none)"


def parse_verdict(text: str) -> dict[str, Any]:
    """The JSON verdict of the judge (tolerates text around it)."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:  # e.g. LaTeX backslashes: read the fields directly
        verdict = re.search(r'"verdict"\s*:\s*"(\w+)"', text or "")
        reason = re.search(r'"reason"\s*:\s*"(.*?)"\s*}', text or "", re.DOTALL)
        data = {"verdict": verdict.group(1) if verdict else "",
                "reason": reason.group(1) if reason else ""}  # fmt: skip
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in SCORES:
        verdict = "incorrect"
        data["reason"] = f"unparsable judge output: {(text or '')[:200]}"
    return {"verdict": verdict, "reason": str(data.get("reason", ""))[:1000]}


async def judge_turn(
    task: Task, index: int, turn: dict[str, Any], ask: Ask, template: str
) -> dict[str, Any]:
    spec = task.turns[index]
    answer = turn.get("answer") or ""
    found = len(spec.must_include) - len(missing_facts(answer, spec.must_include))
    facts = {"facts_found": found, "facts_total": len(spec.must_include)}
    if spec.check == "exact":
        correct = found == len(spec.must_include)
        verdict = "correct" if correct else "incorrect"
        return {**facts, "method": "exact", "verdict": verdict, "reason": "key facts"}
    if not answer.strip() or answer.startswith("(OpenCode"):
        return {
            **facts,
            "method": "rule",
            "verdict": "incorrect",
            "reason": "no answer",
        }
    prompt = judge_prompt(template, task, index, answer)
    reply = await ask(prompt)
    if not (reply or "").strip():  # a reasoning model may spend the budget thinking
        reply = await ask(prompt)
    verdict = parse_verdict(reply)
    return {**facts, "method": "llm", **verdict}


def with_scores(verdict: dict[str, Any]) -> dict[str, Any]:
    score = SCORES[verdict["verdict"]]
    return {**verdict, "score": score, "correct": score == 1.0}


async def judge_records(  # noqa: PLR0917
    records: Sequence[dict[str, Any]], tasks: dict[str, Task], ask: Ask, template: str,
    concurrency: int = 4, previous: dict[tuple[str, int], dict] | None = None,
) -> list[dict[str, Any]]:  # fmt: skip
    """Records with a ``judge`` verdict added to every turn (input not mutated)."""
    slots = asyncio.Semaphore(concurrency)

    async def one(task: Task, index: int, turn: dict[str, Any]) -> dict[str, Any]:
        old = (previous or {}).get((task.id, index))
        if old and not str(old.get("reason", "")).startswith(FAILED_VERDICTS):
            return {**turn, "judge": with_scores(old)}
        async with slots:
            try:
                verdict = await judge_turn(task, index, turn, ask, template)
            except Exception as error:  # noqa: BLE001 - one failed grading is reported
                logger.warning("judge failed for %s/%d: %s", task.id, index, error)
                verdict = {
                    "method": "llm",
                    "verdict": "incorrect",
                    "reason": f"judge error: {error}",
                }
            return {**turn, "judge": with_scores(verdict)}

    judged = []
    for record in records:
        task = tasks[record["task_id"]]
        turns = await asyncio.gather(
            *(one(task, i, t) for i, t in enumerate(record["turns"][: len(task.turns)]))
        )
        judged.append({**record, "turns": list(turns)})
    return judged


def verdicts_of(judged: Sequence[dict[str, Any]]) -> dict[tuple[str, int], dict]:
    """Verdicts of an earlier judging by (task id, turn), to reuse the good ones."""
    return {
        (record["task_id"], turn["turn"]): turn["judge"]
        for record in judged
        for turn in record["turns"]
        if "judge" in turn
    }


def openai_judge(config: JudgeConfig) -> tuple[Ask, dict[str, float]]:
    """An ``ask`` coroutine for the judge model and a usage counter."""
    client = AsyncOpenAI(
        base_url=config.base_url,
        api_key=require_env(config.api_key_env),
        default_headers={"x-opencode-session": "lab3-judge"},
        max_retries=3,
        timeout=300,
    )
    usage = {"calls": 0, "input": 0, "output": 0, "cost_usd": 0.0}

    async def ask(prompt: str) -> str:
        response = await client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        if response.usage:
            usage["calls"] += 1
            usage["input"] += response.usage.prompt_tokens
            usage["output"] += response.usage.completion_tokens
            usage["cost_usd"] += (
                response.usage.prompt_tokens * config.price_input_per_mtok
                + response.usage.completion_tokens * config.price_output_per_mtok
            ) / 1e6
        return response.choices[0].message.content or ""

    return ask, usage


class _IdempotentScores:
    """Scores with ids derived from their target and name: a re-push overwrites."""

    def __init__(self, client: Any):
        self._client = client

    def create_score(self, **fields: Any) -> None:
        key = "|".join(
            str(fields.get(k))
            for k in ("name", "trace_id", "observation_id", "dataset_run_id")
        )
        score_id = uuid.uuid5(uuid.NAMESPACE_URL, f"lab3-score|{key}").hex
        self._client.create_score(score_id=score_id, **fields)

    def flush(self) -> None:
        self._client.flush()


def push_scores(client: Any, judged: Sequence[dict[str, Any]], tasks: dict[str, Task],
                experiment_id: str | None) -> dict[str, Any]:  # fmt: skip
    """Turn, task and run scores in Langfuse; returns the run summary."""
    client = _IdempotentScores(client)
    all_rows = []
    for record in judged:
        task = tasks[record["task_id"]]
        rows = turn_rows(task, record)
        all_rows.extend(rows)
        for turn, row in zip(record["turns"], rows, strict=False):
            judge = turn["judge"]
            target = {"trace_id": turn.get("trace_id") or record.get("item_trace_id")}
            if turn.get("observation_id"):
                target["observation_id"] = turn["observation_id"]
            client.create_score(
                name="turn_correct",
                value=float(row.correct),
                data_type="BOOLEAN",
                comment=judge.get("reason"),
                **target,
            )
            client.create_score(name="turn_score", value=row.score, **target)
            if row.depends_on_history:
                client.create_score(
                    name="history_turn_correct",
                    value=float(row.correct),
                    data_type="BOOLEAN",
                    **target,
                )
        item = {"trace_id": record.get("item_trace_id")}
        stats = summarise(rows)
        client.create_score(name="task_success", value=float(task_success(rows)),
                            data_type="BOOLEAN", **item)  # fmt: skip
        client.create_score(name="turn_accuracy", value=stats["turn_accuracy"], **item)
        client.create_score(name="cost_usd", value=stats["cost_total_usd"], **item)
        client.create_score(
            name="latency_s", value=sum(r.latency_s for r in rows), **item
        )
    summary = summarise(all_rows)
    if experiment_id:
        for name in RUN_SCORES:
            if summary.get(name) is not None:
                client.create_score(name=name, value=float(summary[name]),
                                    dataset_run_id=experiment_id)  # fmt: skip
    client.flush()
    return summary
