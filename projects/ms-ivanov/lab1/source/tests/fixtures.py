"""Общие фикстуры тестов: минимальная запись прогона."""

from __future__ import annotations

from typing import Any


def make_record(prompt: str, preset: str, repeat: int, ts: str, answer: str = "ok") -> dict[str, Any]:
    return {
        "run_id": f"{prompt}__{preset}__r{repeat}", "model": "m", "timestamp": ts,
        "prompt_id": prompt, "preset_key": preset, "repeat": repeat, "repeats_total": 2,
        "prompt_title": "t", "prompt_type": "generation", "preset_label": preset, "mode": "tuned",
        "params_desc": "-", "system": "s", "user": "u", "answer": answer, "reasoning": "",
        "error": "", "quality_score": 1.0, "checks": [],
        "metrics": {"latency_s": 1.0, "completion_tokens": 10},
    }
