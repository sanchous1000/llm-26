from __future__ import annotations

import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
REQUEST_TIMEOUT_SECONDS = 600
NATIVE_PROBE_TOKENS = 16
TUNED_OPTIONS = {"temperature": 0.2, "top_p": 0.85, "max_tokens": 128}
MODES = ("A_baseline", "B_tuned")
MODELS = [
    {"model": "qwen2.5:3b", "family": "Qwen", "approx_size_gb": 1.9},
    {"model": "llama3.2:3b", "family": "Llama", "approx_size_gb": 2.0},
    {"model": "mistral:7b", "family": "Mistral", "approx_size_gb": 4.4},
]
FIELDNAMES = [
    "run_id", "timestamp", "model", "family", "prompt_id", "prompt_title",
    "mode", "repeat_index", "parameters_json", "answer", "response_time_s",
    "prompt_tokens", "completion_tokens", "total_tokens", "token_count_source",
    "end_to_end_output_tokens_per_sec", "prompt_eval_tokens", "prompt_eval_time_s",
    "prompt_eval_tokens_per_sec", "probe_output_tokens", "probe_eval_time_s",
    "probe_generation_tokens_per_sec", "model_load_time_s", "task_score",
    "score_details_json", "error",
]
