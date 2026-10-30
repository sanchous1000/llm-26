"""Validated configuration of the dialogue agent (``configs/agent/agent.yaml``)."""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from source import config as project
from source.indexing.config import _apply_override

DEFAULT_AGENT_CONFIG = project.PROJECT_ROOT / "configs" / "agent" / "agent.yaml"
ENV_FILE = project.PROJECT_ROOT / ".env"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Frozen):
    name: str
    base_url: str
    api_key_env: str = "OPENCODE_API_KEY"
    temperature: float = 0.2
    max_output_tokens: int = Field(4096, gt=0)
    request_timeout_s: float = Field(120, gt=0)
    price_input_per_mtok: float = Field(0.0, ge=0)
    price_output_per_mtok: float = Field(0.0, ge=0)
    price_cache_read_per_mtok: float = Field(0.0, ge=0)


class McpConfig(_Frozen):
    name: str = "papers-rag"
    url: str
    token_env: str | None = None
    timeout_s: float = Field(60, gt=0)


class PromptFiles(_Frozen):
    system: str = "system_prompt.md"
    tools: str = "tools.yaml"
    summary: str = "summary_prompt.md"


class LimitsConfig(_Frozen):
    max_steps: int = Field(12, ge=1)
    max_tool_calls: int = Field(24, ge=1)
    turn_timeout_s: float = Field(300, gt=0)
    tool_timeout_s: float = Field(60, gt=0)
    tool_retries: int = Field(2, ge=0)


class ContextConfig(_Frozen):
    strategy: Literal["none", "window", "summary"] = "summary"
    limit_tokens: int = Field(32000, ge=500)
    keep_recent_turns: int = Field(2, ge=0)
    old_tool_result_chars: int = Field(300, ge=50)
    summary_max_tokens: int = Field(1200, ge=100)
    tokenizer: str = "o200k_base"


class LangfuseConfig(_Frozen):
    agent_tag: str = "agent:own"
    version: str = "own-0.1"
    user_id: str = "anonymous"


class ProxyConfig(_Frozen):
    """LLM proxy between a ready-made agent (OpenCode) and the provider."""

    host: str = "127.0.0.1"
    port: int = Field(4100, gt=0, lt=65536)


class OpencodeConfig(_Frozen):
    """OpenCode settings for the papers task (configs/agent/opencode.yaml)."""

    agent: str = "papers"
    provider_id: str = "lab3"
    deny_builtin_tools: tuple[str, ...] = (
        "shell", "edit", "write", "read", "grep", "glob", "list", "lsp", "webfetch",
        "websearch", "browser", "subagent", "task", "todowrite", "question", "skill",
    )  # fmt: skip
    run_timeout_s: float = Field(600, gt=0)


class AgentConfig(_Frozen):
    name: str
    model: ModelConfig
    mcp: McpConfig
    prompts: PromptFiles = PromptFiles()
    limits: LimitsConfig = LimitsConfig()
    context: ContextConfig = ContextConfig()
    langfuse: LangfuseConfig = LangfuseConfig()
    proxy: ProxyConfig = ProxyConfig()
    opencode: OpencodeConfig = OpencodeConfig()

    @property
    def tags(self) -> list[str]:
        """Langfuse tags that identify the agent, configuration and model."""
        return [
            self.langfuse.agent_tag,
            f"config:{self.name}",
            f"model:{self.model.name}",
            f"context:{self.context.strategy}",
        ]


class Prompts(BaseModel):
    """Texts loaded from the prompt files next to the configuration."""

    model_config = ConfigDict(frozen=True)

    system: str
    summary: str
    tools: dict[str, Any]


def load_agent_config(
    path: Path = DEFAULT_AGENT_CONFIG, overrides: list[str] | None = None
) -> AgentConfig:
    """Read the YAML file and apply ``section.key=value`` overrides."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        _apply_override(data, item)
    return AgentConfig.model_validate(data)


def load_prompts(
    config: AgentConfig, config_path: Path = DEFAULT_AGENT_CONFIG
) -> Prompts:
    base = config_path.parent
    tools = yaml.safe_load((base / config.prompts.tools).read_text(encoding="utf-8"))
    return Prompts(
        system=(base / config.prompts.system).read_text(encoding="utf-8").strip(),
        summary=(base / config.prompts.summary).read_text(encoding="utf-8").strip(),
        tools=tools or {},
    )


def load_env_file(path: Path = ENV_FILE) -> None:
    """Load ``KEY=value`` lines of the project .env (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"environment variable {name} is not set (see .env.example)")
    return value
