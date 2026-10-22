"""The language model behind the answers: DeepSeek served by ``opencode run``."""

import tempfile
from pathlib import Path
from typing import Protocol

from source.evaluation import opencode
from source.rag.errors import RagError

_INLINE_PROMPT_LIMIT = 60_000  # longer prompts travel as an attached file


class Llm(Protocol):
    """Anything that turns a prompt into text."""

    model: str

    def complete(self, prompt: str) -> str: ...


class OpencodeLlm:
    """Calls ``opencode run`` in an empty directory (the agent cannot see the repo)."""

    def __init__(
        self,
        model: str = opencode.DEFAULT_MODEL,
        timeout_s: int = 300,
        retries: int = 1,
    ):
        self.model = model
        self._timeout_s = timeout_s
        self._retries = retries
        self._workdir = Path(tempfile.mkdtemp(prefix="opencode-rag-"))

    def complete(self, prompt: str) -> str:
        attachment, message = None, prompt
        if len(prompt) > _INLINE_PROMPT_LIMIT:
            attachment = self._workdir / "prompt.md"
            attachment.write_text(prompt, encoding="utf-8")
            message = (
                "Follow the instructions in the attached file and answer the question."
            )
        last: opencode.OpencodeError | None = None
        for _ in range(self._retries + 1):
            try:
                return opencode.ask(
                    message,
                    attachment,
                    model=self.model,
                    timeout_s=self._timeout_s,
                    workdir=self._workdir,
                )
            except opencode.OpencodeError as error:
                last = error
                if not error.retryable:
                    break
        raise RagError(f"language model call failed: {last}") from last
