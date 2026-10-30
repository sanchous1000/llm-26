"""Sequential, time-limited calls to the ``opencode run`` CLI."""

import contextlib
import json
import logging
import os
import signal
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "opencode-go/deepseek-v4.1-flash"


class OpencodeError(Exception):
    """The model call failed; ``retryable`` tells whether another try may help."""

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


def ask(
    prompt: str,
    attachment: Path | None = None,
    *,
    model: str = DEFAULT_MODEL,
    timeout_s: int = 600,
    workdir: Path,
) -> str:
    """Send ``prompt`` (plus an optional file) and return the model's text answer."""
    workdir.mkdir(parents=True, exist_ok=True)
    command = ["opencode", "run", "-m", model, "--format", "json"]
    if attachment is not None:
        command += ["--file", str(attachment)]
    command.append(prompt)
    process = subprocess.Popen(  # noqa: S603 - fixed executable, no shell
        command,
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        _kill_group(process)
        process.communicate()
        raise OpencodeError(
            f"opencode timed out after {timeout_s}s", retryable=True
        ) from error
    if process.returncode != 0 and not stdout.strip():
        raise OpencodeError(
            f"opencode exited with {process.returncode}: {stderr[-300:]}",
            retryable=True,
        )
    return parse_events(stdout)


def _kill_group(process: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)


def parse_events(stdout: str) -> str:
    """Concatenate the ``text`` events of a ``--format json`` run."""
    texts: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "error":
            raise _error_from(event)
        if event.get("type") == "text":
            texts.append(event.get("part", {}).get("text", ""))
    answer = "".join(texts).strip()
    if not answer:
        raise OpencodeError("model returned no text", retryable=True)
    return answer


def _error_from(event: dict) -> OpencodeError:
    error = event.get("error", {})
    status = (error.get("response") or {}).get("status") or error.get("status")
    message = f"{error.get('type')}: {error.get('message')}"
    return OpencodeError(message, retryable=status not in (401, 402, 403))
