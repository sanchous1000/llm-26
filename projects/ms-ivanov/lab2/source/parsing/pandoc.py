"""Thin wrapper around the ``pandoc`` executable."""

import shutil
import subprocess

from source.parsing.errors import ConversionError

PANDOC_TIMEOUT_S = 300


def convert(text: str, source_format: str) -> str:
    """Convert ``text`` from ``source_format`` to GitHub-flavoured Markdown."""
    executable = shutil.which("pandoc")
    if executable is None:
        raise ConversionError("pandoc is not installed (https://pandoc.org/installing)")
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            [executable, "-f", source_format, "-t", "gfm", "--wrap=none"],
            input=text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=PANDOC_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ConversionError("pandoc timed out") from error
    if completed.returncode != 0:
        raise ConversionError(f"pandoc failed: {completed.stderr.strip()[:300]}")
    return completed.stdout
