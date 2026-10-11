"""Opaque tokens that protect math and code from Markdown escaping.

Pandoc escapes underscores, asterisks and backslashes everywhere, which
would corrupt TeX.
Math and code blocks are therefore replaced by alphanumeric tokens before
the conversion and substituted back afterwards.
"""

import re

_TOKEN_RE = re.compile(r"ZZPH([A-Z])(\d+)ZZ")
_INLINE, _TABLE, _DISPLAY, _CODE = "I", "T", "D", "C"


class PlaceholderStore:
    """Collects protected snippets and restores them in the final Markdown."""

    def __init__(self) -> None:
        self._items: list[tuple[str, str]] = []

    def inline_math(self, tex: str, in_table: bool = False) -> str:
        return self._add(_TABLE if in_table else _INLINE, tex)

    def display_math(self, tex: str) -> str:
        return self._add(_DISPLAY, tex)

    def code_block(self, text: str) -> str:
        return self._add(_CODE, text)

    def _add(self, kind: str, payload: str) -> str:
        self._items.append((kind, payload))
        return f"ZZPH{kind}{len(self._items) - 1}ZZ"

    def restore(self, markdown: str) -> str:
        """Substitute every token; block tokens get their own paragraph."""
        return _TOKEN_RE.sub(self._render, markdown)

    def _render(self, match: re.Match) -> str:
        kind, payload = self._items[int(match.group(2))]
        if kind == _INLINE:
            return f"${payload}$"
        if kind == _TABLE:
            return f"${_escape_pipes(payload)}$"
        if kind == _DISPLAY:
            return f"\n$$\n{payload}\n$$\n"
        return _fence(payload)


def _escape_pipes(tex: str) -> str:
    """Table cells cannot contain a bare ``|``; use the TeX delimiters instead."""
    return tex.replace("\\|", "\\Vert ").replace("|", "\\vert ")


def _fence(code: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", code)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"\n{fence}\n{code}\n{fence}\n"
