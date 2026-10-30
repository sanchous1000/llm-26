"""Final cleanup of Markdown text shared by all converters."""

import re

_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff\u2060]")
_NUMBERED_ITEM_RE = re.compile(r"^(\s*\d+\.)  +")
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_MATH_RE = re.compile(r"(\$[^$\n]+\$)")
# Pandoc escapes "<" and brackets defensively; these are noise in plain text.
_PANDOC_SEPARATOR = "<!-- -->"
_ESCAPED_LT_RE = re.compile(r"\\<(?![A-Za-z/!?])")
_DOUBLE_ESCAPED_ENTITY_RE = re.compile(r"\\?&(amp|lt|gt);")
_ENTITY_CHARS = {"amp": "&", "lt": "<", "gt": ">"}
_ESCAPED_PARENS_RE = re.compile(r"\\([()])")
_ESCAPED_BRACKETS_RE = re.compile(r"\\\[([^\[\]\n]{1,60}?)\\\](?![(:\[])")


def tidy(markdown: str) -> str:
    """Normalise whitespace and drop typographic debris and needless escapes."""
    out: list[str] = []
    fence: str | None = None
    in_display_math = False
    for raw in markdown.splitlines():
        line = _ZERO_WIDTH_RE.sub("", raw).replace("\u00a0", " ").rstrip()
        if fence is None and line == _PANDOC_SEPARATOR:
            continue  # pandoc marker between adjacent lists
        fence_match = _FENCE_RE.match(line)
        if fence is not None:
            if fence_match and fence_match.group(1).startswith(fence[0]):
                fence = None
        elif fence_match:
            fence = fence_match.group(1)
        elif line.strip() == "$$":
            in_display_math = not in_display_math
        elif not in_display_math:
            line = _clean_prose(line)
        out.append(line)
    return _BLANK_RUN_RE.sub("\n\n", "\n".join(out)).strip() + "\n"


def _clean_prose(line: str) -> str:
    line = _NUMBERED_ITEM_RE.sub(r"\1 ", line)
    parts = _INLINE_MATH_RE.split(line)
    for index in range(0, len(parts), 2):  # odd parts are inline math: keep verbatim
        parts[index] = _unescape(parts[index])
    return "".join(parts)


def _unescape(text: str) -> str:
    text = _DOUBLE_ESCAPED_ENTITY_RE.sub(lambda m: _ENTITY_CHARS[m.group(1)], text)
    text = _ESCAPED_LT_RE.sub("<", text)
    text = _ESCAPED_PARENS_RE.sub(r"\1", text)
    return _ESCAPED_BRACKETS_RE.sub(r"[\1]", text)
