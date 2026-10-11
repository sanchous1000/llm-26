"""Blocks -> atomic "units" that a chunk may never split.

A unit glues together blocks that only make sense together in a scientific
text: a heading with the text that follows it, a display equation with the
sentence introducing it and the "where ..." clause explaining its symbols,
figure panels with their captions, a table/algorithm with its caption.
"""

import re
from dataclasses import dataclass

from source.indexing import blocks as b
from source.indexing.tokens import TokenCounter

_WHERE_RE = re.compile(
    r"^(where|with|and|such that|if|for|here|in which|denotes?|which|so that|then)\b",
    re.IGNORECASE,
)
_CAPTION_RE = re.compile(
    r"^\*\*(Figure|Table|Algorithm|Fig\.)\b|^\*\*\([a-z]\)\*\*", re.IGNORECASE
)
_FLOAT_LEAD_RE = re.compile(r"^\*\*(Table|Algorithm)\b", re.IGNORECASE)
_THEOREM_RE = re.compile(
    r"^\*\*(Theorem|Lemma|Proposition|Corollary|Definition|Assumption|Claim|Remark|"
    r"Example|Proof|Conjecture|Property|Fact|Observation|Notation|Problem)\b",
    re.IGNORECASE,
)
_FLOATS = (b.IMAGE, b.TABLE, b.CODE)


@dataclass(frozen=True)
class Unit:
    """An indivisible span of the body."""

    start: int
    end: int
    tokens: int
    heading_level: int  # level of the heading that opens the unit, 0 if none
    break_before: bool  # a natural place for a chunk boundary (theorem-like lead)


@dataclass
class _Group:
    blocks: list[b.Block]
    wants_where: bool = False  # last block was a display equation
    wants_float: bool = False  # caption that precedes its table/algorithm

    @property
    def last_kind(self) -> str:
        return self.blocks[-1].kind


def build_units(body: str, blocks: list[b.Block], counter: TokenCounter) -> list[Unit]:
    """Group ``blocks`` and count the tokens of each group."""
    groups = _group(body, blocks)
    return [_to_unit(body, group, counter) for group in groups]


def _group(body: str, blocks: list[b.Block]) -> list[_Group]:
    groups: list[_Group] = []
    pending: list[b.Block] = []  # headings / page markers glue to what follows
    for block in blocks:
        if block.kind in (b.HEADING, b.PAGE):
            pending.append(block)
            continue
        text = body[block.start : block.end]
        target = None if pending else _merge_target(groups, block, text)
        if target is not None:
            target.blocks.append(block)
            _update_flags(target, block, text)
            continue
        group = _Group([*pending, block])
        pending = []
        _update_flags(group, block, text)
        groups.append(group)
    if pending:  # trailing headings with no content
        groups.append(_Group(pending))
    return groups


def _merge_target(  # noqa: PLR0911 - one early return per glue rule
    groups: list[_Group], block: b.Block, text: str
) -> _Group | None:
    if not groups:
        return None
    previous = groups[-1]
    if block.kind == b.MATH and previous.last_kind in (b.PARAGRAPH, b.LIST, b.MATH):
        return previous
    if block.kind == b.PARAGRAPH and previous.wants_where and _WHERE_RE.match(text):
        return previous
    if block.kind in _FLOATS and previous.wants_float:
        return previous
    if block.kind == b.IMAGE and previous.last_kind == b.IMAGE:
        return previous
    if (
        block.kind == b.PARAGRAPH
        and _CAPTION_RE.match(text)
        and previous.last_kind in _FLOATS
    ):
        return previous
    return None


def _update_flags(group: _Group, block: b.Block, text: str) -> None:
    group.wants_where = block.kind == b.MATH
    group.wants_float = block.kind == b.PARAGRAPH and bool(_FLOAT_LEAD_RE.match(text))


def _to_unit(body: str, group: _Group, counter: TokenCounter) -> Unit:
    start, end = group.blocks[0].start, group.blocks[-1].end
    first = group.blocks[0]
    content = next((x for x in group.blocks if x.kind not in (b.HEADING, b.PAGE)), None)
    lead = body[content.start : content.end] if content else ""
    return Unit(
        start=start,
        end=end,
        tokens=counter.count(body[start:end]),
        heading_level=first.level if first.kind == b.HEADING else 0,
        break_before=bool(_THEOREM_RE.match(lead)) or first.kind == b.HEADING,
    )
