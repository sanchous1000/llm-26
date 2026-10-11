import pytest

from source.indexing import blocks as b
from source.indexing.chunking import chunk_document
from source.indexing.config import ChunkingConfig
from source.indexing.diagnostics import evaluate_chunking, group_by_doc
from source.indexing.splitters import build_splitter
from source.indexing.units import build_units
from source.tests.helpers import WordCounter

COUNTER = WordCounter()


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


DOC = f"""# Paper Title

## Abstract

{words(30, "a")}

## 1 Introduction

{words(40, "i")}

We define the cost as

$$
c(x,y) = \\|x-y\\|^2
$$

where $x$ and $y$ are points of the space.

**Lemma 1.** {words(20, "l")}

**Proof.** {words(25, "p")}

### 1.1 Details

{words(35, "d")}

```
for i in range(n):
    update(i)
```

| a | b |
|---|---|
| 1 | 2 |

**Table 1:** caption words here.

## 2 Method

{words(60, "m")}

## References

- **[1]** Somebody. A paper.
"""


def test_block_parser_types_and_exact_spans():
    parsed = b.parse_blocks(DOC)

    kinds = [x.kind for x in parsed]
    assert kinds.count(b.HEADING) == 6 and b.MATH in kinds and b.CODE in kinds
    assert b.TABLE in kinds and b.LIST in kinds
    assert all(DOC[x.start : x.end] == DOC[x.start : x.end].strip() for x in parsed)
    heading = next(x for x in parsed if x.title == "1.1 Details")
    assert heading.level == 3


def test_block_parser_keeps_multiline_math_code_and_loose_lists_together():
    text = "$$\na\n\nb\n$$\n\n- one\n\n- two\n  cont\n\nafter\n\n```\nx\n\ny\n```\n"

    parsed = b.parse_blocks(text)

    assert [x.kind for x in parsed] == [b.MATH, b.LIST, b.PARAGRAPH, b.CODE]
    assert "b\n$$" in text[parsed[0].start : parsed[0].end]
    assert text[parsed[1].start : parsed[1].end].endswith("cont")


def test_units_glue_equation_with_intro_and_where_clause():
    parsed = b.parse_blocks(DOC)

    units = build_units(DOC, parsed, COUNTER)

    glued = next(u for u in units if "We define the cost" in DOC[u.start : u.end])
    text = DOC[glued.start : glued.end]
    assert "$$" in text and text.rstrip().endswith("of the space.")


def test_units_glue_table_with_caption_and_heading_with_text():
    units = build_units(DOC, b.parse_blocks(DOC), COUNTER)

    table_unit = next(u for u in units if "| a | b |" in DOC[u.start : u.end])
    assert "**Table 1:**" in DOC[table_unit.start : table_unit.end]
    first = units[0]
    assert DOC[first.start : first.end].startswith("# Paper Title")
    assert first.heading_level == 1


def cfg(**kwargs) -> ChunkingConfig:
    return ChunkingConfig(
        tokenizer="words", chunk_size=100, min_chunk_tokens=10, **kwargs
    )


def test_structured_chunks_never_exceed_size_and_cover_without_gaps_in_content():
    chunks = chunk_document("k", "Title", DOC, cfg(), COUNTER)

    assert all(c.token_count <= 100 for c in chunks)
    assert all(DOC[c.start : c.end] == c.text for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert "update(i)" in joined and "| 1 | 2 |" in joined


def test_references_are_excluded_unless_requested():
    without = chunk_document("k", "T", DOC, cfg(), COUNTER)
    with_refs = chunk_document("k", "T", DOC, cfg(include_references=True), COUNTER)

    assert not any("Somebody" in c.text for c in without)
    assert any("Somebody" in c.text for c in with_refs)


def test_headings_start_new_chunks_and_paths_are_recorded():
    chunks = chunk_document("k", "Title", DOC, cfg(), COUNTER)

    method = next(c for c in chunks if "m0" in c.text)
    assert method.section_path == ("2 Method",)
    details = next(c for c in chunks if "d0" in c.text)
    assert (
        details.section_path[-1] == "1.1 Details"
        or details.section_path[0] == "1 Introduction"
    )


def test_embed_text_has_title_and_path_and_no_images_or_page_markers():
    body = "# T\n\n## S\n\n<!-- page: 3 -->\n\n![Figure 1](http://x/y.png)\n\n" + words(
        30
    )
    with_headers = chunk_document("k", "My Title", body, cfg(), COUNTER)
    plain = chunk_document("k", "My Title", body, cfg(include_headings=False), COUNTER)

    assert with_headers[0].embed_text.startswith("My Title\nS\n\n")
    assert (
        "page:" not in with_headers[0].embed_text
        and "![" not in with_headers[0].embed_text
    )
    assert plain[0].embed_text.startswith("# T")  # no prefix: headings stay in the text
    assert with_headers[0].page_start == 3


def test_oversized_block_is_split_and_overlap_is_applied():
    big = "# T\n\n## S\n\n" + "\n".join(f"line{i} " + words(8) for i in range(40))
    chunks = chunk_document("k", "T", big, cfg(), COUNTER)

    assert len(chunks) > 2 and all(c.token_count <= 100 for c in chunks)
    paragraphs = "\n\n".join(f"par{i} " + words(30) for i in range(12))
    text = "# T\n\n## S\n\n" + paragraphs
    plain = chunk_document("k", "T", text, cfg(overlap=0), COUNTER)
    overlapped = chunk_document("k", "T", text, cfg(overlap=40), COUNTER)
    assert any(
        a.end > b_.start for a, b_ in zip(overlapped, overlapped[1:], strict=False)
    )
    assert not any(a.end > b_.start for a, b_ in zip(plain, plain[1:], strict=False))


@pytest.mark.parametrize("splitter", ["structured", "recursive", "fixed"])
def test_every_splitter_returns_valid_ordered_spans(splitter):
    text = "\n\n".join(f"par{i} " + words(20) for i in range(30))
    config = ChunkingConfig(splitter=splitter, tokenizer="words", chunk_size=60)

    spans = build_splitter(config, COUNTER).split(text)

    assert spans and all(0 <= s < e <= len(text) for s, e in spans)
    assert [s for s, _ in spans] == sorted(s for s, _ in spans)
    assert all(COUNTER.count(text[s:e]) <= 60 for s, e in spans)


def test_diagnostics_measure_containment_and_dilution():
    chunks = chunk_document("k", "T", DOC, cfg(), COUNTER)
    start = DOC.index("We define the cost")
    end = DOC.index("of the space.") + len("of the space.")
    questions = [{"doc_key": "k", "evidence": {"start": start, "end": end}}]

    report = evaluate_chunking(group_by_doc(chunks), questions, {"k": DOC}, COUNTER)

    assert report.contained == 1.0 and 0 < report.dilution_median <= 1
    assert report.chunks == len(chunks)
