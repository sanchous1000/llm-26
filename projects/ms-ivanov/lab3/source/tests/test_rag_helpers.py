"""Helpers shared by indexing and RAG: document body, quote location, opencode CLI."""

import json

import pytest

from source.parsing.document_body import read_body, references_span
from source.rag import opencode
from source.rag.locate import locate

PASSAGE = (
    "The Sinkhorn algorithm solves entropic optimal transport by alternately "
    "rescaling the rows and the columns of a Gibbs kernel until both marginals match "
    "the prescribed distributions, which converges linearly in the Hilbert metric."
)
BODY = (
    "# Title\n\n## 1 Intro\n\nSome intro text.\n\n### 1.1 Details\n\n"
    f"{PASSAGE}\n\n## References\n\n- **[1]** Somebody. A paper.\n\n## Appendix A Proofs\n\nMore."
)


def test_read_body_strips_front_matter(tmp_path):
    path = tmp_path / "document.md"
    path.write_text("---\nkey: k\n---\n\n# Title\n\nBody.\n")

    assert read_body(path) == "# Title\n\nBody.\n"


def test_references_span_ends_at_the_next_section():
    start, end = references_span(BODY)

    assert "Somebody" in BODY[start:end] and BODY[end:].startswith("## Appendix A")
    assert references_span("no refs here") is None


def test_locate_ignores_whitespace_and_returns_document_text():
    quote = PASSAGE.replace(" ", "  ", 3).replace("kernel until", "kernel\nuntil")

    found = locate(BODY, quote)

    assert found is not None and found.text == PASSAGE and found.score == 100.0
    assert BODY[found.start : found.end] == PASSAGE


def test_locate_tolerates_small_edits_but_rejects_inventions():
    edited = PASSAGE.replace("converges linearly", "converges quickly")

    assert locate(BODY, edited) is not None
    assert (
        locate(BODY, "Gradient boosting trees partition the feature space recursively.")
        is None
    )
    assert locate(BODY, "too short") is None


def test_parse_events_joins_text_and_raises_on_errors():
    ok = "\n".join(
        json.dumps(e)
        for e in (
            {"type": "step_start"},
            {"type": "text", "part": {"text": "[1,"}},
            {"type": "text", "part": {"text": " 2]"}},
        )
    )
    quota = json.dumps(
        {
            "type": "error",
            "error": {"type": "provider.quota", "status": 402, "message": "no funds"},
        }
    )

    assert opencode.parse_events(ok) == "[1, 2]"
    with pytest.raises(opencode.OpencodeError) as error:
        opencode.parse_events(quota)
    assert error.value.retryable is False
    with pytest.raises(opencode.OpencodeError):
        opencode.parse_events('{"type": "step_start"}')
