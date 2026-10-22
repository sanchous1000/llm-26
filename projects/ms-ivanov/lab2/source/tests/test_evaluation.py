import json

import pytest

from source.evaluation import opencode
from source.evaluation.candidates import parse_answer, verify
from source.evaluation.documents import (
    heading_path,
    page_at,
    read_body,
    references_span,
    split_into_parts,
    strip_references,
)
from source.evaluation.locate import locate
from source.evaluation.prompts import build_prompt

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


def test_references_are_removed_without_touching_appendices():
    stripped = strip_references(BODY)

    assert "Somebody" not in stripped and "## Appendix A Proofs" in stripped
    assert references_span("no refs here") is None


def test_split_into_parts_cuts_at_headings_and_respects_limit():
    text = "## A\n\n" + "x" * 50 + "\n\n## B\n\n" + "y" * 50 + "\n\n## C\n\n" + "z" * 50

    parts = split_into_parts(text, 80)

    assert len(parts) == 3 and all(len(p.text) <= 80 for p in parts)
    assert "".join(p.text for p in parts) == text
    assert [p.index for p in parts] == [1, 2, 3] and parts[0].total == 3


def test_split_into_parts_hard_cuts_huge_sections_at_paragraphs():
    text = "## A\n\n" + "\n\n".join("w" * 30 for _ in range(10))

    parts = split_into_parts(text, 100)

    assert len(parts) > 1 and "".join(p.text for p in parts) == text


def test_heading_path_and_page_markers():
    offset = BODY.index("The Sinkhorn")

    assert heading_path(BODY, offset) == ["Title", "1 Intro", "1.1 Details"]
    assert page_at("<!-- page: 1 -->\n\na\n\n<!-- page: 2 -->\n\nb", 40) == 2
    assert page_at("no markers", 3) is None


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


def test_parse_answer_handles_fences_and_prose():
    raw = 'Here you go:\n```json\n[{"question": "q", "evidence": "e"}, "junk"]\n```'

    assert parse_answer(raw) == [{"question": "q", "evidence": "e"}]
    with pytest.raises(ValueError):
        parse_answer("no json here")


def _item(**overrides):
    base = {
        "question": "How does the Sinkhorn algorithm solve entropic optimal transport?",
        "question_ru": "Как?",
        "type": "method",
        "answer": "By alternate rescaling.",
        "evidence": PASSAGE,
    }
    return {**base, **overrides}


def test_verify_accepts_located_evidence_with_metadata():
    candidate = verify([_item()], BODY, part=1)[0]

    assert candidate.status == "ok" and candidate.flags == ()
    assert candidate.evidence["section_path"][-1] == "1.1 Details"
    assert BODY[candidate.evidence["start"] : candidate.evidence["end"]] == PASSAGE


def test_verify_rejects_bad_items_and_flags_leaky_questions():
    missing = verify([_item(evidence="")], BODY, 1)[0]
    invented = verify([_item(evidence="Completely invented sentence " * 8)], BODY, 1)[0]
    short = verify([_item(evidence=PASSAGE[:60])], BODY, 1)[0]
    leaky = verify([_item(question="What does Theorem 2 say about it?")], BODY, 1)[0]

    assert missing.status == "invalid"
    assert invented.status == "evidence_not_found"
    assert short.status in ("evidence_size", "evidence_not_found")
    assert leaky.flags == ("leaky_question",)


def test_prompt_mentions_count_and_part():
    prompt = build_prompt(count=5, part=2, parts=3)

    assert "exactly 5 questions" in prompt and "part 2 of 3" in prompt


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


def test_review_and_export_roundtrip(tmp_path):
    from source.evaluation.export import export
    from source.evaluation.review import render

    candidates = verify([_item(), _item(evidence="invented " * 40)], BODY, part=1)
    data = {
        "key": "doc",
        "title": "Doc",
        "body_chars": len(BODY),
        "body_sha256": "abc",
        "candidates": [c.to_dict() for c in candidates],
    }
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(data))

    listing = render(path)
    questions = export(
        {"doc": [{"i": 0, "question": "Edited?"}]}, tmp_path, tmp_path / "q.json"
    )

    assert "DROPPED evidence_not_found" in listing and "[0] (method)" in listing
    assert questions[0]["id"] == "doc-q1" and questions[0]["question"] == "Edited?"
    assert questions[0]["evidence"]["text"] == PASSAGE
    with pytest.raises(ValueError):
        export({"doc": [1]}, tmp_path, tmp_path / "q.json")
