import asyncio

import pytest

from source.rag.commands import build_parser_for_tests, chat_loop
from source.rag.engine import RagEngine
from source.rag.errors import RagError
from source.rag.llm import OpencodeLlm
from source.rag.mcp_server import build_server
from source.rag.models import RetrievedChunk
from source.rag.prompt import (
    best_snippet,
    build_prompt,
    make_citations,
    parse_citation_numbers,
    parse_quotes,
    split_sources_block,
)
from source.rag.render import format_answer, format_chunks, language_of
from source.rag.retriever import Retriever, resolve_mode
from source.tests.helpers import FakeEmbedder, config


def chunk(
    rank=1, text="Alpha beta gamma. Delta epsilon zeta.", tokens=100, **kw
) -> RetrievedChunk:
    fields = {
        "rank": rank, "score": 0.5, "chunk_id": f"c{rank}", "doc_key": "doc",
        "title": "A Paper", "authors": ("Jane Doe", "John Roe"), "year": 2020,
        "venue": "V", "source_url": "http://x", "ord": rank, "start_char": 0,
        "end_char": len(text), "token_count": tokens,
        "section_path": ("1 Intro", "1.1 Setup"), "page": None, "text": text,
    }  # fmt: skip
    return RetrievedChunk(**{**fields, **kw})


# ---- prompt ---------------------------------------------------------------


def test_prompt_numbers_passages_and_respects_the_token_budget():
    contexts = [chunk(1, tokens=400), chunk(2, tokens=400), chunk(3, tokens=400)]

    bundle = build_prompt("What is alpha?", contexts, max_context_tokens=900)

    assert len(bundle.contexts) == 2
    assert "[1] A Paper (Doe et al., 2020) | 1 Intro > 1.1 Setup" in bundle.text
    assert "[3]" not in bundle.text.split("Passages:")[1]
    assert bundle.text.rstrip().endswith("Question: What is alpha?\n\nAnswer:")
    assert "SOURCES" in bundle.text  # the output format is part of the instructions


def test_prompt_always_keeps_the_first_passage_and_includes_history():
    bundle = build_prompt(
        "And beta?", [chunk(1, tokens=5000)], history=[("What is alpha?", "A letter.")],
        max_context_tokens=100,
    )  # fmt: skip

    assert len(bundle.contexts) == 1
    assert "User: What is alpha?\nAssistant: A letter." in bundle.text


def test_prompt_strips_page_markers_and_images_from_passages():
    text = "<!-- page: 3 -->\n\n![f](http://i.png)\n\nReal content here."

    bundle = build_prompt("q", [chunk(text=text, page=3)], max_context_tokens=1000)

    assert "page: 3 -->" not in bundle.text and "![f]" not in bundle.text
    assert "| page 3" in bundle.text and "Real content here." in bundle.text


def test_sources_block_with_header_without_header_and_absent():
    quote = '[1] "Alpha beta gamma is the claim in passage one."'

    with_header = split_sources_block(f"Answer [1].\n\nSOURCES:\n{quote}")
    bold_header = split_sources_block(f"Answer [1].\n\n**SOURCES:**\n{quote}")
    headerless = split_sources_block(f"Answer [1].\n\n{quote}\n{quote}")
    none = split_sources_block("Plain answer [1] with no quotes.")

    assert with_header == ("Answer [1].", quote)
    assert bold_header[0] == "Answer [1]." and bold_header[1] == quote
    assert headerless[0] == "Answer [1]." and headerless[1].count("[1]") == 2
    assert none == ("Plain answer [1] with no quotes.", "")


def test_citation_numbers_are_ordered_deduplicated_and_validated():
    valid, unknown = parse_citation_numbers("A [2]. B [1][2]. C [7].", available=3)

    assert valid == [2, 1] and unknown == [7]


def test_quotes_may_repeat_per_number():
    block = '[1] "first quote here"\n[1] "second quote here"\n[3] “curly quote”'

    assert parse_quotes(block) == {
        1: ["first quote here", "second quote here"],
        3: ["curly quote"],
    }


def test_citations_use_verified_quotes_and_fall_back_to_a_matching_sentence():
    text = (
        "Intro sentence here. The Monge gap measures optimality of a map. Last words."
    )
    contexts = [
        chunk(1, text=text),
        chunk(2, text="Entirely different content about topology."),
    ]
    answer = "The gap measures optimality [1]. Topology is covered [2]."
    block = '[1] "The Monge gap measures optimality of a map."\n[2] "this quote does not occur anywhere"'

    citations, unknown = make_citations(
        answer, block, contexts, "What is the Monge gap?", 200
    )

    assert [c.number for c in citations] == [1, 2] and not unknown
    assert (
        citations[0].verified
        and citations[0].snippet == "The Monge gap measures optimality of a map."
    )
    assert not citations[1].verified and "topology" in citations[1].snippet.lower()


def test_unverified_quote_of_wrong_passage_is_not_trusted():
    contexts = [chunk(1, text="Passage one talks about transport plans and costs.")]

    citations, _ = make_citations(
        "Claim [1].",
        '[1] "A sentence copied from a completely different paper."',
        contexts,
        "q",
        200,
    )

    assert not citations[0].verified


def test_best_snippet_prefers_overlap_and_respects_the_limit():
    text = (
        "Cats purr loudly. The Sinkhorn algorithm rescales rows and columns. Dogs bark."
    )

    snippet = best_snippet(text, "How does the Sinkhorn algorithm work?", 60)

    assert "Sinkhorn" in snippet and len(snippet) <= 60


# ---- mode resolution / models / rendering ---------------------------------


@pytest.mark.parametrize(
    ("query", "requested", "built", "expected"),
    [
        ("optimal transport", "auto", "hybrid", "hybrid"),
        ("оптимальный транспорт", "auto", "hybrid", "dense"),
        ("mixed текст with words", "auto", "hybrid", "hybrid"),
        ("anything", "auto", "dense", "dense"),
        ("anything", "sparse", "hybrid", "sparse"),
    ],
)
def test_resolve_mode(query, requested, built, expected):
    assert resolve_mode(query, requested, built) == expected


def test_unavailable_mode_is_rejected():
    with pytest.raises(ValueError, match="not available"):
        resolve_mode("q", "sparse", "dense")


def test_chunk_labels_and_serialisation():
    c = chunk(text="x" * 50)

    assert c.short_citation == "Doe et al., 2020"
    assert chunk(authors=("Solo Author",)).short_citation == "Author, 2020"
    data = c.to_dict(max_chars=10)
    assert (
        data["text"].endswith("[...]")
        and data["truncated"]
        and data["section"] == "1 Intro > 1.1 Setup"
    )


def test_rendering_in_both_languages():
    assert language_of("Что такое?") == "ru" and language_of("What?") == "en"
    assert "(nothing found)" in format_chunks([])
    assert "#1 A Paper (Doe et al., 2020)" in format_chunks([chunk()])


# ---- engine / retriever on a tiny real index ------------------------------


class ScriptedLlm:
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_retriever_returns_citable_passages_with_metadata(retriever):
    retr, _ = retriever
    probe = retr.search("anything", top_k=1)[0]

    hits = retr.search(probe.text.replace("\n", " "), top_k=3, mode="dense")

    assert len(hits) == 3 and all(h.title.startswith("Doc ") for h in hits)
    assert hits[0].rank == 1 and hits[0].doc_key in ("doc0", "doc1")
    assert hits[0].authors == ("A",) and hits[0].year == 2020 and hits[0].text


def test_retriever_can_restrict_the_search_to_documents(retriever):
    retr, _ = retriever

    hits = retr.search("anything", top_k=10, doc_keys=["doc1"])

    assert hits and {h.doc_key for h in hits} == {"doc1"}


def test_neighbors_documents_and_outline(retriever):
    retr, _ = retriever
    middle = next(
        h for h in retr.search("anything", top_k=20, doc_keys=["doc0"]) if h.ord == 1
    )

    window = retr.neighbors(middle.chunk_id, before=1, after=1)

    assert [c.ord for c in window] == [0, 1, 2]
    assert [d.key for d in retr.documents()] == ["doc0", "doc1"]
    assert [d.key for d in retr.documents("doc1")] == ["doc1"]
    assert retr.outline("doc0")[0]["chunks"]
    with pytest.raises(RagError):
        retr.neighbors("missing")
    with pytest.raises(RagError):
        retr.outline("nope")


def test_missing_collection_gives_actionable_error(sessions, client):
    retr = Retriever(config(), client, sessions, embedder_factory=FakeEmbedder)

    with pytest.raises(RagError, match="python -m source index"):
        retr.search("q")


def test_engine_answers_with_verified_citations_and_records_timings(retriever):
    retr, cfg = retriever
    top = retr.search("anything", top_k=2)
    quote = " ".join(top[0].text.split()[:12]).replace("#", "")
    llm = ScriptedLlm([f'Answer based on source [1].\n\nSOURCES:\n[1] "{quote}"'])

    answer = RagEngine(retr, llm, cfg.rag).answer("anything", top_k=2)

    assert answer.text == "Answer based on source [1]."
    assert [c.number for c in answer.citations] == [1]
    assert answer.model == "scripted" and answer.mode in ("hybrid", "dense")
    assert {"retrieve_s", "llm_s"} <= set(answer.timings)
    assert answer.to_dict()["citations"][0]["doc_key"] == top[0].doc_key


def test_engine_flags_nonexistent_citations_and_rejects_empty_answers(retriever):
    retr, cfg = retriever
    engine = RagEngine(retr, ScriptedLlm(["Claim [9].", "  "]), cfg.rag)

    answer = engine.answer("anything", top_k=1)
    assert answer.unknown_citations == (9,) and not answer.citations
    with pytest.raises(RagError, match="empty"):
        engine.answer("anything", top_k=1)


def test_engine_condenses_follow_ups_and_keeps_history_in_the_prompt(retriever):
    retr, cfg = retriever
    llm = ScriptedLlm(["standalone query about alpha", "Because of alpha [1]."])

    answer = RagEngine(retr, llm, cfg.rag).answer(
        "why?", history=[("What is alpha?", "A letter.")], condense=True, top_k=1
    )

    assert answer.standalone_question == "standalone query about alpha"
    assert (
        "Conversation so far" in llm.prompts[1] and "Question: why?" in llm.prompts[1]
    )


def test_chat_loop_handles_questions_commands_and_errors(retriever):
    retr, cfg = retriever
    llm = ScriptedLlm(["First [1].", "Second [1]."])
    script = iter(["", "/help", "/top 2", "/mode dense", "/bogus", "hello", "/sources", "/ctx",
                   "/reset", "again", "/exit"])  # fmt: skip
    output: list[str] = []

    code = chat_loop(
        RagEngine(retr, llm, cfg.rag), lambda _: next(script), output.append
    )

    text = "\n".join(output)
    assert code == 0 and len(llm.prompts) == 2
    assert "top_k = 2" in text and "mode = dense" in text and "unknown command" in text
    assert "Conversation so far" not in llm.prompts[1]  # /reset cleared the history
    assert "First [1]." in text and "conversation cleared" in text


def test_chat_loop_survives_llm_errors_and_eof(retriever):
    retr, cfg = retriever

    class Broken:
        model = "broken"

        def complete(self, prompt):
            raise RagError("model unavailable")

    lines = iter(["question"])

    def read(_):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    output: list[str] = []
    assert chat_loop(RagEngine(retr, Broken(), cfg.rag), read, output.append) == 0
    assert any("model unavailable" in line for line in output)


def test_format_answer_lists_sources_in_the_question_language():
    contexts = [chunk(1, text="Alpha beta gamma delta epsilon zeta eta theta.", page=4)]
    citations, _ = make_citations("Ответ [1].", "", contexts, "Вопрос?", 100)
    from source.rag.models import Answer

    answer = Answer("Вопрос?", "Вопрос?", "Ответ [1].", tuple(citations), tuple(contexts),
                    model="m", mode="dense", timings={"retrieve_s": 0.1, "llm_s": 1.0})  # fmt: skip

    text = format_answer(answer)

    assert "Источники:" in text and "стр. 4" in text and "(фрагмент подобран)" in text
    assert "http://x" in text and "[search 0.1s (dense), LLM 1.0s, m]" in text


def test_opencode_llm_translates_failures(monkeypatch):
    from source.evaluation import opencode

    def failing(*args, **kwargs):
        raise opencode.OpencodeError("quota", retryable=False)

    monkeypatch.setattr(opencode, "ask", failing)

    with pytest.raises(RagError, match="quota"):
        OpencodeLlm("m").complete("prompt")


def test_opencode_llm_attaches_very_long_prompts(monkeypatch):
    from source.evaluation import opencode

    seen = {}

    def fake(message, attachment, **kwargs):
        seen["message"], seen["attachment"] = message, attachment
        return "ok"

    monkeypatch.setattr(opencode, "ask", fake)

    assert OpencodeLlm("m").complete("x" * 70_000) == "ok"
    assert seen["attachment"] is not None and len(seen["message"]) < 200
    assert seen["attachment"].read_text() == "x" * 70_000


# ---- MCP -----------------------------------------------------------------


def call(server, name, **arguments):
    result = asyncio.run(server.call_tool(name, arguments))
    return result.structured_content


def test_mcp_server_exposes_four_documented_tools(retriever):
    retr, _ = retriever
    server = build_server(retr, warm_up=False)

    tools = {t.name: t for t in asyncio.run(server.list_tools())}

    assert set(tools) == {"search_papers", "get_chunk_context", "list_documents",
                          "get_document_outline"}  # fmt: skip
    assert "query" in tools["search_papers"].input_schema["required"]
    assert all(t.description for t in tools.values())


def test_mcp_tools_return_structured_results(retriever):
    retr, _ = retriever
    server = build_server(retr, warm_up=False)

    found = call(server, "search_papers", query="anything", top_k=3, max_chars=40)
    chunk_id = found["results"][0]["chunk_id"]
    context = call(server, "get_chunk_context", chunk_id=chunk_id, before=0, after=1)
    docs = call(server, "list_documents", query="doc0")
    outline = call(server, "get_document_outline", doc_key="doc0")

    assert found["count"] == 3 and len(found["results"][0]["text"]) <= 40 + len(
        " [...]"
    )
    assert {"title", "section", "citation", "url", "score"} <= set(found["results"][0])
    assert 1 <= len(context["chunks"]) <= 2
    assert docs["documents"][0]["doc_key"] == "doc0" and outline["sections"]


def test_mcp_tool_errors_carry_a_readable_message(retriever):
    from mcp.server.mcpserver.exceptions import ToolError

    retr, _ = retriever
    server = build_server(retr, warm_up=False)

    with pytest.raises(ToolError, match="top_k must be in 1..20"):
        asyncio.run(server.call_tool("search_papers", {"query": "q", "top_k": 99}))
    with pytest.raises(ToolError, match="unknown document"):
        asyncio.run(server.call_tool("get_document_outline", {"doc_key": "zzz"}))
    with pytest.raises(ToolError, match="mode must be one of"):
        asyncio.run(server.call_tool("search_papers", {"query": "q", "mode": "magic"}))


def test_cli_parses_the_new_commands():
    parser = build_parser_for_tests()

    ask = parser.parse_args(["ask", "what", "is", "x", "--top-k", "7", "--json"])
    mcp = parser.parse_args(["mcp", "--transport", "streamable-http", "--port", "9000"])

    assert ask.question == ["what", "is", "x"] and ask.top_k == 7 and ask.json
    assert mcp.transport == "streamable-http" and mcp.port == 9000
