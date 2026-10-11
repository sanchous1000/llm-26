"""The question-answering pipeline: retrieve -> prompt -> LLM -> cited answer."""

import time
from collections.abc import Sequence

from source.indexing.config import RagConfig
from source.rag.errors import RagError
from source.rag.llm import Llm
from source.rag.models import Answer
from source.rag.prompt import build_prompt, make_citations, split_sources_block
from source.rag.retriever import Retriever, resolve_mode

CONDENSE_PROMPT = """\
Rewrite the last user question as a standalone search query that can be understood
without the conversation. Keep the language of the question. Reply with the query
only, no explanations.

{history}

Last question: {question}

Standalone query:"""

History = Sequence[tuple[str, str]]


class RagEngine:
    def __init__(self, retriever: Retriever, llm: Llm, config: RagConfig):
        self._retriever = retriever
        self._llm = llm
        self._config = config

    def retrieve(self, question: str, *, top_k=None, mode=None, doc_keys=None):
        return self._retriever.search(
            question, top_k=top_k, mode=mode, doc_keys=doc_keys
        )

    def answer(
        self,
        question: str,
        *,
        history: History = (),
        top_k: int | None = None,
        mode: str | None = None,
        doc_keys: Sequence[str] | None = None,
        condense: bool = False,
    ) -> Answer:
        """Answer ``question`` from the retrieved passages, citing them."""
        cfg = self._config
        recent = tuple(history)[-cfg.history_turns :] if cfg.history_turns else ()
        started = time.monotonic()
        standalone = (
            self._condense(question, recent) if condense and recent else question
        )
        condensed_s = time.monotonic() - started
        contexts = self._retriever.search(
            standalone, top_k=top_k, mode=mode, doc_keys=doc_keys
        )
        retrieved_s = time.monotonic() - started - condensed_s
        used_mode = resolve_mode(standalone, mode or cfg.mode, self._built_mode)
        if not contexts:
            return Answer(
                question,
                standalone,
                _NOTHING_FOUND,
                (),
                (),
                model=self._llm.model,
                mode=used_mode,
                timings={"retrieve_s": retrieved_s},
            )
        bundle = build_prompt(
            question,
            contexts,
            history=recent,
            max_context_tokens=cfg.max_context_tokens,
        )
        before_llm = time.monotonic()
        raw = self._llm.complete(bundle.text)
        llm_s = time.monotonic() - before_llm
        text, sources = split_sources_block(raw)
        if not text:
            raise RagError("the language model returned an empty answer")
        citations, unknown = make_citations(
            text, sources, bundle.contexts, question, cfg.snippet_chars
        )
        return Answer(
            question,
            standalone,
            text,
            tuple(citations),
            bundle.contexts,
            unknown_citations=tuple(unknown),
            model=self._llm.model,
            mode=used_mode,
            timings={
                "condense_s": condensed_s,
                "retrieve_s": retrieved_s,
                "llm_s": llm_s,
            },
        )

    @property
    def _built_mode(self) -> str:
        return self._retriever.built_mode

    def _condense(self, question: str, history: History) -> str:
        lines = "\n".join(f"User: {q}\nAssistant: {a[:500]}" for q, a in history)
        rewritten = self._llm.complete(
            CONDENSE_PROMPT.format(history=lines, question=question)
        )
        return rewritten.strip().strip('"') or question


_NOTHING_FOUND = "No passages were found for this question in the indexed papers."
