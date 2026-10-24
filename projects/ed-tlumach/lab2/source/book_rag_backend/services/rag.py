from book_rag_backend.services.embeddings import EmbeddingsService, SparseEmbeddingsService
from book_rag_backend.services.llm import LLMService
from book_rag_backend.services.qdrant import QdrantService

SYSTEM_PROMPT = (
    "Ты — ассистент, отвечающий на вопросы по корпусу научных статей. "
    "Используй только предоставленные источники для ответа, ссылайся на них в квадратных "
    "скобках, например [2]. Если информации недостаточно, честно скажи об этом. "
    "Отвечай на том же языке, на котором задан вопрос."
)


class RAGService:
    def __init__(self, qdrant: QdrantService, vectorizer: str, embeddings: EmbeddingsService | None,
                 sparse: SparseEmbeddingsService | None, llm: LLMService | None,
                 top_k: int = 10, ef_search: int = 64, max_context_chars: int = 6000):
        self.qdrant = qdrant
        self.vectorizer = vectorizer
        self.embeddings = embeddings
        self.sparse = sparse
        self.llm = llm
        self.top_k = top_k
        self.ef_search = ef_search
        self.max_context_chars = max_context_chars

    async def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        dense_vector = self.embeddings.embed(question) if self.vectorizer in ("dense", "hybrid") else None
        sparse_vector = self.sparse.embed(question) if self.vectorizer in ("sparse", "hybrid") else None
        return await self.qdrant.search(dense_vector, sparse_vector, self.vectorizer,
                                        k or self.top_k, self.ef_search)

    async def search(self, messages: list[dict]) -> dict:
        question = messages[-1]["content"]
        hits = await self.retrieve(question)

        context_parts, citations, used = [], [], 0
        for i, h in enumerate(hits, 1):
            snippet = h["content"][:self.max_context_chars // max(1, self.top_k)]
            if used + len(snippet) > self.max_context_chars:
                break
            used += len(snippet)
            context_parts.append(f"[{i}] (источник: {h['metadata']['source']}, стр. {h['metadata']['page']}): {snippet}")
            citations.append({
                "n": i, "source": h["metadata"]["source"], "page": h["metadata"]["page"],
                "heading": h["metadata"].get("heading", ""),
                "score": round(h["score"], 4), "snippet": snippet[:200],
            })
        context = "\n\n".join(context_parts)

        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        llm_messages += [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]
        llm_messages.append({"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"})

        answer = await self.llm.generate(llm_messages)
        return {"messages": messages + [{"role": "assistant", "content": answer}], "citations": citations}