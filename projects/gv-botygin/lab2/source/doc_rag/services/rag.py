import re
from dataclasses import dataclass, field

from doc_rag.config import SearchConfig
from doc_rag.services.embeddings import EmbeddingsService
from doc_rag.services.llm import LLMService
from doc_rag.services.qdrant import QdrantService

SNIPPET_LEN = 300
NO_CONTEXT_ANSWER = "В загруженных документах не найдено информации по этому вопросу."

SYSTEM_PROMPT = (
    "Ты - ассистент, отвечающий на вопросы по документам. "
    "Отвечай только на основе приведённых фрагментов контекста. "
    "Если в контексте нет ответа, прямо скажи, что в документах такой информации нет, и ничего не выдумывай. "
    "После каждого утверждения ставь ссылку на фрагмент в формате [1] или [1][2]. "
    "Отвечай кратко, на языке вопроса."
)


@dataclass
class Hit:
    score: float
    text: str
    source: str
    page_start: int
    page_end: int
    section: str  # путь заголовков "H1 > H2 > H3" или ""

    @property
    def pages(self) -> str:
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


@dataclass
class Citation:
    n: int  # номер фрагмента в промпте, [n] в ответе
    source: str
    page_start: int
    page_end: int
    section: str
    snippet: str
    score: float


@dataclass
class Answer:
    answer: str
    citations: list[Citation] = field(default_factory=list)  # только фрагменты, на которые сослался ответ
    retrieved: list[Citation] = field(default_factory=list)  # все найденные фрагменты


class RAGService:
    def __init__(
        self,
        qdrant: QdrantService,
        embeddings: EmbeddingsService,
        llm: LLMService | None,
        search_config: SearchConfig,
    ):
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.llm = llm
        self.defaults = search_config

    # ------------------------------------------------------------------ поиск

    def search(self, query: str, top_k: int | None = None, score_threshold: float | None = None) -> list[Hit]:
        top_k = top_k or self.defaults.top_k
        threshold = self.defaults.score_threshold if score_threshold is None else score_threshold

        vector = self.embeddings.embed(query)
        raw = self.qdrant.search(vector, top_k=top_k, score_threshold=threshold)
        return [
            Hit(
                score=r["score"],
                text=r["text"],
                source=r["source"],
                page_start=r["page_start"],
                page_end=r["page_end"],
                section=" > ".join(h for h in (r.get("h1"), r.get("h2"), r.get("h3")) if h),
            )
            for r in raw
        ]

    # ------------------------------------------------------------------ поиск + LLM

    def ask(self, query: str, top_k: int | None = None, score_threshold: float | None = None) -> Answer:
        hits = self.search(query, top_k, score_threshold)
        if not hits:
            return Answer(answer=NO_CONTEXT_ANSWER)

        answer = self.llm.generate(self.build_messages(query, hits))
        retrieved = [self._citation(n, h) for n, h in enumerate(hits, 1)]
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        return Answer(answer=answer, citations=[c for c in retrieved if c.n in cited], retrieved=retrieved)

    @staticmethod
    def build_messages(query: str, hits: list[Hit]) -> list[dict]:
        context = "\n\n".join(
            f"[{n}] (документ: {h.source}, стр. {h.pages}"
            + (f", раздел: {h.section}" if h.section else "")
            + f")\n{h.text}"
            for n, h in enumerate(hits, 1)
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"},
        ]

    @staticmethod
    def _citation(n: int, h: Hit) -> Citation:
        text = re.sub(r"\s+", " ", h.text).strip()
        snippet = text if len(text) <= SNIPPET_LEN else text[:SNIPPET_LEN].rstrip() + "..."
        return Citation(n, h.source, h.page_start, h.page_end, h.section, snippet, round(h.score, 4))
