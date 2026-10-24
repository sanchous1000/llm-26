import hashlib
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from book_rag_backend.services.parser import Document

MIN_TOKENS = 5  # более короткие чанки (например, одинокий заголовок) отбрасываются
HEADING_RE = re.compile(r"^#{1,3}\s+(.*)$", re.M)  # заголовки, которые проставил parser.py

SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


def count_tokens(text: str) -> int:
    return len(text.split())


class SplitterService:
    def __init__(
            self,
            chunk_size: int,  # Размер чанка в токенах
            chunk_overlap: int,  # Перекрытие для сохранения контекста
            embeddings,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=embeddings.token_count,
            is_separator_regex=False,
            separators=SEPARATORS,
        )

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        chunks = self.splitter.split_text(text)
        return chunks


def _heading_before(chunk_text: str, page_text: str, headings: list[tuple[int, str]]) -> str:
    """Последний заголовок страницы, который встретился раньше начала этого чанка."""
    pos = page_text.find(chunk_text[:40])
    if pos == -1:
        pos = 0
    title = ""
    for hpos, h in headings:
        if hpos <= pos:
            title = h
        else:
            break
    return title


def chunk_documents(docs: list[Document], splitter: SplitterService, include_headings: bool = False) -> list[dict]:
    chunks = []
    for doc in docs:
        idx = 0
        for page in doc.pages:
            headings = [(m.start(), m.group(1).strip()) for m in HEADING_RE.finditer(page.markdown)]
            for text in splitter.split(page.markdown):
                text = text.strip()
                n = count_tokens(text)
                if n < MIN_TOKENS:
                    continue
                heading = _heading_before(text, page.markdown, headings) if include_headings else ""
                embed_text = f"{heading}\n{text}" if heading else text
                cid = hashlib.sha1(f"{doc.source}|{page.number}|{idx}|{text}".encode()).hexdigest()[:16]
                chunks.append({
                    "id": cid, "source": doc.source, "date": doc.date, "page": page.number,
                    "chunk_index": idx, "heading": heading, "text": text,
                    "embed_text": embed_text, "n_tokens": n,
                })
                idx += 1
    return chunks