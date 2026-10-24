from fastembed import SparseTextEmbedding, TextEmbedding


class EmbeddingsService:
    """Dense-эмбеддинги """

    def __init__(self, model: TextEmbedding):
        self.model = model

    @classmethod
    def from_name(cls, model_name: str) -> "EmbeddingsService":
        return cls(TextEmbedding(model_name=model_name))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self.model.embed(texts, batch_size=32)]

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def token_count(self, text: str) -> int:
        """Подсчёт числа токенов"""
        return self.model.token_count(text)


class SparseEmbeddingsService:
    """Sparse (BM25) через готовую модель fastembed ("Qdrant/bm25")"""

    def __init__(self, model: SparseTextEmbedding):
        self.model = model

    @classmethod
    def from_name(cls, model_name: str = "Qdrant/bm25") -> "SparseEmbeddingsService":
        return cls(SparseTextEmbedding(model_name=model_name))

    def embed_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [(e.indices.tolist(), e.values.tolist()) for e in self.model.embed(texts, batch_size=32)]

    def embed(self, text: str) -> tuple[list[int], list[float]]:
        return self.embed_batch([text])[0]