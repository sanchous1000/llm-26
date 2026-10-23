import torch
from sentence_transformers import SentenceTransformer, models

from doc_rag.config import EmbeddingsConfig


class EmbeddingsService:
    """Dense-эмбеддинги из локальной модели (bge-m3) + её токенизатор для сплиттера.
    """

    def __init__(self, config: EmbeddingsConfig):
        path = config.resolved_model_path()
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"Модель эмбеддингов не найдена: {path} (embeddings.model_path в config.yml)")

        self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.device = "cuda" if config.device == "auto" and torch.cuda.is_available() else (
            "cpu" if config.device == "auto" else config.device
        )

        transformer = models.Transformer(str(path), max_seq_length=config.max_seq_length)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="cls")
        self.model = SentenceTransformer(modules=[transformer, pooling], device=self.device)
        if config.fp16 and self.device.startswith("cuda"):
            self.model.half()

        self.tokenizer = self.model.tokenizer
        self.max_seq_length: int = config.max_seq_length
        self.dim: int = self.model.get_sentence_embedding_dimension()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]
