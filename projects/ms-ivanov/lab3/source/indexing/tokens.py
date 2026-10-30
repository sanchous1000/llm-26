"""Token counting with the tokenizer of the embedding model."""

from typing import Protocol


class TokenCounter(Protocol):
    """Counts tokens and maps them back to character offsets."""

    def count(self, text: str) -> int: ...

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of every token (special tokens excluded)."""
        ...


class HfTokenCounter:
    """Hugging Face fast tokenizer (sentencepiece for bge-m3)."""

    def __init__(self, name: str):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._tokenizer.model_max_length = 10**9  # long inputs are expected here

    def count(self, text: str) -> int:
        if not text:
            return 0
        encoded = self._tokenizer(
            text, add_special_tokens=False, return_attention_mask=False
        )
        return len(encoded["input_ids"])

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        return [tuple(pair) for pair in encoded["offset_mapping"]]
