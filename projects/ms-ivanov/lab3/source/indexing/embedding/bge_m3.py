"""BGE-M3: dense (CLS) and sparse (lexical weights) vectors from one forward pass.

Re-implements the two heads of the reference ``FlagEmbedding`` model on top of
``transformers`` so that it runs on Apple's MPS backend without extra
dependencies: dense = normalised ``[CLS]`` state, sparse = ``relu(Linear(h))``
per token, max-pooled per vocabulary id with special tokens removed.
"""

import logging
from collections.abc import Callable, Sequence

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from source.indexing.config import EmbeddingConfig
from source.indexing.embedding.base import EmbeddedText, plan_batches

logger = logging.getLogger(__name__)


def resolve_device(preference: str) -> torch.device:
    """``auto`` prefers MPS, then CUDA, then CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BgeM3Embedder:
    def __init__(self, config: EmbeddingConfig):
        self._config = config
        self._device = resolve_device(config.device)
        self._dtype = self._resolve_dtype(config.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(config.model)
        self._model = (
            AutoModel.from_pretrained(config.model, dtype=self._dtype)
            .to(self._device)
            .eval()
        )
        self.dense_dim = int(self._model.config.hidden_size)
        self._sparse = self._load_sparse_head(config.model)
        self._special = {
            self._tokenizer.cls_token_id,
            self._tokenizer.eos_token_id,
            self._tokenizer.pad_token_id,
            self._tokenizer.unk_token_id,
        }
        logger.info("bge-m3 on %s (%s)", self._device, self._dtype)

    def _resolve_dtype(self, name: str) -> torch.dtype:
        if self._device.type == "cpu" and name != "float32":
            logger.warning("%s is slow/unsupported on CPU, using float32", name)
            return torch.float32
        return getattr(torch, name)

    def _load_sparse_head(self, model: str) -> torch.nn.Linear:
        head = torch.nn.Linear(self.dense_dim, 1)
        weights = torch.load(
            hf_hub_download(model, "sparse_linear.pt"), map_location="cpu"
        )
        head.load_state_dict(weights)
        return head.to(self._device, dtype=self._dtype).eval()

    @torch.inference_mode()
    def embed(
        self, texts: Sequence[str], on_batch: Callable[[int, int], None] | None = None
    ) -> list[EmbeddedText]:
        if not texts:
            return []
        cfg = self._config
        ids = self._tokenizer(
            list(texts),
            truncation=True,
            max_length=cfg.max_length,
            return_attention_mask=False,
        )["input_ids"]
        truncated = sum(1 for text_ids in ids if len(text_ids) >= cfg.max_length)
        if truncated:
            logger.warning(
                "%d text(s) truncated to %d tokens", truncated, cfg.max_length
            )
        results: list[EmbeddedText | None] = [None] * len(texts)
        done = 0
        for batch in plan_batches([len(x) for x in ids], cfg.batch_tokens):
            for index, vectors in zip(
                batch, self._forward([ids[i] for i in batch]), strict=True
            ):
                results[index] = vectors
            done += len(batch)
            if on_batch:
                on_batch(done, len(texts))
        return results  # type: ignore[return-value]

    def _forward(self, id_lists: list[list[int]]) -> list[EmbeddedText]:
        padded = self._tokenizer.pad({"input_ids": id_lists}, return_tensors="pt")
        input_ids = padded["input_ids"].to(self._device)
        mask = padded["attention_mask"].to(self._device)
        hidden = self._model(input_ids=input_ids, attention_mask=mask).last_hidden_state
        mode = self._config.mode
        dense = None
        if mode in ("dense", "hybrid"):
            dense = (
                torch.nn.functional.normalize(hidden[:, 0].float(), dim=-1)
                .cpu()
                .tolist()
            )
        weights = None
        if mode in ("sparse", "hybrid"):
            weights = torch.relu(self._sparse(hidden)).squeeze(-1).float().cpu()
        ids_cpu = input_ids.cpu()
        if self._device.type == "mps":
            torch.mps.empty_cache()
        return [
            EmbeddedText(
                dense=dense[row] if dense is not None else None,
                sparse=(
                    self._sparse_dict(ids_cpu[row], weights[row])
                    if weights is not None
                    else None
                ),
                n_tokens=int(mask[row].sum()),
            )
            for row in range(len(id_lists))
        ]

    def _sparse_dict(
        self, token_ids: torch.Tensor, weights: torch.Tensor
    ) -> dict[int, float]:
        pooled: dict[int, float] = {}
        for token, weight in zip(token_ids.tolist(), weights.tolist(), strict=True):
            if token in self._special or weight <= 0:
                continue
            if weight > pooled.get(token, 0.0):
                pooled[token] = weight
        return pooled
