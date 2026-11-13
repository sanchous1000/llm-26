"""Dense-эмбеддинги через OpenAI-совместимый API Ollama (/v1/embeddings) с кэшем на диске."""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np
from openai import OpenAI


class DenseEncoder:
    def __init__(self, model: str, base_url: str, batch_size: int = 16, cache_path: Path | None = None):
        self.model, self.batch_size = model, batch_size
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=600)
        self.db = None
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(cache_path))
            self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}\x00{text}".encode()).hexdigest()

    def _embed_api(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)], dtype=np.float32)
        return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

    def encode(self, texts: list[str], progress: bool = False) -> tuple[np.ndarray, dict]:
        keys = [self._key(t) for t in texts]
        cached: dict[str, np.ndarray] = {}
        if self.db is not None:
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})"
                for k, blob in self.db.execute(q, part):
                    cached[k] = np.frombuffer(blob, dtype=np.float32)
        todo = [i for i, k in enumerate(keys) if k not in cached]
        t0 = time.perf_counter()
        for s in range(0, len(todo), self.batch_size):
            idx = todo[s:s + self.batch_size]
            vecs = self._embed_api([texts[i] for i in idx])
            for i, v in zip(idx, vecs):
                cached[keys[i]] = v
                if self.db is not None:
                    self.db.execute("INSERT OR REPLACE INTO emb VALUES (?, ?)", (keys[i], v.tobytes()))
            if self.db is not None:
                self.db.commit()
            if progress:
                print(f"\r  dense: {min(s + self.batch_size, len(todo))}/{len(todo)}", end="", flush=True)
        if progress and todo:
            print()
        elapsed = time.perf_counter() - t0
        stats = {"embedded": len(todo), "from_cache": len(texts) - len(todo),
                 "embed_seconds": round(elapsed, 2),
                 "texts_per_second": round(len(todo) / elapsed, 2) if todo and elapsed else None}
        return np.stack([cached[k] for k in keys]) if keys else np.zeros((0, 0), np.float32), stats

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed_api([text])[0]
