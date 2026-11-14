"""Реализует инструменты поиска теории, получения карточек сцен и сохранения планов подготовки."""

import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..tracing import TraceRecorder


class StudyTools:
    def __init__(self, docs_dir: Path, store_dir: Path, output_dir: Path, trace=None):
        self.output_dir = output_dir
        self.trace = trace or TraceRecorder()
        self.store_dir = store_dir
        scenes = json.loads((docs_dir / "scenes.json").read_text(encoding="utf-8"))
        self.scenes = {scene["id"]: scene for scene in scenes}
        self.chunks = [
            json.loads(line)
            for line in (store_dir / "chunks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
        embedding = manifest["embedding_manifest"]
        self.model = SentenceTransformer(
            embedding["embedding"]["model"],
            revision=embedding["model_revision"],
            device="cpu",
        )
        faiss.omp_set_num_threads(1)
        self.index = faiss.read_index(str(store_dir / "index.faiss"))

    def search_theory(self, query: str) -> list[dict]:
        """Возвращает 10 ближайших фрагментов теории из FAISS-индекса lab2."""
        with self.trace.scope(
            "retrieval",
            as_type="retriever",
            input={"query": query},
            metadata={"vector_store": str(self.store_dir), "k": 10},
        ) as event:
            vector = self.model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                precision="float32",
                prompt="",
                show_progress_bar=False,
            )
            distances, labels = self.index.search(
                np.ascontiguousarray(vector, dtype=np.float32), 10
            )
            results = []
            for rank, (label, distance) in enumerate(zip(labels[0], distances[0]), 1):
                if label == -1:
                    continue
                chunk = self.chunks[int(label)]
                results.append(
                    {
                        "rank": rank,
                        "chunk_id": chunk["id"],
                        "source": chunk["metadata"]["source"],
                        "page": chunk["metadata"]["page"],
                        "squared_l2": float(distance),
                        "text": chunk["text"],
                    }
                )
            event["output"] = results
            return results

    def get_scene(self, scene_id: str) -> dict:
        """Возвращает карточку сцены из docs/scenes.json по её идентификатору."""
        return self.scenes[scene_id]

    def save_study_plan(
        self, ticket_id: str, points: list[str], operation_id: str | None = None
    ) -> dict:
        """Сохраняет билет и пункты плана подготовки в JSON-файл."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        name = uuid5(NAMESPACE_URL, operation_id).hex if operation_id else uuid4().hex
        path = self.output_dir / f"{name}.json"
        plan = {"ticket_id": ticket_id, "points": points}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
        return {"status": "saved", "path": str(path), **plan}
