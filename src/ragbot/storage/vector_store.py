"""Vector index on ChromaDB (Apache-2.0): embedded, persistent, HNSW with cosine distance."""

from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

_COLLECTION = "chunks"
_WRITE_BATCH = 1000


class VectorStore:
    def __init__(
        self, path: Path, max_neighbors: int, ef_construction: int, ef_search: int
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            _COLLECTION,
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "max_neighbors": max_neighbors,
                    "ef_construction": ef_construction,
                    "ef_search": ef_search,
                }
            },
        )

    def upsert(self, chunk_ids: list[str], vectors: np.ndarray, doc_id: str) -> None:
        for start in range(0, len(chunk_ids), _WRITE_BATCH):
            ids = chunk_ids[start : start + _WRITE_BATCH]
            self._collection.upsert(
                ids=ids,
                embeddings=vectors[start : start + _WRITE_BATCH],
                metadatas=[{"doc_id": doc_id}] * len(ids),
            )

    def delete(self, chunk_ids: list[str]) -> None:
        for start in range(0, len(chunk_ids), _WRITE_BATCH):
            self._collection.delete(ids=chunk_ids[start : start + _WRITE_BATCH])

    def search(
        self, vector: np.ndarray, limit: int, doc_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Returns (chunk_id, cosine similarity), most similar first."""
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        result = self._collection.query(
            query_embeddings=[vector], n_results=limit, where=where, include=["distances"]
        )
        return [
            (chunk_id, 1.0 - float(distance))
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]

    def count(self) -> int:
        return self._collection.count()
