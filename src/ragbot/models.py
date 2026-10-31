"""Thin wrappers around the ONNX embedding and cross-encoder models (via fastembed)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache

import numpy as np
import onnxruntime
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from ragbot.config import Settings

logger = logging.getLogger(__name__)

_GPU_PROVIDERS = ("DmlExecutionProvider", "CUDAExecutionProvider")


class Embedder:
    """Queries always run on the CPU, leaving GPU memory for the LLM while serving.

    Bulk document embedding loads a second session on the GPU, only when ingestion needs one.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._prefix = settings.embedding_query_prefix
        self._batch_size = settings.embedding_batch_size
        self._model = self._load(["CPUExecutionProvider"])
        self._passage_model: TextEmbedding | None = None
        self.embed_query = lru_cache(maxsize=2048)(self._embed_query)

    def embed_passages(
        self, texts: list[str], on_progress: Callable[[int], None] | None = None
    ) -> np.ndarray:
        if self._passage_model is None:
            gpu = _available_gpu_providers() if self._settings.embedding_use_gpu else []
            self._passage_model = self._load([*gpu, "CPUExecutionProvider"]) if gpu else self._model
        vectors: list[np.ndarray] = []
        for vector in self._passage_model.embed(texts, batch_size=self._batch_size):
            vectors.append(vector)
            if on_progress and len(vectors) % self._batch_size == 0:
                on_progress(len(vectors))
        return _normalize(np.vstack(vectors))

    def embed_sentences(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.vstack(list(self._model.embed(texts, batch_size=len(texts)))))

    def _embed_query(self, query: str) -> np.ndarray:
        return _normalize(next(iter(self._model.embed([f"{self._prefix}{query}"]))))

    def _load(self, providers: list[str]) -> TextEmbedding:
        return TextEmbedding(
            self._settings.embedding_model,
            cache_dir=str(self._settings.models_dir),
            threads=self._settings.embedding_threads,
            providers=providers,
        )


class Reranker:
    """Runs on the CPU by default, over short focused excerpts, which gives steady latency.

    GPU use is optional. Sharing the GPU with the LLM was measured to stall for tens of seconds under
    memory pressure, so if it is enabled and fails, scoring continues on the CPU.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._batch_size = settings.rerank_batch_size
        gpu = _available_gpu_providers() if settings.rerank_use_gpu else []
        self._on_gpu = bool(gpu)
        self._model = self._load([*gpu, "CPUExecutionProvider"])

    def score(self, query: str, passages: list[str]) -> list[float]:
        try:
            return self._run(query, passages)
        except Exception:
            if not self._on_gpu:
                raise
            logger.warning("GPU reranking failed, switching to the CPU", exc_info=True)
            self._on_gpu = False
            self._model = self._load(["CPUExecutionProvider"])
            return self._run(query, passages)

    def _run(self, query: str, passages: list[str]) -> list[float]:
        scores = self._model.rerank(query, passages, batch_size=self._batch_size)
        return [float(score) for score in scores]

    def _load(self, providers: list[str]) -> TextCrossEncoder:
        return TextCrossEncoder(
            self._settings.rerank_model,
            cache_dir=str(self._settings.models_dir),
            threads=self._settings.rerank_threads,
            providers=providers,
        )


def _available_gpu_providers() -> list[str]:
    available = onnxruntime.get_available_providers()
    return [provider for provider in _GPU_PROVIDERS if provider in available]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norm, 1e-12, None)
