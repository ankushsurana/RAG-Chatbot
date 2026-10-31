"""Application settings. Every tunable lives here and can be overridden via RAGBOT_* env vars."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Bump when extraction or cleaning logic changes, so cached output and indexes are rebuilt.
PIPELINE_VERSION = "1"
# Bump when post-cache text repair or chunking logic changes: reindexes without redoing OCR.
INDEX_VERSION = "2"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGBOT_", env_file=".env", extra="ignore")

    # Storage
    data_dir: Path = Path("data")

    # Extraction and OCR
    ocr_enabled: bool = True
    ocr_dpi: int = 200
    ocr_threads: int = 2
    ocr_min_confidence: float = 0.6
    ocr_page_min_native_chars: int = 40
    ocr_page_min_image_coverage: float = 0.3
    ocr_image_min_coverage: float = 0.08
    ingest_workers: int = max(1, min(4, (os.cpu_count() or 2) // 4))

    # Chunking (token counts use the embedding model's tokenizer)
    chunk_tokens: int = 500
    chunk_overlap_ratio: float = 0.15
    chunk_min_tokens: int = 40

    # Embedding
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_tokenizer: str = "BAAI/bge-base-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = 16
    embedding_threads: int | None = None
    embedding_use_gpu: bool = True

    # Vector index (HNSW)
    hnsw_max_neighbors: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128

    # Retrieval
    dense_candidates: int = 40
    lexical_candidates: int = 40
    rrf_k: int = 60
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 16
    rerank_threads: int | None = None
    rerank_use_gpu: bool = False
    rerank_focus_chars: int = 700
    rerank_batch_size: int = 24
    retrieval_top_k: int = 8
    # A question whose best reranker logit falls below this is refused without calling the LLM.
    min_relevance_logit: float = -6.0

    # Generation
    llm_provider: Literal["ollama", "openai"] = "ollama"
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:3b"
    llm_api_key: str | None = None
    llm_num_ctx: int = 4096
    llm_max_tokens: int = 160
    llm_timeout_seconds: float = 60.0
    llm_seed: int = 42
    context_max_chunks: int = 4
    context_max_tokens: int = 1800
    context_focus_chars: int = 1500
    # Passages scoring this far below the best one are noise that confuses a small model.
    context_score_margin: float = 4.0
    # When the LLM declines despite a strong match, ask again using only the best passages.
    retry_min_logit: float = 0.0
    retry_max_chunks: int = 1
    retry_focus_chars: int = 700

    # Answer verification
    support_min_overlap: float = 0.5
    support_min_similarity: float = 0.72

    # API
    chat_db_file: Path | None = None
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    public_questions_per_minute: int = 12

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def catalog_path(self) -> Path:
        return self.index_dir / "catalog.sqlite"

    @property
    def chat_db_path(self) -> Path:
        return self.chat_db_file or self.data_dir / "chat" / "chat.sqlite"

    @property
    def vector_dir(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def reports_dir(self) -> Path:
        return Path("reports")

    @property
    def extraction_fingerprint(self) -> str:
        """Identifies cached extraction output: changes when extraction settings change."""
        return _fingerprint(
            PIPELINE_VERSION,
            self.ocr_enabled,
            self.ocr_dpi,
            self.ocr_min_confidence,
            self.ocr_page_min_native_chars,
            self.ocr_page_min_image_coverage,
            self.ocr_image_min_coverage,
        )

    @property
    def index_fingerprint(self) -> str:
        """Identifies a document's indexed form: changes force re-chunking and re-embedding."""
        return _fingerprint(
            INDEX_VERSION,
            self.extraction_fingerprint,
            self.chunk_tokens,
            self.chunk_overlap_ratio,
            self.chunk_min_tokens,
            self.embedding_model,
            self.embedding_tokenizer,
        )

    def ensure_dirs(self) -> None:
        for path in (self.pdf_dir, self.raw_dir, self.cache_dir, self.index_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


def _fingerprint(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:16]


@lru_cache
def get_settings() -> Settings:
    return Settings()
