"""Ingestion orchestration: PDF -> extract/OCR -> clean -> chunk -> embed -> index.

Idempotent: a document whose content hash and pipeline fingerprint are unchanged is skipped.
A document is marked ready only after all stores are written, so a crash causes a clean redo.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ragbot.config import Settings
from ragbot.domain import Chunk, DocumentRecord, ExtractedDocument
from ragbot.ingestion.chunking import TokenCounter, build_chunks
from ragbot.ingestion.extract import load_or_extract
from ragbot.models import Embedder
from ragbot.storage.catalog import Catalog
from ragbot.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestProgress:
    filename: str
    stage: str  # queued | extract | chunk | embed | index | ready | skipped | failed
    detail: str = ""
    completed: int = 0
    total: int = 0


ProgressCallback = Callable[[IngestProgress], None]


@dataclass(slots=True)
class IngestSummary:
    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        catalog: Catalog,
        store: VectorStore,
        embedder: Embedder,
        counter: TokenCounter | None = None,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._store = store
        self._embedder = embedder
        self._counter = counter or TokenCounter(settings.embedding_tokenizer)

    def ingest_directory(
        self,
        directory: Path | None = None,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IngestSummary:
        directory = directory or self._settings.pdf_dir
        return self.ingest(sorted(directory.glob("*.pdf")), force, progress)

    def ingest(
        self, paths: list[Path], force: bool = False, progress: ProgressCallback | None = None
    ) -> IngestSummary:
        report = progress or (lambda _: None)
        summary = IngestSummary()

        pending: dict[str, tuple[Path, str]] = {}
        for path in paths:
            sha256 = _hash_file(path)
            doc_id = sha256[:16]
            existing = self._catalog.get_document(doc_id)
            if (
                not force
                and existing
                and existing.status == "ready"
                and existing.fingerprint == self._settings.index_fingerprint
            ):
                summary.skipped.append(path.name)
                report(IngestProgress(path.name, "skipped", "already indexed"))
                continue
            pending[doc_id] = (path, sha256)
            report(IngestProgress(path.name, "queued"))

        if not pending:
            return summary

        workers = min(self._settings.ingest_workers, len(pending))
        extracted: dict[str, ExtractedDocument] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for doc_id, (path, sha256) in pending.items():
                report(IngestProgress(path.name, "extract", "text extraction and OCR"))
                futures[pool.submit(load_or_extract, path, sha256, self._settings)] = doc_id
            for future in as_completed(futures):
                doc_id = futures[future]
                path, sha256 = pending[doc_id]
                try:
                    extracted[doc_id] = future.result()
                except Exception as error:
                    self._handle_failure(path.name, doc_id, sha256, error, summary, report)

        # Embedding starts only once OCR workers exit: the two would oversubscribe the CPU.
        # It stays in this process because the model and stores cannot be shared across workers.
        for doc_id, document in extracted.items():
            path, sha256 = pending[doc_id]
            try:
                self._index(document, doc_id, sha256, report)
                summary.ingested.append(path.name)
            except Exception as error:
                self._handle_failure(path.name, doc_id, sha256, error, summary, report)
        return summary

    def _handle_failure(
        self,
        filename: str,
        doc_id: str,
        sha256: str,
        error: Exception,
        summary: IngestSummary,
        report: ProgressCallback,
    ) -> None:
        logger.error("ingestion failed for %s: %s", filename, error, exc_info=error)
        summary.failed[filename] = str(error)
        self._record_failure(filename, doc_id, sha256, str(error))
        report(IngestProgress(filename, "failed", str(error)))

    def delete(self, doc_id: str) -> bool:
        if self._catalog.get_document(doc_id) is None:
            return False
        self._store.delete(self._catalog.delete_document(doc_id))
        return True

    def _index(
        self, document: ExtractedDocument, doc_id: str, sha256: str, report: ProgressCallback
    ) -> None:
        settings = self._settings
        filename = document.filename
        self.delete(doc_id)  # clears partial or outdated data from an earlier run

        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            title=document.title,
            sha256=sha256,
            page_count=document.page_count,
            ocr_pages=document.ocr_pages,
            image_ocr_blocks=document.image_ocr_blocks,
            skipped_pages=document.skipped_pages,
            removed_boilerplate_blocks=document.removed_boilerplate_blocks,
            language=document.language,
            fingerprint=settings.index_fingerprint,
            extract_seconds=round(document.extract_seconds, 2),
        )
        self._catalog.upsert_document(record)

        started = time.perf_counter()
        report(IngestProgress(filename, "chunk", f"{document.page_count} pages"))
        chunks = build_chunks(
            document,
            doc_id,
            self._counter,
            settings.chunk_tokens,
            settings.chunk_overlap_ratio,
            settings.chunk_min_tokens,
            settings.index_fingerprint,
        )
        if not chunks:
            raise ValueError("no text could be extracted from this document")
        record.chunk_seconds = round(time.perf_counter() - started, 2)
        record.chunk_count = len(chunks)

        started = time.perf_counter()
        total = len(chunks)
        report(IngestProgress(filename, "embed", f"{total} chunks", 0, total))
        vectors = self._embedder.embed_passages(
            [embedding_text(document.title, chunk) for chunk in chunks],
            on_progress=lambda done: report(IngestProgress(filename, "embed", "", done, total)),
        )
        record.embed_seconds = round(time.perf_counter() - started, 2)

        started = time.perf_counter()
        report(IngestProgress(filename, "index", "writing HNSW and BM25 indexes"))
        self._catalog.add_chunks(document.title, chunks)
        self._store.upsert([chunk.chunk_id for chunk in chunks], vectors, doc_id)
        record.index_seconds = round(time.perf_counter() - started, 2)

        record.status = "ready"
        record.ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
        self._catalog.upsert_document(record)
        report(IngestProgress(filename, "ready", f"{total} chunks", total, total))

    def _record_failure(self, filename: str, doc_id: str, sha256: str, error: str) -> None:
        self._catalog.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                filename=filename,
                title=filename,
                sha256=sha256,
                page_count=0,
                status="failed",
                fingerprint=self._settings.index_fingerprint,
                error=error,
            )
        )


def embedding_text(title: str, chunk: Chunk) -> str:
    """The document and section title disambiguate passages that read alike out of context."""
    header = f"{title}: {chunk.section}" if chunk.section else title
    return f"{header}\n{chunk.text}"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()
