"""Hybrid retrieval: dense HNSW + BM25, reciprocal rank fusion, then cross-encoder reranking."""

from __future__ import annotations

import re
import time

from ragbot.config import Settings
from ragbot.domain import DocumentRecord, RetrievalResult, RetrievedChunk, Timings
from ragbot.generation.focus import focus_text
from ragbot.models import Embedder, Reranker
from ragbot.storage.catalog import Catalog
from ragbot.storage.vector_store import VectorStore

_WORD = re.compile(r"[^\W\d_]{3,}")
_IDENTIFIER = re.compile(r"\d+[A-Za-z]?-\d+[A-Za-z]?")


def reciprocal_rank_fusion(rankings: list[list[str]], k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def resolve_named_documents(query: str, documents: list[DocumentRecord]) -> tuple[list[str], str]:
    """Finds documents the question names, such as "NIST SP 800-82", and removes the name from it.

    Returns (document ids, query to search with). The filter stops similar documents crowding out
    the one asked about. The name itself is then dropped, because it appears throughout that
    document and would otherwise outweigh the words that describe what the user is looking for.
    Names shared by several documents ("NIST SP") cannot identify one, but are removed along with it.
    """
    names_by_document = {doc.doc_id: _document_names(doc.filename) for doc in documents}
    owners: dict[str, set[str]] = {}
    for doc_id, names in names_by_document.items():
        for name in names:
            owners.setdefault(name, set()).add(doc_id)

    named = {
        doc_id
        for name, doc_ids in owners.items()
        if len(doc_ids) == 1 and _contains_name(query, name)
        for doc_id in doc_ids
    }
    search_query = query
    for name in sorted(
        {n for doc_id in named for n in names_by_document[doc_id]}, key=len, reverse=True
    ):
        search_query = _name_pattern(name).sub(" ", search_query)
    search_query = " ".join(search_query.split())
    return sorted(named), search_query if _WORD.search(search_query) else query


def _name_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(name)}(?:'s)?(?!\w)", re.I)


def _contains_name(query: str, name: str) -> bool:
    return bool(_name_pattern(name).search(query))


def _document_names(filename: str) -> set[str]:
    words = filename.removesuffix(".pdf").split("_")
    names = {word for word in words if _IDENTIFIER.fullmatch(word)}
    names.add(" ".join(words[:2]))
    if words[0].isupper() and len(words[0]) >= 3:
        names.add(words[0])
    return names


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        catalog: Catalog,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._store = store
        self._embedder = embedder
        self._reranker = reranker

    def retrieve(
        self, query: str, top_k: int | None = None, doc_ids: list[str] | None = None
    ) -> RetrievalResult:
        settings = self._settings
        top_k = top_k or settings.retrieval_top_k
        timings = Timings()
        named, search_query = resolve_named_documents(query, self._catalog.list_documents())
        doc_ids = doc_ids or named or None
        overall = time.perf_counter()

        started = time.perf_counter()
        vector = self._embedder.embed_query(search_query)
        timings.embed_ms = _elapsed(started)

        started = time.perf_counter()
        dense = self._store.search(vector, settings.dense_candidates, doc_ids)
        timings.dense_ms = _elapsed(started)

        started = time.perf_counter()
        lexical = self._catalog.lexical_search(search_query, settings.lexical_candidates, doc_ids)
        timings.lexical_ms = _elapsed(started)

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [[chunk_id for chunk_id, _ in dense], [chunk_id for chunk_id, _ in lexical]],
            settings.rrf_k,
        )
        ordered = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))
        shortlisted = ordered[: settings.rerank_candidates]
        loaded = self._catalog.get_chunks(shortlisted)
        chunks = self._hydrate(shortlisted, loaded, fused, dense, lexical)
        timings.fusion_ms = _elapsed(started)

        if self._reranker is not None and chunks:
            started = time.perf_counter()
            passages = [
                f"{chunk.title}: {chunk.section}\n"
                f"{focus_text(chunk.text, search_query, settings.rerank_focus_chars)}"
                for chunk in chunks
            ]
            for chunk, score in zip(
                chunks, self._reranker.score(search_query, passages), strict=True
            ):
                chunk.rerank_score = round(score, 4)
            chunks.sort(key=lambda chunk: (-(chunk.rerank_score or 0.0), chunk.chunk_id))
            timings.rerank_ms = _elapsed(started)

        timings.retrieval_ms = _elapsed(overall)
        return RetrievalResult(query=query, chunks=chunks[:top_k], timings=timings)

    @staticmethod
    def _hydrate(
        shortlisted: list[str],
        loaded: dict[str, RetrievedChunk],
        fused: dict[str, float],
        dense: list[tuple[str, float]],
        lexical: list[tuple[str, float]],
    ) -> list[RetrievedChunk]:
        dense_info = {cid: (rank, score) for rank, (cid, score) in enumerate(dense, start=1)}
        lexical_info = {cid: (rank, score) for rank, (cid, score) in enumerate(lexical, start=1)}
        chunks: list[RetrievedChunk] = []
        for chunk_id in shortlisted:
            chunk = loaded.get(chunk_id)
            if chunk is None:
                continue  # present in the vector index but deleted from the catalog
            chunk.fusion_score = round(fused[chunk_id], 5)
            if chunk_id in dense_info:
                chunk.dense_rank, score = dense_info[chunk_id]
                chunk.dense_score = round(score, 4)
            if chunk_id in lexical_info:
                chunk.lexical_rank, score = lexical_info[chunk_id]
                chunk.lexical_score = round(score, 3)
            chunks.append(chunk)
        return chunks


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
