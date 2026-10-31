"""The RAG pipeline as a service: retrieve, gate on relevance, generate, verify."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from ragbot.config import Settings
from ragbot.domain import Answer, RetrievalResult, RetrievedChunk, Timings
from ragbot.generation.llm import LLMClient, build_llm
from ragbot.generation.prompts import REFUSAL_TEXT, build_messages, select_context
from ragbot.generation.verification import (
    VerifiedAnswer,
    is_refusal,
    refused_answer,
    verify_answer,
)
from ragbot.ingestion.pipeline import IngestionPipeline
from ragbot.models import Embedder, Reranker
from ragbot.retrieval import HybridRetriever
from ragbot.storage.catalog import Catalog
from ragbot.storage.vector_store import VectorStore


@dataclass(slots=True)
class StreamEvent:
    type: Literal["retrieval", "token", "final"]
    payload: dict


@dataclass(slots=True)
class _GenerationOutcome:
    finish_reason: str | None = None


class RagService:
    def __init__(
        self,
        settings: Settings,
        catalog: Catalog,
        store: VectorStore,
        embedder: Embedder,
        retriever: HybridRetriever,
        llm: LLMClient,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.store = store
        self.embedder = embedder
        self.retriever = retriever
        self.llm = llm

    def retrieve(
        self, query: str, top_k: int | None = None, doc_ids: list[str] | None = None
    ) -> RetrievalResult:
        return self.retriever.retrieve(query, top_k, doc_ids)

    def stream_answer(
        self, question: str, doc_ids: list[str] | None = None
    ) -> Iterator[StreamEvent]:
        settings = self.settings
        started = time.perf_counter()

        retrieval = self.retriever.retrieve(question, doc_ids=doc_ids)
        timings = retrieval.timings
        chunks = retrieval.chunks

        sources = self._select_sources(chunks)
        yield StreamEvent(
            "retrieval",
            {"chunks": [c.model_dump() for c in chunks], "timings": timings.model_dump()},
        )

        if not sources:
            yield StreamEvent("token", {"text": REFUSAL_TEXT})
            yield self._final(question, refused_answer(), chunks, timings, started)
            return

        generation_started = time.perf_counter()
        parts: list[str] = []
        outcome = _GenerationOutcome()
        yield from self._generate(
            build_messages(question, sources, focus_chars=settings.context_focus_chars),
            parts,
            outcome,
            timings,
            started,
        )

        best = chunks[0].rerank_score
        if is_refusal("".join(parts)) and best is not None and best >= settings.retry_min_logit:
            # Small models often decline when the answer is present but surrounded by other text.
            focused = sources[: settings.retry_max_chunks]
            parts.clear()
            yield from self._generate(
                build_messages(
                    question, focused, focused=True, focus_chars=settings.retry_focus_chars
                ),
                parts,
                outcome,
                timings,
                started,
            )
        timings.generation_ms = _elapsed(generation_started)

        verification_started = time.perf_counter()
        verified = verify_answer(
            "".join(parts),
            sources,
            self.embedder.embed_sentences,
            settings.support_min_overlap,
            settings.support_min_similarity,
            truncated=outcome.finish_reason == "length",
            question=question,
        )
        timings.verification_ms = _elapsed(verification_started)
        if not verified.refused and verified.verification.groundedness == 0.0:
            # Nothing in the answer could be traced to a source, so it must not be shown as fact.
            verified = refused_answer()
        yield self._final(question, verified, chunks, timings, started)

    def _generate(
        self,
        messages: list[dict[str, str]],
        parts: list[str],
        outcome: _GenerationOutcome,
        timings: Timings,
        started: float,
    ) -> Iterator[StreamEvent]:
        """Streams the LLM reply, holding back text that may turn out to be a refusal."""
        held = ""
        released = False
        for event in self.llm.stream(messages):
            if event.text:
                parts.append(event.text)
                if released:
                    yield StreamEvent("token", {"text": event.text})
                else:
                    held += event.text
                    if not REFUSAL_TEXT.startswith(held.lstrip()[: len(REFUSAL_TEXT)]):
                        released = True
                        if timings.first_token_ms is None:
                            timings.first_token_ms = _elapsed(started)
                        yield StreamEvent("token", {"text": held})
            if event.finish_reason:
                outcome.finish_reason = event.finish_reason
                timings.prompt_tokens = event.prompt_tokens
                timings.completion_tokens = event.completion_tokens

    def answer(self, question: str, doc_ids: list[str] | None = None) -> Answer:
        for event in self.stream_answer(question, doc_ids):
            if event.type == "final":
                return Answer.model_validate(event.payload)
        raise RuntimeError("answer stream ended without a final event")

    def _select_sources(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        settings = self.settings
        if not chunks:
            return []
        best = chunks[0].rerank_score
        if best is not None and best < settings.min_relevance_logit:
            return []
        if best is not None:
            floor = best - settings.context_score_margin
            chunks = [chunk for chunk in chunks if (chunk.rerank_score or 0.0) >= floor]
        return select_context(chunks, settings.context_max_chunks, settings.context_max_tokens)

    def _final(
        self,
        question: str,
        verified: VerifiedAnswer,
        chunks: list[RetrievedChunk],
        timings: Timings,
        started: float,
    ) -> StreamEvent:
        timings.total_ms = _elapsed(started)
        answer = Answer(
            question=question,
            answer=verified.text,
            refused=verified.refused,
            citations=verified.citations,
            verification=verified.verification,
            retrieved=chunks,
            timings=timings,
            model=self.llm.model,
        )
        return StreamEvent("final", answer.model_dump())


def build_service(settings: Settings) -> tuple[RagService, IngestionPipeline]:
    """Wires up the whole application. Heavy models load once, here."""
    settings.ensure_dirs()
    catalog = Catalog(settings.catalog_path)
    store = VectorStore(
        settings.vector_dir,
        settings.hnsw_max_neighbors,
        settings.hnsw_ef_construction,
        settings.hnsw_ef_search,
    )
    embedder = Embedder(settings)
    reranker = Reranker(settings) if settings.rerank_enabled else None
    retriever = HybridRetriever(settings, catalog, store, embedder, reranker)
    llm = build_llm(settings)
    service = RagService(settings, catalog, store, embedder, retriever, llm)
    pipeline = IngestionPipeline(settings, catalog, store, embedder)
    return service, pipeline


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
