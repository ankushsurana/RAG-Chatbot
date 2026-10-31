"""Core data structures shared across ingestion, retrieval, generation and the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]
BlockSource = Literal["native", "ocr_page", "ocr_image"]


# --- Ingestion ---------------------------------------------------------------------------------


@dataclass(slots=True)
class TextBlock:
    text: str
    bbox: BBox
    source: BlockSource = "native"


@dataclass(slots=True)
class ExtractedPage:
    number: int  # 1-based physical page number
    width: float
    height: float
    blocks: list[TextBlock] = field(default_factory=list)
    section: str = ""
    ocr_used: bool = False
    image_ocr_blocks: int = 0
    skipped_reason: str | None = None


@dataclass(slots=True)
class ExtractedDocument:
    filename: str
    title: str
    page_count: int
    language: str
    pages: list[ExtractedPage]
    extract_seconds: float = 0.0
    removed_boilerplate_blocks: int = 0

    @property
    def ocr_pages(self) -> int:
        return sum(page.ocr_used for page in self.pages)

    @property
    def image_ocr_blocks(self) -> int:
        return sum(page.image_ocr_blocks for page in self.pages)

    @property
    def skipped_pages(self) -> int:
        return sum(page.skipped_reason is not None for page in self.pages)


@dataclass(slots=True)
class ChunkSpan:
    """Where a slice of a chunk's text sits in the source PDF."""

    page: int
    start: int
    end: int
    bbox: BBox


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    section: str
    page_start: int
    page_end: int
    token_count: int
    language: str
    has_ocr: bool
    spans: list[ChunkSpan]


class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    title: str
    sha256: str
    page_count: int
    ocr_pages: int = 0
    image_ocr_blocks: int = 0
    skipped_pages: int = 0
    removed_boilerplate_blocks: int = 0
    chunk_count: int = 0
    language: str = "unknown"
    status: Literal["processing", "ready", "failed"] = "processing"
    fingerprint: str = ""
    extract_seconds: float = 0.0
    chunk_seconds: float = 0.0
    embed_seconds: float = 0.0
    index_seconds: float = 0.0
    ingested_at: str = ""
    error: str | None = None


# --- Retrieval and answering -------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    title: str
    section: str
    text: str
    page_start: int
    page_end: int
    token_count: int = 0
    has_ocr: bool
    spans: list[dict] = Field(default_factory=list)
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None
    used_in_context: bool = False
    source_number: int | None = None


class Timings(BaseModel):
    rewrite_ms: float = 0.0
    embed_ms: float = 0.0
    dense_ms: float = 0.0
    lexical_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    retrieval_ms: float = 0.0
    first_token_ms: float | None = None
    generation_ms: float = 0.0
    verification_ms: float = 0.0
    total_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    timings: Timings = Field(default_factory=Timings)


class Citation(BaseModel):
    number: int
    chunk_id: str
    doc_id: str
    filename: str
    title: str
    section: str
    page: int
    page_start: int
    page_end: int
    snippet: str
    bbox: BBox | None = None
    has_ocr: bool = False


class SentenceCheck(BaseModel):
    text: str
    citations: list[int]
    supported: bool
    support_score: float
    unmatched_numbers: list[str] = Field(default_factory=list)


class Verification(BaseModel):
    status: Literal["verified", "partial", "unverified", "refused", "not_applicable"]
    groundedness: float
    sentences: list[SentenceCheck] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[Citation]
    verification: Verification
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    timings: Timings
    model: str
    turn_id: int | None = None
    conversation_id: str | None = None
    standalone_question: str | None = None
    repeated: bool = False
    conversational: bool = False
