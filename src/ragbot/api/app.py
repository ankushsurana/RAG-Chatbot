"""FastAPI application: chat (JSON and SSE), saved chats, retrieval, documents, page previews."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pymupdf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragbot import __version__
from ragbot.chat import ChatService
from ragbot.config import Settings, get_settings
from ragbot.domain import Answer, DocumentRecord, RetrievalResult
from ragbot.generation.llm import LLMError
from ragbot.service import build_service
from ragbot.storage.chat_store import ChatStore

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_PREVIEW_DPI = 110
_HIGHLIGHT_COLOR = (1.0, 0.82, 0.2)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=64)
    doc_ids: list[str] | None = None


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"] | None


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=24)
    doc_ids: list[str] | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO)
        service, _ = build_service(settings)
        app.state.service = service
        app.state.chat_store = ChatStore(settings.chat_db_path)
        app.state.chat = ChatService(service, app.state.chat_store)
        service.embedder.embed_query("warm up")
        try:
            service.llm.warm_up()
        except LLMError as error:
            logger.warning(
                "LLM warm-up failed, answers are unavailable until it recovers: %s", error
            )
        yield

    app = FastAPI(title="RAG Chatbot", version=__version__, lifespan=lifespan)

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, error: LLMError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.get("/api/health")
    def health(request: Request) -> dict:
        service = request.app.state.service
        documents = service.catalog.list_documents()
        return {
            "status": "ok",
            "version": __version__,
            "llm": {"model": service.llm.model, "available": service.llm.is_available()},
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model if settings.rerank_enabled else None,
            "vector_db": "ChromaDB (HNSW, cosine)",
            "documents": len(documents),
            "pages": sum(document.page_count for document in documents),
            "chunks": service.catalog.chunk_count(),
            "vectors": service.store.count(),
        }

    @app.post("/api/chat", response_model=Answer)
    def chat(body: ChatRequest, request: Request) -> dict:
        return request.app.state.chat.answer(body.question, body.conversation_id, body.doc_ids)

    @app.post("/api/chat/stream")
    def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
        return StreamingResponse(
            _sse_events(request.app.state.chat, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/conversations")
    def conversations(request: Request) -> list[dict]:
        return request.app.state.chat_store.list_conversations()

    @app.get("/api/conversations/{conversation_id}")
    def conversation(conversation_id: str, request: Request) -> list[dict]:
        store = request.app.state.chat_store
        if not store.conversation_exists(conversation_id):
            raise HTTPException(404, "Chat not found")
        return [
            {
                "id": turn.id,
                "question": turn.question,
                "answer": turn.answer.model_dump(),
                "rating": turn.rating,
            }
            for turn in store.turns(conversation_id)
        ]

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: str, request: Request) -> Response:
        if not request.app.state.chat_store.delete_conversation(conversation_id):
            raise HTTPException(404, "Chat not found")
        return Response(status_code=204)

    @app.post("/api/turns/{turn_id}/feedback")
    def feedback(turn_id: int, body: FeedbackRequest, request: Request) -> dict:
        if not request.app.state.chat_store.set_rating(turn_id, body.rating):
            raise HTTPException(404, "Message not found")
        return {"turn_id": turn_id, "rating": body.rating}

    @app.post("/api/retrieve", response_model=RetrievalResult)
    def retrieve(body: RetrieveRequest, request: Request) -> RetrievalResult:
        return request.app.state.service.retrieve(body.query, body.top_k, body.doc_ids)

    @app.get("/api/documents", response_model=list[DocumentRecord])
    def documents(request: Request) -> list[DocumentRecord]:
        return request.app.state.service.catalog.list_documents()

    @app.get("/api/documents/{doc_id}/pages/{page_number}")
    def page_preview(
        doc_id: str,
        page_number: int,
        request: Request,
        highlight: str | None = Query(default=None, pattern=r"^[\d.,\-]+$"),
    ) -> Response:
        """Renders a page as PNG, optionally highlighting the cited text region."""
        record = request.app.state.service.catalog.get_document(doc_id)
        if record is None:
            raise HTTPException(404, "Document not found")
        path = settings.pdf_dir / record.filename
        if not path.exists():
            raise HTTPException(404, "Source PDF is no longer available")
        with pymupdf.open(path) as pdf:
            if not 1 <= page_number <= pdf.page_count:
                raise HTTPException(404, "Page out of range")
            page = pdf[page_number - 1]
            if highlight:
                rect = _parse_bbox(highlight)
                annotation = page.add_rect_annot(rect)
                annotation.set_colors(stroke=_HIGHLIGHT_COLOR, fill=_HIGHLIGHT_COLOR)
                annotation.set_opacity(0.35)
                annotation.update()
            image = page.get_pixmap(dpi=_PREVIEW_DPI, annots=True, alpha=False).tobytes("png")
        return Response(image, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
    return app


def _sse_events(chat: ChatService, body: ChatRequest) -> Iterator[str]:
    try:
        for event in chat.stream(body.question, body.conversation_id, body.doc_ids):
            yield f"event: {event.type}\ndata: {json.dumps(event.payload)}\n\n"
    except LLMError as error:
        logger.warning("generation failed: %s", error)
        payload = {"detail": "The language model is unavailable. Check that Ollama is running."}
        yield f"event: error\ndata: {json.dumps(payload)}\n\n"


def _parse_bbox(value: str) -> pymupdf.Rect:
    try:
        parts = [float(part) for part in value.split(",")]
    except ValueError as error:
        raise HTTPException(422, "Invalid highlight box") from error
    if len(parts) != 4:
        raise HTTPException(422, "Invalid highlight box")
    return pymupdf.Rect(parts)
