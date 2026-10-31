"""SQLite catalog: document registry, chunk text and metadata, and the FTS5 (BM25) lexical index."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

from ragbot.domain import Chunk, DocumentRecord, RetrievedChunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    record TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    rid INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    section TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    language TEXT NOT NULL,
    has_ocr INTEGER NOT NULL,
    spans TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id, ordinal);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, section, title, tokenize = "porter unicode61"
);
"""

_TERM = re.compile(r"[A-Za-z0-9]+(?:[-._/][A-Za-z0-9]+)*")
_STOPWORDS = frozenset(
    [
        "a",
        "about",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "me",
        "might",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)
_BM25_WEIGHTS = "1.0, 0.6, 0.3"  # text, section, title


def build_match_query(query: str) -> str:
    """Turns free text into a safe FTS5 OR query: each term is quoted, so no operator injection."""
    terms = [term for term in _TERM.findall(query) if term.lower() not in _STOPWORDS]
    if not terms:
        terms = _TERM.findall(query)
    # Adjacent terms are added as phrases too, so "discrete process" outranks either word alone.
    phrases = [f"{first} {second}" for first, second in pairwise(terms)]
    return " OR ".join(f'"{term}"' for term in dict.fromkeys([*phrases, *terms]))


class Catalog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection as connection:
            connection.executescript(_SCHEMA)

    @property
    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self._path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA mmap_size=268435456")
            self._local.connection = connection
        return connection

    # --- Documents ------------------------------------------------------------------------------

    def upsert_document(self, record: DocumentRecord) -> None:
        with self._connection as connection:
            connection.execute(
                """INSERT INTO documents (doc_id, filename, sha256, status, fingerprint, record)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(doc_id) DO UPDATE SET filename = excluded.filename,
                     sha256 = excluded.sha256, status = excluded.status,
                     fingerprint = excluded.fingerprint, record = excluded.record""",
                (
                    record.doc_id,
                    record.filename,
                    record.sha256,
                    record.status,
                    record.fingerprint,
                    record.model_dump_json(),
                ),
            )

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        row = self._connection.execute(
            "SELECT record FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return DocumentRecord.model_validate_json(row["record"]) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        rows = self._connection.execute("SELECT record FROM documents ORDER BY filename").fetchall()
        return [DocumentRecord.model_validate_json(row["record"]) for row in rows]

    def delete_document(self, doc_id: str) -> list[str]:
        """Removes a document and its chunks, returning the chunk ids to drop from the vector DB."""
        with self._connection as connection:
            rows = connection.execute(
                "SELECT rid, chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchall()
            connection.executemany(
                "DELETE FROM chunks_fts WHERE rowid = ?", ((row["rid"],) for row in rows)
            )
            connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return [row["chunk_id"] for row in rows]

    # --- Chunks ---------------------------------------------------------------------------------

    def add_chunks(self, title: str, chunks: Iterable[Chunk]) -> None:
        with self._connection as connection:
            for chunk in chunks:
                cursor = connection.execute(
                    """INSERT INTO chunks (chunk_id, doc_id, ordinal, text, section, page_start,
                       page_end, token_count, language, has_ocr, spans)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.ordinal,
                        chunk.text,
                        chunk.section,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.token_count,
                        chunk.language,
                        int(chunk.has_ocr),
                        json.dumps([asdict(span) for span in chunk.spans]),
                    ),
                )
                connection.execute(
                    "INSERT INTO chunks_fts (rowid, text, section, title) VALUES (?, ?, ?, ?)",
                    (cursor.lastrowid, chunk.text, chunk.section, title),
                )

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, RetrievedChunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._connection.execute(
            f"""SELECT c.*, d.filename, d.record FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.chunk_id IN ({placeholders})""",
            chunk_ids,
        ).fetchall()
        return {
            row["chunk_id"]: RetrievedChunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                filename=row["filename"],
                title=json.loads(row["record"])["title"],
                section=row["section"],
                text=row["text"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                token_count=row["token_count"],
                has_ocr=bool(row["has_ocr"]),
                spans=json.loads(row["spans"]),
            )
            for row in rows
        }

    def lexical_search(
        self, query: str, limit: int, doc_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """BM25 search. Returns (chunk_id, score) with higher scores being better."""
        match = build_match_query(query)
        if not match:
            return []
        doc_filter = ""
        params: list[object] = [match]
        if doc_ids:
            doc_filter = f"AND c.doc_id IN ({','.join('?' * len(doc_ids))})"
            params.extend(doc_ids)
        params.append(limit)
        rows = self._connection.execute(
            f"""SELECT c.chunk_id, bm25(chunks_fts, {_BM25_WEIGHTS}) AS rank
                FROM chunks_fts JOIN chunks c ON c.rid = chunks_fts.rowid
                WHERE chunks_fts MATCH ? {doc_filter}
                ORDER BY rank LIMIT ?""",
            params,
        ).fetchall()
        return [(row["chunk_id"], -float(row["rank"])) for row in rows]

    def chunk_count(self) -> int:
        return self._connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
