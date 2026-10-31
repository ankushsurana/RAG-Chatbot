"""Durable chat history: conversations, their question/answer turns, and user feedback."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ragbot.domain import Answer

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    scope TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_conversation ON turns(conversation_id, id);
"""

_TITLE_MAX_CHARS = 60

# The owner ids a request may act for. Chats made before visitors had ids have an empty owner.
Owners = tuple[str, ...]
NO_OWNER: Owners = ("",)


@dataclass(slots=True)
class Turn:
    id: int
    question: str
    scope: str
    answer: Answer
    rating: str | None


class ChatStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection as connection:
            connection.executescript(_SCHEMA)
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(conversations)")
            }
            if "owner" not in columns:
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN owner TEXT NOT NULL DEFAULT ''"
                )

    @property
    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self._path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            self._local.connection = connection
        return connection

    def list_conversations(self, owners: Owners = NO_OWNER) -> list[dict]:
        rows = self._connection.execute(
            f"SELECT id, title, updated_at FROM conversations WHERE owner IN ({_marks(owners)}) "
            "ORDER BY updated_at DESC, id",
            owners,
        ).fetchall()
        return [dict(row) for row in rows]

    def conversation_exists(self, conversation_id: str, owners: Owners = NO_OWNER) -> bool:
        row = self._connection.execute(
            f"SELECT 1 FROM conversations WHERE id = ? AND owner IN ({_marks(owners)})",
            (conversation_id, *owners),
        ).fetchone()
        return row is not None

    def turns(self, conversation_id: str) -> list[Turn]:
        rows = self._connection.execute(
            "SELECT id, question, scope, answer, rating FROM turns "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [
            Turn(
                id=row["id"],
                question=row["question"],
                scope=row["scope"],
                answer=Answer.model_validate_json(row["answer"]),
                rating=row["rating"],
            )
            for row in rows
        ]

    def add_turn(
        self,
        conversation_id: str | None,
        question: str,
        scope: str,
        answer: Answer,
        owner: str = "",
    ) -> tuple[str, int]:
        """Stores a turn, creating the conversation if needed. Returns (conversation id, turn id)."""
        now = datetime.now(UTC).isoformat()
        stored = answer.model_dump_json(exclude={"retrieved": {"__all__": {"spans"}}})
        with self._connection as connection:
            if conversation_id is None:
                conversation_id = uuid.uuid4().hex
                connection.execute(
                    "INSERT INTO conversations (id, title, created_at, updated_at, owner) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, _make_title(question), now, now, owner),
                )
            else:
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
                )
            cursor = connection.execute(
                "INSERT INTO turns (conversation_id, question, scope, answer, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, question, scope, stored, now),
            )
        return conversation_id, int(cursor.lastrowid)

    def delete_conversation(self, conversation_id: str, owners: Owners = NO_OWNER) -> bool:
        with self._connection as connection:
            cursor = connection.execute(
                f"DELETE FROM conversations WHERE id = ? AND owner IN ({_marks(owners)})",
                (conversation_id, *owners),
            )
        return cursor.rowcount > 0

    def set_rating(self, turn_id: int, rating: str | None, owners: Owners = NO_OWNER) -> bool:
        with self._connection as connection:
            cursor = connection.execute(
                "UPDATE turns SET rating = ? WHERE id = ? AND conversation_id IN "
                f"(SELECT id FROM conversations WHERE owner IN ({_marks(owners)}))",
                (rating, turn_id, *owners),
            )
        return cursor.rowcount > 0


def _marks(owners: Owners) -> str:
    return ",".join("?" * len(owners))


def _make_title(question: str) -> str:
    title = " ".join(question.split())
    if len(title) <= _TITLE_MAX_CHARS:
        return title
    return f"{title[:_TITLE_MAX_CHARS].rsplit(' ', 1)[0]}..."
