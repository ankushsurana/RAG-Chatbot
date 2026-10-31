"""Conversation layer on top of the RAG pipeline.

Decides how each message is handled, in order:
  1. small talk         answered directly, no documents involved
  2. about the chat     "what did I just ask?" is answered exactly from stored history
  3. repeated question  returns the earlier answer, so the same question stays consistent
  4. follow-up          rewritten into a standalone question using earlier turns, then RAG
  5. anything else      RAG
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator

from ragbot.domain import Answer, Citation, Timings, Verification
from ragbot.generation.prompts import build_rewrite_messages
from ragbot.service import RagService, StreamEvent
from ragbot.storage.chat_store import ChatStore, Turn

_SMALL_TALK = (
    (
        re.compile(
            r"^(?:hi+|hello+|hey+|hiya|namaste|good (?:morning|afternoon|evening))(?: there)?$"
        ),
        "Hello! Ask me anything about the documents and I'll answer with the file and page it "
        "came from.",
    ),
    (
        re.compile(r"^(?:thanks?|thank you|thx|thank you so much|thanks a lot)$"),
        "You're welcome. Ask me anything else about the documents.",
    ),
    (
        re.compile(r"^(?:ok|okay|cool|great|nice|got it|alright|fine)$"),
        "Great. Let me know if you have another question about the documents.",
    ),
    (
        re.compile(r"^(?:bye|goodbye|see you|see ya)$"),
        "Goodbye! Your chat is saved in the sidebar whenever you want to come back to it.",
    ),
    (
        re.compile(r"^(?:who are you|what are you|what can you do|how can you help(?: me)?|help)$"),
        "I answer questions using only the PDFs in this library, and I cite the file and page for "
        "each answer. Use the dropdown to search all documents or just one.",
    ),
)
_ASKS_PREVIOUS_QUESTION = re.compile(
    r"\b(?:what|which)\b.*\b(?:did|have) i\b.*\bask|\bmy (?:last|previous) question\b", re.I
)
_ASKS_FIRST_QUESTION = re.compile(
    r"\bmy first question\b|\bfirst (?:thing|question) i asked\b", re.I
)
_ASKS_PREVIOUS_ANSWER = re.compile(
    r"\bwhat (?:did|was) (?:you|your)\b.*\b(?:say|said|answer|respon[sd])|\byour (?:last|previous) "
    r"(?:answer|response|reply)\b|\b(?:repeat|say) (?:that|it|your answer)(?: again)?\b",
    re.I,
)
_ASKS_SUMMARY_OF_CHAT = re.compile(
    r"\bwhat (?:have|did) we (?:talk(?:ed)?|discuss(?:ed)?|cover(?:ed)?)\b|\b(?:list|show) my questions\b"
    r"|\bwhat (?:all )?(?:questions )?have i asked\b",
    re.I,
)
_FOLLOW_UP = re.compile(
    r"^(?:and|also|but|so|what about|how about|why|how so|then|more|tell me more|explain|elaborate"
    r"|give (?:me )?an example|for example|what else|is that|are they|can you)\b"
    r"|\b(?:it|its|they|them|their|this|that|these|those|he|she|his|her|the (?:first|second|third|"
    r"last|former|latter)(?: one)?|above|same|previous)\b",
    re.I,
)
_FOLLOW_UP_MAX_WORDS = 14
_VERY_SHORT_WORDS = 3
_MIN_REWRITE_CHARS = 8
_MAX_REWRITE_CHARS = 400
_REWRITE_LABEL = re.compile(r"^(?:standalone question|rewritten question|question)\s*:\s*", re.I)


class ChatService:
    def __init__(self, rag: RagService, store: ChatStore) -> None:
        self._rag = rag
        self._store = store

    def stream(
        self,
        question: str,
        conversation_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> Iterator[StreamEvent]:
        question = " ".join(question.split())
        if conversation_id and not self._store.conversation_exists(conversation_id):
            conversation_id = None
        turns = self._store.turns(conversation_id) if conversation_id else []
        scope = ",".join(sorted(doc_ids or []))

        direct = self._direct_answer(question, turns, scope)
        if direct is not None:
            yield StreamEvent("token", {"text": direct.answer})
            yield self._finish(question, conversation_id, scope, direct)
            return

        started = time.perf_counter()
        standalone = question
        rewrite_ms = 0.0
        if turns and is_follow_up(question):
            standalone = self._rewrite(question, turns)
            rewrite_ms = round((time.perf_counter() - started) * 1000, 1)

        for event in self._rag.stream_answer(standalone, doc_ids):
            if event.type != "final":
                yield event
                continue
            answer = Answer.model_validate(event.payload)
            answer.question = question
            answer.standalone_question = standalone if standalone != question else None
            answer.timings.rewrite_ms = rewrite_ms
            answer.timings.total_ms = round(answer.timings.total_ms + rewrite_ms, 1)
            yield self._finish(question, conversation_id, scope, answer)

    def answer(
        self,
        question: str,
        conversation_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> dict:
        for event in self.stream(question, conversation_id, doc_ids):
            if event.type == "final":
                return event.payload
        raise RuntimeError("chat stream ended without a final event")

    def _finish(
        self, question: str, conversation_id: str | None, scope: str, answer: Answer
    ) -> StreamEvent:
        conversation_id, turn_id = self._store.add_turn(conversation_id, question, scope, answer)
        answer.turn_id = turn_id
        answer.conversation_id = conversation_id
        return StreamEvent("final", answer.model_dump())

    def _direct_answer(self, question: str, turns: list[Turn], scope: str) -> Answer | None:
        normalized = _normalize(question)
        for pattern, reply in _SMALL_TALK:
            if pattern.match(normalized):
                return self._conversational(question, reply)

        if _ASKS_FIRST_QUESTION.search(question):
            return self._conversational(question, _describe_question(turns, first=True))
        if _ASKS_SUMMARY_OF_CHAT.search(question):
            return self._conversational(question, _describe_all_questions(turns))
        if _ASKS_PREVIOUS_QUESTION.search(question):
            return self._conversational(question, _describe_question(turns, first=False))
        if _ASKS_PREVIOUS_ANSWER.search(question):
            real = _document_turns(turns)
            if not real:
                return self._conversational(
                    question, "I haven't answered anything yet in this chat."
                )
            last = real[-1].answer
            return self._conversational(
                question,
                f"My previous answer was: {last.answer}",
                last.citations,
                last.verification,
            )

        for turn in reversed(_document_turns(turns)):
            if turn.scope == scope and _normalize(turn.question) == normalized:
                repeated = turn.answer.model_copy(deep=True)
                repeated.question = question
                repeated.repeated = True
                repeated.timings = Timings()
                return repeated
        return None

    def _conversational(
        self,
        question: str,
        text: str,
        citations: list[Citation] | None = None,
        verification: Verification | None = None,
    ) -> Answer:
        return Answer(
            question=question,
            answer=text,
            refused=False,
            citations=citations or [],
            verification=verification or Verification(status="not_applicable", groundedness=1.0),
            retrieved=[],
            timings=Timings(),
            model=self._rag.llm.model,
            conversational=True,
        )

    def _rewrite(self, question: str, turns: list[Turn]) -> str:
        real = _document_turns(turns)
        if not real:
            return question
        history = [(turn.question, turn.answer.answer) for turn in real[-3:]]
        reply = "".join(
            event.text for event in self._rag.llm.stream(build_rewrite_messages(question, history))
        )
        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        rewritten = _REWRITE_LABEL.sub("", lines[0]).strip("\"'\u201c\u201d ") if lines else ""
        if _MIN_REWRITE_CHARS <= len(rewritten) <= _MAX_REWRITE_CHARS:
            return rewritten
        return f"{real[-1].question} {question}"


def is_follow_up(question: str) -> bool:
    words = len(question.split())
    return words <= _VERY_SHORT_WORDS or (
        words <= _FOLLOW_UP_MAX_WORDS and bool(_FOLLOW_UP.search(question))
    )


def _normalize(question: str) -> str:
    return re.sub(r"[^\w\s]", "", question.lower()).strip()


def _document_turns(turns: list[Turn]) -> list[Turn]:
    """Turns that asked the documents something, leaving out greetings and chat meta-questions."""
    return [turn for turn in turns if not turn.answer.conversational]


def _describe_question(turns: list[Turn], first: bool) -> str:
    real = _document_turns(turns)
    if not real:
        return "You haven't asked me anything about the documents yet in this chat."
    if first:
        return f"Your first question was: \u201c{real[0].question}\u201d"
    return f"Your previous question was: \u201c{real[-1].question}\u201d"


def _describe_all_questions(turns: list[Turn]) -> str:
    real = _document_turns(turns)
    if not real:
        return "You haven't asked me anything about the documents yet in this chat."
    questions = "\n".join(f"- {turn.question}" for turn in real)
    return f"So far you have asked:\n{questions}"
