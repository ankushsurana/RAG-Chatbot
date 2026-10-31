from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragbot.chat import ChatService, is_follow_up
from ragbot.domain import Answer, Timings, Verification
from ragbot.generation.llm import LLMEvent
from ragbot.service import StreamEvent
from ragbot.storage.chat_store import ChatStore


class FakeRag:
    """Stands in for the pipeline: records what was searched and answers from a script."""

    def __init__(self, rewrite: str = "What does git reset --hard do?") -> None:
        self.searched: list[str] = []
        self.scopes: list[list[str] | None] = []
        self._rewrite = rewrite
        self.rewrites = 0
        self.llm = SimpleNamespace(model="fake", stream=self._stream)

    def _stream(self, messages: list[dict[str, str]]) -> Iterator[LLMEvent]:
        self.rewrites += 1
        self.last_rewrite_prompt = messages[1]["content"]
        yield LLMEvent(text=self._rewrite)

    def stream_answer(self, question: str, doc_ids=None) -> Iterator[StreamEvent]:
        self.searched.append(question)
        self.scopes.append(doc_ids)
        answer = Answer(
            question=question,
            answer=f"Answer about {question} [1].",
            refused=False,
            citations=[],
            verification=Verification(status="verified", groundedness=1.0),
            retrieved=[],
            timings=Timings(total_ms=100.0),
            model="fake",
        )
        yield StreamEvent("token", {"text": answer.answer})
        yield StreamEvent("final", answer.model_dump())


@pytest.fixture
def store(tmp_path: Path) -> ChatStore:
    return ChatStore(tmp_path / "chat.sqlite")


@pytest.fixture
def rag() -> FakeRag:
    return FakeRag()


@pytest.fixture
def chat(rag: FakeRag, store: ChatStore) -> ChatService:
    return ChatService(rag, store)


class TestHistory:
    def test_first_message_creates_titled_conversation(self, chat, store):
        result = chat.answer("How do I undo a commit?")
        conversations = store.list_conversations()
        assert [c["id"] for c in conversations] == [result["conversation_id"]]
        assert conversations[0]["title"] == "How do I undo a commit?"

    def test_turns_join_the_same_conversation_and_survive_reload(self, chat, store, tmp_path):
        first = chat.answer("How do I undo a commit?")
        second = chat.answer("How do I create a branch in Git?", first["conversation_id"])
        assert second["conversation_id"] == first["conversation_id"]
        reloaded = ChatStore(tmp_path / "chat.sqlite")
        assert [t.question for t in reloaded.turns(first["conversation_id"])] == [
            "How do I undo a commit?",
            "How do I create a branch in Git?",
        ]

    def test_unknown_conversation_starts_a_new_one(self, chat):
        result = chat.answer("How do I undo a commit?", "does-not-exist")
        assert result["conversation_id"] != "does-not-exist"

    def test_new_chat_has_no_memory_of_other_chats(self, chat, rag):
        chat.answer("How do I undo a commit?")
        chat.answer("Why is that dangerous?")
        assert rag.rewrites == 0
        assert rag.searched[-1] == "Why is that dangerous?"

    def test_long_titles_are_shortened(self, chat, store):
        chat.answer("word " * 40)
        title = store.list_conversations()[0]["title"]
        assert len(title) <= 63
        assert title.endswith("...")


class TestContext:
    def test_follow_up_is_rewritten_before_searching(self, chat, rag):
        first = chat.answer("What does git reset do?")
        result = chat.answer("Why is it dangerous?", first["conversation_id"])
        assert rag.searched[-1] == "What does git reset --hard do?"
        assert "What does git reset do?" in rag.last_rewrite_prompt
        assert result["question"] == "Why is it dangerous?"
        assert result["standalone_question"] == "What does git reset --hard do?"

    def test_standalone_question_is_not_rewritten(self, chat, rag):
        first = chat.answer("What does git reset do?")
        chat.answer(
            "How many pages describe the FAA instrument approach systems?", first["conversation_id"]
        )
        assert rag.rewrites == 0

    def test_failed_rewrite_falls_back_to_joining_questions(self, store):
        rag = FakeRag(rewrite="")
        chat = ChatService(rag, store)
        first = chat.answer("What does git reset do?")
        chat.answer("Why is it dangerous?", first["conversation_id"])
        assert rag.searched[-1] == "What does git reset do? Why is it dangerous?"

    @pytest.mark.parametrize(
        "question",
        ["Why is it dangerous?", "Explain more", "What about the second one?", "And then?", "more"],
    )
    def test_detects_follow_ups(self, question):
        assert is_follow_up(question)

    def test_full_question_is_not_a_follow_up(self):
        assert not is_follow_up("How do I rebase a branch onto main in Git?")


class TestRepeatedQuestions:
    def test_repeat_returns_earlier_answer_without_new_search(self, chat, rag):
        first = chat.answer("What does git reset do?")
        repeat = chat.answer("what does  GIT reset do", first["conversation_id"])
        assert len(rag.searched) == 1
        assert repeat["repeated"] is True
        assert repeat["answer"] == first["answer"]
        assert repeat["turn_id"] != first["turn_id"]

    def test_same_question_in_another_scope_searches_again(self, chat, rag):
        first = chat.answer("What does git reset do?")
        chat.answer("What does git reset do?", first["conversation_id"], ["doc1"])
        assert len(rag.searched) == 2
        assert rag.scopes[-1] == ["doc1"]


class TestConversationalMessages:
    @pytest.mark.parametrize(
        "message", ["Hi", "hello!", "Thanks", "thank you", "Who are you?", "ok"]
    )
    def test_small_talk_skips_documents(self, chat, rag, message):
        result = chat.answer(message)
        assert rag.searched == []
        assert result["conversational"] is True
        assert result["citations"] == []

    def test_recalls_previous_question(self, chat):
        first = chat.answer("What does git reset do?")
        chat.answer("Thanks", first["conversation_id"])
        result = chat.answer("What was my last question?", first["conversation_id"])
        assert "What does git reset do?" in result["answer"]

    def test_recalls_first_question_and_list(self, chat):
        cid = chat.answer("What does git reset do?")["conversation_id"]
        chat.answer("How do I create a branch in Git?", cid)
        assert "git reset" in chat.answer("What was my first question?", cid)["answer"]
        listed = chat.answer("What have I asked so far?", cid)["answer"]
        assert "git reset" in listed
        assert "create a branch" in listed

    def test_repeats_previous_answer(self, chat):
        first = chat.answer("What does git reset do?")
        result = chat.answer("What did you say?", first["conversation_id"])
        assert first["answer"] in result["answer"]

    def test_history_question_in_empty_chat(self, chat):
        assert "haven't asked" in chat.answer("What was my last question?")["answer"]

    def test_small_talk_never_becomes_context_for_rewrites(self, chat, rag):
        first = chat.answer("Hi")
        chat.answer("Why is it dangerous?", first["conversation_id"])
        assert rag.rewrites == 0


class TestDeleteChat:
    def test_removes_chat_and_its_turns(self, chat, store):
        cid = chat.answer("What does git reset do?")["conversation_id"]
        assert store.delete_conversation(cid)
        assert store.list_conversations() == []
        assert store.turns(cid) == []
        assert not store.conversation_exists(cid)

    def test_leaves_other_chats_alone(self, chat, store):
        keep = chat.answer("What does git reset do?")["conversation_id"]
        remove = chat.answer("How do I create a branch in Git?")["conversation_id"]
        store.delete_conversation(remove)
        assert [c["id"] for c in store.list_conversations()] == [keep]

    def test_deleting_unknown_chat_reports_it(self, store):
        assert not store.delete_conversation("nope")


class TestFeedback:
    def test_rating_is_stored_changed_and_cleared(self, chat, store):
        result = chat.answer("What does git reset do?")
        turn_id, cid = result["turn_id"], result["conversation_id"]
        assert store.set_rating(turn_id, "up")
        assert store.turns(cid)[0].rating == "up"
        assert store.set_rating(turn_id, "down")
        assert store.turns(cid)[0].rating == "down"
        assert store.set_rating(turn_id, None)
        assert store.turns(cid)[0].rating is None

    def test_rating_unknown_message_fails(self, store):
        assert not store.set_rating(999, "up")
