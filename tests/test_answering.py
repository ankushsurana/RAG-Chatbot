from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest

from ragbot.config import Settings
from ragbot.domain import DocumentRecord, RetrievalResult, Timings
from ragbot.evaluation.metrics import (
    first_relevant_rank,
    matches_answer_key,
    mean_reciprocal_rank,
    percentile,
    recall_at_k,
)
from ragbot.generation.llm import LLMEvent
from ragbot.generation.prompts import REFUSAL_TEXT, build_messages, select_context
from ragbot.generation.verification import verify_answer
from ragbot.retrieval import reciprocal_rank_fusion, resolve_named_documents
from ragbot.service import RagService
from ragbot.storage.catalog import build_match_query

SOURCE = "The relief valve opens at 150 psi. Operators must inspect the valve every 30 days."


def _no_semantic(texts: list[str]) -> np.ndarray:
    """Orthogonal vectors: semantic similarity never rescues a claim in these tests."""
    return np.eye(len(texts))


def _verify(answer: str, sources: list, **kwargs):
    return verify_answer(answer, sources, _no_semantic, 0.5, 0.72, **kwargs)


class TestVerification:
    def test_supported_claim_is_verified_with_exact_page(self, make_source):
        result = _verify(
            "The relief valve opens at 150 psi [1].", [make_source(1, SOURCE, page=42)]
        )
        assert result.verification.status == "verified"
        assert [(c.number, c.page, c.filename) for c in result.citations] == [(1, 42, "manual.pdf")]

    def test_wrong_number_is_flagged_even_with_matching_words(self, make_source):
        result = _verify("The relief valve opens at 175 psi [1].", [make_source(1, SOURCE)])
        assert result.verification.status == "unverified"
        assert result.verification.sentences[0].unmatched_numbers == ["175"]

    def test_fabricated_claim_is_flagged(self, make_source):
        result = _verify(
            "Quantum flux capacitors harmonize turbine plasma [1].", [make_source(1, SOURCE)]
        )
        assert not result.verification.sentences[0].supported

    def test_citation_to_missing_source_is_removed(self, make_source):
        result = _verify("The relief valve opens at 150 psi [1][9].", [make_source(1, SOURCE)])
        assert result.verification.invalid_citations == [9]
        assert "[9]" not in result.text

    def test_missing_marker_is_attached_to_supporting_source(self, make_source):
        sources = [make_source(1, "Bananas grow in tropical climates."), make_source(2, SOURCE)]
        result = _verify("Operators must inspect the valve every 30 days.", sources)
        assert "[2]" in result.text
        assert result.verification.status == "verified"

    def test_citation_after_period_is_normalized(self, make_source):
        result = _verify("The relief valve opens at 150 psi. [1]", [make_source(1, SOURCE)])
        assert result.verification.sentences[0].citations == [1]

    def test_reference_tag_copied_from_source_is_removed(self, make_source):
        result = _verify(
            "The relief valve opens at 150 psi. [SP800-34r1] [1]", [make_source(1, SOURCE)]
        )
        assert "SP800" not in result.text
        assert result.verification.sentences[0].citations == [1]
        assert result.verification.status == "verified"

    def test_numbers_from_the_question_are_not_treated_as_claims(self, make_source):
        result = _verify(
            "SP 800-82 says the relief valve opens at 150 psi [1].",
            [make_source(1, SOURCE)],
            question="What does SP 800-82 say about the relief valve?",
        )
        assert result.verification.status == "verified"

    def test_numbers_from_the_document_name_are_not_treated_as_claims(self, make_source):
        source = make_source(1, SOURCE)
        source.filename = "NIST_SP_800-82_r3_OT_Security_Guide.pdf"
        result = _verify("NIST SP 800-82 says the relief valve opens at 150 psi [1].", [source])
        assert result.verification.status == "verified"

    def test_invented_number_is_still_flagged(self, make_source):
        result = _verify(
            "SP 800-82 says the relief valve opens at 175 psi [1].",
            [make_source(1, SOURCE)],
            question="What does SP 800-82 say about the relief valve?",
        )
        assert result.verification.sentences[0].unmatched_numbers == ["175"]

    def test_refusal_returns_no_citations(self, make_source):
        result = _verify(REFUSAL_TEXT, [make_source(1, SOURCE)])
        assert result.refused
        assert result.citations == []

    def test_truncated_answer_drops_partial_sentence(self, make_source):
        result = _verify(
            "The relief valve opens at 150 psi [1]. Operators must also",
            [make_source(1, SOURCE)],
            truncated=True,
        )
        assert result.text == "The relief valve opens at 150 psi [1]."

    def test_truncated_list_drops_dangling_marker(self, make_source):
        result = _verify(
            "The relief valve opens at 150 psi [1].\n2. Operators must inspect the valve [1].\n3.",
            [make_source(1, SOURCE)],
            truncated=True,
        )
        assert not result.text.rstrip().endswith("3.")

    def test_page_comes_from_block_matching_the_claim(self, make_source):
        first = "Bananas grow in tropical climates."
        text = f"{first} {SOURCE}"
        source = make_source(1, text)
        source.page_end = 8
        source.spans = [
            {"page": 7, "start": 0, "end": len(first), "bbox": [0, 0, 1, 1]},
            {"page": 8, "start": len(first) + 1, "end": len(text), "bbox": [2, 2, 3, 3]},
        ]
        result = _verify("The relief valve opens at 150 psi [1].", [source])
        assert result.citations[0].page == 8
        assert result.citations[0].bbox == (2, 2, 3, 3)


class TestRetrievalHelpers:
    def test_rrf_rewards_agreement(self):
        scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
        assert max(scores, key=scores.get) == "b"

    def test_match_query_neutralizes_fts_syntax(self):
        query = build_match_query('NEAR(valve) OR "drop table" * AND:x SP 800-53A')
        assert query.count('"') % 2 == 0
        assert '"800-53A"' in query
        assert "*" not in query

    def test_match_query_adds_adjacent_terms_as_phrases(self):
        query = build_match_query("How does it define a discrete process?")
        assert '"discrete process"' in query
        assert '"discrete"' in query

    def test_match_query_keeps_terms_for_all_stopword_questions(self):
        assert build_match_query("what is it") != ""


class TestNamedDocuments:
    DOCUMENTS = (
        ("a", "NIST_SP_800-82_r3_OT_Security_Guide.pdf"),
        ("b", "NIST_SP_800-53A_r5_Security_Control_Assessment.pdf"),
        ("c", "Think_Stats_2e.pdf"),
        ("d", "NASA_Systems_Engineering_Handbook_Rev2.pdf"),
    )

    def _resolve(self, query: str) -> tuple[list[str], str]:
        records = [
            DocumentRecord(doc_id=doc_id, filename=name, title=name, sha256="x", page_count=1)
            for doc_id, name in self.DOCUMENTS
        ]
        return resolve_named_documents(query, records)

    def _named(self, query: str) -> list[str]:
        return self._resolve(query)[0]

    def test_identifier_selects_its_document(self):
        assert self._named("How does NIST SP 800-82 define a discrete process?") == ["a"]

    def test_title_selects_its_document(self):
        assert self._named("What does think stats say about the CDF?") == ["c"]

    def test_organization_selects_its_document(self):
        assert self._named("What is NASA's definition of verification?") == ["d"]

    def test_name_shared_by_documents_selects_nothing(self):
        assert self._named("What does NIST SP say about risk?") == []

    def test_partial_identifier_does_not_match(self):
        assert self._named("What is in SP 800-53?") == []

    def test_name_is_removed_from_the_search_text(self):
        assert self._resolve("How does NIST SP 800-82 define a discrete process?")[1] == (
            "How does define a discrete process?"
        )

    def test_question_is_kept_when_it_is_only_a_name(self):
        assert self._resolve("Think Stats")[1] == "Think Stats"

    def test_unnamed_question_is_searched_unchanged(self):
        assert self._resolve("What is a discrete process?") == ([], "What is a discrete process?")

    def test_ordinary_question_selects_nothing(self):
        assert self._named("What is a discrete process?") == []


class TestPrompt:
    def test_context_respects_budget_and_numbers_sources(self, make_source):
        chunks = [make_source(0, f"text {i}") for i in range(5)]
        for chunk in chunks:
            chunk.token_count = 500
        selected = select_context(chunks, max_chunks=4, max_tokens=1100)
        assert [chunk.source_number for chunk in selected] == [1, 2]

    def test_messages_contain_provenance_and_question(self, make_source):
        messages = build_messages("Why?", [make_source(1, SOURCE, page=42)])
        assert "page 42" in messages[1]["content"]
        assert messages[1]["content"].endswith("Question: Why?")
        assert REFUSAL_TEXT in messages[0]["content"]


class FakeLLM:
    model = "fake"

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def stream(self, messages: list[dict[str, str]]) -> Iterator[LLMEvent]:
        self.calls += 1
        yield LLMEvent(text=self._text)
        yield LLMEvent(prompt_tokens=10, completion_tokens=5, finish_reason="stop")


def _service(llm, chunks) -> RagService:
    retriever = SimpleNamespace(
        retrieve=lambda query, top_k=None, doc_ids=None: RetrievalResult(
            query=query, chunks=chunks, timings=Timings(retrieval_ms=1.0)
        )
    )
    embedder = SimpleNamespace(embed_sentences=_no_semantic)
    return RagService(Settings(min_relevance_logit=-6.0), None, None, embedder, retriever, llm)


class TestService:
    def test_irrelevant_results_are_refused_without_calling_llm(self, make_source):
        chunk = make_source(0, SOURCE)
        chunk.rerank_score = -10.5
        llm = FakeLLM("should not be used")
        answer = _service(llm, [chunk]).answer("What is the capital of France?")
        assert answer.refused
        assert llm.calls == 0

    def test_empty_corpus_is_refused(self):
        assert _service(FakeLLM("x"), []).answer("Anything?").refused

    def test_grounded_answer_flows_through(self, make_source):
        chunk = make_source(0, SOURCE, page=11)
        chunk.rerank_score = 5.0
        chunk.token_count = 50
        answer = _service(FakeLLM("The relief valve opens at 150 psi [1]."), [chunk]).answer(
            "Valve?"
        )
        assert not answer.refused
        assert answer.citations[0].page == 11
        assert answer.verification.status == "verified"
        assert answer.timings.completion_tokens == 5


class ScriptedLLM:
    """Replies with each scripted text in turn and records the prompts it received."""

    model = "scripted"

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def stream(self, messages: list[dict[str, str]]) -> Iterator[LLMEvent]:
        self.prompts.append(messages[1]["content"])
        yield LLMEvent(text=self._replies.pop(0))
        yield LLMEvent(prompt_tokens=10, completion_tokens=5, finish_reason="stop")


def _ranked_chunks(make_source, *scores: float) -> list:
    chunks = []
    for index, score in enumerate(scores):
        chunk = make_source(0, SOURCE if index == 0 else f"Other passage number {index}.")
        chunk.chunk_id = f"c{index}"
        chunk.rerank_score = score
        chunk.token_count = 50
        chunks.append(chunk)
    return chunks


class TestRefusalRetry:
    def test_strong_match_gets_a_focused_second_try(self, make_source):
        llm = ScriptedLLM(REFUSAL_TEXT, "The relief valve opens at 150 psi [1].")
        answer = _service(llm, _ranked_chunks(make_source, 5.0, 4.0, 3.0)).answer("Valve?")
        assert not answer.refused
        assert "150 psi" in answer.answer
        assert len(llm.prompts) == 2
        assert "Other passage number 2" in llm.prompts[0]
        assert "Other passage number 2" not in llm.prompts[1]

    def test_weak_match_is_not_retried(self, make_source):
        llm = ScriptedLLM(REFUSAL_TEXT)
        answer = _service(llm, _ranked_chunks(make_source, -2.0)).answer("Valve?")
        assert answer.refused
        assert len(llm.prompts) == 1

    def test_second_refusal_stands(self, make_source):
        llm = ScriptedLLM(REFUSAL_TEXT, REFUSAL_TEXT)
        assert _service(llm, _ranked_chunks(make_source, 5.0)).answer("Valve?").refused

    def test_refusal_text_from_first_try_is_not_streamed(self, make_source):
        llm = ScriptedLLM(REFUSAL_TEXT, "The relief valve opens at 150 psi [1].")
        service = _service(llm, _ranked_chunks(make_source, 5.0))
        tokens = [e.payload["text"] for e in service.stream_answer("Valve?") if e.type == "token"]
        assert REFUSAL_TEXT not in "".join(tokens)
        assert "150 psi" in "".join(tokens)

    def test_best_source_is_placed_last_in_the_prompt(self, make_source):
        llm = ScriptedLLM("The relief valve opens at 150 psi [1].")
        _service(llm, _ranked_chunks(make_source, 5.0, 4.0)).answer("Valve?")
        assert llm.prompts[0].index("Other passage") < llm.prompts[0].index("relief valve")


class TestMetrics:
    def test_percentile_nearest_rank(self):
        values = list(range(1, 101))
        assert percentile(values, 95) == 95
        assert percentile(values, 50) == 50
        assert percentile([], 95) == 0.0

    def test_recall_and_mrr(self):
        ranks = [1, 3, None, 2]
        assert recall_at_k(ranks, 1) == 0.25
        assert recall_at_k(ranks, 3) == 0.75
        assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 1 / 3 + 0.5) / 4)

    def test_first_relevant_rank(self):
        assert first_relevant_rank([False, True]) == 2
        assert first_relevant_rank([False]) is None

    def test_answer_key_needs_all_facts(self):
        key = [["150 psi"], ["30 days", "thirty days"]]
        assert matches_answer_key("Opens at 150 PSI, inspect every thirty days.", key)
        assert not matches_answer_key("Opens at 150 psi.", key)
