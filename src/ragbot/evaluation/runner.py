"""Runs the golden question set against the real pipeline and writes a report.

Golden entries (eval/golden.jsonl):
  {"id", "question", "filename", "pages": [..], "key": [[spellings of fact 1], ...]}
Unanswerable entries omit filename and pages and set "answerable": false.

Definitions:
  retrieval hit       a retrieved chunk is from the gold file and spans a gold page
  answer correct      every fact in the answer key appears in the answer, for answered questions
  citation accuracy   of all citations in answers, the share resolving to the gold file and page
  hallucination       an answerable question answered with a wrong key or an unsupported sentence,
                      or an unanswerable question answered rather than refused
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ragbot.config import Settings
from ragbot.domain import Answer, RetrievedChunk
from ragbot.evaluation.metrics import (
    first_relevant_rank,
    matches_answer_key,
    mean_reciprocal_rank,
    percentile,
    ratio,
    recall_at_k,
)
from ragbot.service import build_service

RECALL_CUTOFFS = (1, 3, 5, 8)
# Retrieval runs before generation, so scoring it on a warm run reflects serving conditions.
_WARMUP_QUERIES = 2


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chunk_is_relevant(chunk: RetrievedChunk, item: dict) -> bool:
    return chunk.filename == item["filename"] and any(
        chunk.page_start <= page <= chunk.page_end for page in item["pages"]
    )


def score_answer(answer: Answer, item: dict) -> dict:
    answerable = item.get("answerable", True)
    unsupported = [s.text for s in answer.verification.sentences if not s.supported]
    if not answerable:
        correct = answer.refused
        hallucinated = not answer.refused
        citations_correct = citations_total = 0
    else:
        correct = not answer.refused and matches_answer_key(answer.answer, item["key"])
        hallucinated = not answer.refused and (not correct or bool(unsupported))
        citations_total = len(answer.citations)
        citations_correct = sum(
            citation.filename == item["filename"]
            and any(citation.page_start <= page <= citation.page_end for page in item["pages"])
            for citation in answer.citations
        )
    return {
        "correct": correct,
        "hallucinated": hallucinated,
        "refused": answer.refused,
        "unsupported_sentences": unsupported,
        "citations_total": citations_total,
        "citations_correct": citations_correct,
    }


def run_evaluation(
    settings: Settings, golden_path: Path, retrieval_only: bool = False, limit: int | None = None
) -> dict:
    items = load_golden(golden_path)[: limit or None]
    service, _ = build_service(settings)
    if not retrieval_only:
        service.llm.warm_up()
    for item in items[:_WARMUP_QUERIES]:
        service.retrieve(item["question"])

    records = []
    for item in items:
        answerable = item.get("answerable", True)
        retrieval = service.retrieve(item["question"])
        record = {
            "id": item["id"],
            "question": item["question"],
            "answerable": answerable,
            "retrieval_ms": retrieval.timings.retrieval_ms,
            "rerank_ms": retrieval.timings.rerank_ms,
            "embed_ms": retrieval.timings.embed_ms,
            "dense_ms": retrieval.timings.dense_ms,
            "lexical_ms": retrieval.timings.lexical_ms,
            "top_score": retrieval.chunks[0].rerank_score if retrieval.chunks else None,
            "retrieved": [
                {
                    "filename": c.filename,
                    "pages": [c.page_start, c.page_end],
                    "score": c.rerank_score,
                }
                for c in retrieval.chunks[:3]
            ],
        }
        if answerable:
            record["relevant_rank"] = first_relevant_rank(
                [chunk_is_relevant(chunk, item) for chunk in retrieval.chunks]
            )
            record["gold"] = {"filename": item["filename"], "pages": item["pages"]}
        if not retrieval_only:
            answer = service.answer(item["question"])
            record.update(score_answer(answer, item))
            record.update(
                answer=answer.answer,
                status=answer.verification.status,
                citations=[{"file": c.filename, "page": c.page} for c in answer.citations],
                total_ms=answer.timings.total_ms,
                first_token_ms=answer.timings.first_token_ms,
                generation_ms=answer.timings.generation_ms,
                prompt_tokens=answer.timings.prompt_tokens,
                completion_tokens=answer.timings.completion_tokens,
            )
        records.append(record)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model if settings.rerank_enabled else None,
            "chunk_tokens": settings.chunk_tokens,
            "chunk_overlap_ratio": settings.chunk_overlap_ratio,
            "context_max_chunks": settings.context_max_chunks,
        },
        "summary": summarize(records, retrieval_only),
        "records": records,
    }
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    name = "retrieval_evaluation.json" if retrieval_only else "evaluation.json"
    (settings.reports_dir / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def summarize(records: list[dict], retrieval_only: bool = False) -> dict:
    answerable = [record for record in records if record["answerable"]]
    ranks = [record["relevant_rank"] for record in answerable]
    retrieval_latency = [record["retrieval_ms"] for record in records]
    summary = {
        "questions": len(records),
        "answerable": len(answerable),
        "unanswerable": len(records) - len(answerable),
        "retrieval": {
            **{f"recall_at_{k}": round(recall_at_k(ranks, k), 4) for k in RECALL_CUTOFFS},
            "mrr": round(mean_reciprocal_rank(ranks), 4),
            "latency_p50_ms": percentile(retrieval_latency, 50),
            "latency_p95_ms": percentile(retrieval_latency, 95),
        },
    }
    if retrieval_only:
        return summary

    totals = [record["total_ms"] for record in records]
    generated = [record for record in records if record["first_token_ms"] is not None]
    first_tokens = [record["first_token_ms"] for record in generated]
    unanswerable = [record for record in records if not record["answerable"]]
    citations_total = sum(record["citations_total"] for record in answerable)
    citations_correct = sum(record["citations_correct"] for record in answerable)
    summary["latency"] = {
        "total_p50_ms": percentile(totals, 50),
        "total_p95_ms": percentile(totals, 95),
        "total_max_ms": max(totals, default=0.0),
        "first_token_p50_ms": percentile(first_tokens, 50),
        "first_token_p95_ms": percentile(first_tokens, 95),
        "share_within_5s": ratio(sum(total <= 5000 for total in totals), len(totals)),
    }
    summary["answers"] = {
        "answer_accuracy": ratio(sum(record["correct"] for record in answerable), len(answerable)),
        "false_refusals": sum(record["refused"] for record in answerable),
        "citation_accuracy": ratio(citations_correct, citations_total),
        "answers_with_citations": ratio(
            sum(record["citations_total"] > 0 for record in answerable if not record["refused"]),
            sum(not record["refused"] for record in answerable),
        ),
        "hallucination_rate": ratio(
            sum(record["hallucinated"] for record in records), len(records)
        ),
        "unanswerable_refusal_rate": ratio(
            sum(record["refused"] for record in unanswerable), len(unanswerable)
        ),
    }
    return summary
