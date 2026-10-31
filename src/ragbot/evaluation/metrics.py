"""Pure metric functions, kept free of I/O so they can be unit tested."""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least q% of the data at or below it."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(math.ceil(q / 100 * len(ordered)), 1)
    return ordered[rank - 1]


def first_relevant_rank(relevance: Sequence[bool]) -> int | None:
    for index, relevant in enumerate(relevance, start=1):
        if relevant:
            return index
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Share of questions with at least one relevant chunk in the top k (single-target recall)."""
    if not ranks:
        return 0.0
    return sum(rank is not None and rank <= k for rank in ranks) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks)


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def matches_answer_key(answer: str, key: Sequence[Sequence[str]]) -> bool:
    """The key is a list of required facts, each with acceptable spellings (any one must appear)."""
    lowered = answer.lower()
    return all(any(option.lower() in lowered for option in fact) for fact in key)
