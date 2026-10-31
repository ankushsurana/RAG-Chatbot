"""Picks the part of a passage most relevant to a question.

Cross-encoders and small LLMs both do better, and much faster, on a short relevant excerpt than on
a whole 500-token passage in which the answer is one sentence.
"""

from __future__ import annotations

import re

from ragbot.ingestion.chunking import split_sentences

_WORD = re.compile(r"[a-z0-9]+(?:[-_.()][a-z0-9]+)*\)?")
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


def _terms(text: str) -> set[str]:
    words = {word.rstrip(".") for word in _WORD.findall(text.lower())}
    return {_stem(word) for word in words if word not in _STOPWORDS}


def _stem(word: str) -> str:
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def focus_text(text: str, query: str, max_chars: int) -> str:
    """Returns the run of consecutive sentences of at most `max_chars` matching the query best."""
    if len(text) <= max_chars:
        return text
    query_terms = _terms(query)
    sentences = split_sentences(text)
    if not sentences or not query_terms:
        return text[:max_chars]
    matches = [_terms(sentence) & query_terms for sentence in sentences]

    best_start, best_end, best_score = 0, 0, -1.0
    for start in range(len(sentences)):
        size = 0
        covered: set[str] = set()
        hits = 0
        for end in range(start, len(sentences)):
            size += len(sentences[end]) + 1
            if size > max_chars and end > start:
                break
            covered |= matches[end]
            hits += len(matches[end])
            # Distinct terms matter most; repeats only break ties, and earlier text wins any rest.
            score = len(covered) + 0.1 * hits
            if score > best_score:
                best_start, best_end, best_score = start, end, score
    # Grow the winning run to use the budget, since neighbouring sentences carry its context.
    excerpt = sentences[best_start : best_end + 1]
    size = sum(len(sentence) + 1 for sentence in excerpt)
    before, after = best_start - 1, best_end + 1
    while True:
        if after < len(sentences) and size + len(sentences[after]) < max_chars:
            excerpt.append(sentences[after])
            size += len(sentences[after]) + 1
            after += 1
        elif before >= 0 and size + len(sentences[before]) < max_chars:
            excerpt.insert(0, sentences[before])
            size += len(sentences[before]) + 1
            before -= 1
        else:
            break
    return " ".join(excerpt)[: max(max_chars, len(excerpt[0]))]
