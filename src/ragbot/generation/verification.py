"""Post-generation checks: resolves citations to exact pages and flags unsupported claims.

Each answer sentence is compared with the sources it cites. A claim counts as supported when its
content words appear in the source, or failing that when it is semantically close to a source
sentence. Numbers are checked strictly, since a wrong figure can still read as a close paraphrase.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ragbot.domain import BBox, Citation, RetrievedChunk, SentenceCheck, Verification
from ragbot.generation.prompts import REFUSAL_TEXT
from ragbot.ingestion.chunking import split_sentences

_CITATION_GROUP = re.compile(r"\[\s*\d+(?:\s*(?:,|;|and)\s*\d+)*\s*\]")
_DIGITS = re.compile(r"\d+")
_MISPLACED = re.compile(r"([.!?])\s*((?:\[\s*\d+(?:\s*(?:,|;|and)\s*\d+)*\s*\]\s*)+)(?=\s|$)")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")
# Reference tags the model copies from the source, such as "[SP800-34r1]" before a real citation.
_COPIED_REFERENCE = re.compile(r"\s*\[[A-Za-z][\w .,/()-]*\](?=\s*(?:\[\d|$))", re.M)
_TRAILING_LIST_MARKER = re.compile(r"\s*(?:\d+[.)]|[-*])\s*$")
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022])\s+", re.M)
_WORD = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "she",
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
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "used",
        "use",
        "using",
        "based",
        "according",
        "source",
        "sources",
        "document",
        "documents",
        "following",
        "include",
        "includes",
        "including",
    ]
)
_REFUSAL_MARKERS = (
    REFUSAL_TEXT.lower().rstrip("."),
    "could not find",
    "do not contain",
    "does not contain",
    "not mentioned in the",
    "no information",
)

EmbedFn = Callable[[list[str]], np.ndarray]


@dataclass(slots=True)
class VerifiedAnswer:
    text: str
    refused: bool
    citations: list[Citation]
    verification: Verification


def verify_answer(
    raw_answer: str,
    sources: list[RetrievedChunk],
    embed: EmbedFn,
    min_overlap: float,
    min_similarity: float,
    truncated: bool = False,
    question: str = "",
) -> VerifiedAnswer:
    text = raw_answer.strip()
    if truncated:
        text = _trim_to_last_sentence(text)
    if not text or is_refusal(text):
        return refused_answer()

    text = _COPIED_REFERENCE.sub("", text)
    by_number = {source.source_number: source for source in sources}
    # Numbers from the question or a document's name ("SP 800-82") are references, not claims.
    known_numbers = _numbers(
        " ".join([question, *(f"{src.title} {src.filename.replace('_', ' ')}" for src in sources)])
    )
    text = _MISPLACED.sub(lambda match: f" {match.group(2).strip()}{match.group(1)}", text)
    invalid: set[int] = set()
    checks: list[SentenceCheck] = []
    rewritten = text

    for line in text.splitlines():
        for sentence in split_sentences(_LIST_MARKER.sub("", line)):
            cited = [
                int(n)
                for group in _CITATION_GROUP.findall(sentence)
                for n in _DIGITS.findall(group)
            ]
            invalid.update(number for number in cited if number not in by_number)
            valid = list(dict.fromkeys(number for number in cited if number in by_number))
            bare = _CITATION_GROUP.sub("", sentence).strip()
            if not _content_words(bare):
                continue

            check = _check_sentence(
                bare, valid, by_number, embed, min_overlap, min_similarity, known_numbers
            )
            if not valid:
                # The model forgot a marker: attach the source that actually supports the claim.
                best = _best_source(bare, by_number)
                if best is not None:
                    candidate = _check_sentence(
                        bare, [best], by_number, embed, min_overlap, min_similarity, known_numbers
                    )
                    if candidate.supported:
                        check = candidate
                        rewritten = rewritten.replace(sentence, _with_marker(sentence, best), 1)
            checks.append(check)

    for number in invalid:
        rewritten = re.sub(rf"\[\s*{number}\s*\]", "", rewritten)

    citations = _build_citations(checks, by_number)
    supported = sum(check.supported for check in checks)
    groundedness = supported / len(checks) if checks else 0.0
    status = (
        "verified" if groundedness == 1.0 else "partial" if groundedness >= 0.5 else "unverified"
    )
    return VerifiedAnswer(
        text=rewritten.strip(),
        refused=False,
        citations=citations,
        verification=Verification(
            status=status,
            groundedness=round(groundedness, 3),
            sentences=checks,
            invalid_citations=sorted(invalid),
        ),
    )


def refused_answer() -> VerifiedAnswer:
    return VerifiedAnswer(
        text=REFUSAL_TEXT,
        refused=True,
        citations=[],
        verification=Verification(status="refused", groundedness=1.0),
    )


def is_refusal(text: str) -> bool:
    lowered = text.lower()
    has_citation = bool(_CITATION_GROUP.search(text))
    return any(marker in lowered for marker in _REFUSAL_MARKERS) and (
        not has_citation or REFUSAL_TEXT.lower().rstrip(".") in lowered
    )


def _trim_to_last_sentence(text: str) -> str:
    """Drops the partial final sentence left behind when the token limit cuts the answer short."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text
    end = text.rfind(sentences[-1])
    return _TRAILING_LIST_MARKER.sub("", text[:end]).rstrip()


def _with_marker(sentence: str, number: int) -> str:
    stripped = sentence.rstrip()
    if stripped and stripped[-1] in ".!?":
        return f"{stripped[:-1]} [{number}]{stripped[-1]}"
    return f"{stripped} [{number}]"


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ingly", "ings", "ing", "ies", "ied", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _content_words(text: str) -> set[str]:
    return {_stem(word) for word in _WORD.findall(text.lower()) if word not in _STOPWORDS}


def _numbers(text: str) -> set[str]:
    return {
        match.replace(",", "").rstrip("0").rstrip(".") or "0" for match in _NUMBER.findall(text)
    }


def _overlap(sentence_words: set[str], source_words: set[str]) -> float:
    if not sentence_words:
        return 0.0
    return len(sentence_words & source_words) / len(sentence_words)


def _check_sentence(
    sentence: str,
    cited: list[int],
    by_number: dict[int, RetrievedChunk],
    embed: EmbedFn,
    min_overlap: float,
    min_similarity: float,
    known_numbers: set[str],
) -> SentenceCheck:
    words = _content_words(sentence)
    cited_sources = [by_number[number] for number in cited]
    combined_text = " ".join(source.text for source in cited_sources)
    combined_words = _content_words(combined_text)

    overlap = _overlap(words, combined_words)
    missing_numbers = sorted(_numbers(sentence) - _numbers(combined_text) - known_numbers)
    score = overlap
    supported = overlap >= min_overlap and not missing_numbers

    if cited_sources and not supported and not missing_numbers:
        score = max(score, _semantic_support(sentence, cited_sources, embed))
        supported = score >= min_similarity

    return SentenceCheck(
        text=sentence,
        citations=cited,
        supported=supported,
        support_score=round(score, 3),
        unmatched_numbers=missing_numbers,
    )


def _semantic_support(sentence: str, sources: list[RetrievedChunk], embed: EmbedFn) -> float:
    words = _content_words(sentence)
    candidates = [part for source in sources for part in split_sentences(source.text)]
    candidates.sort(key=lambda part: -_overlap(words, _content_words(part)))
    shortlist = candidates[:3]
    if not shortlist:
        return 0.0
    vectors = embed([sentence, *shortlist])
    return float(np.max(vectors[1:] @ vectors[0]))


def _best_source(sentence: str, by_number: dict[int, RetrievedChunk]) -> int | None:
    words = _content_words(sentence)
    if not by_number or not words:
        return None
    return max(by_number, key=lambda n: (_overlap(words, _content_words(by_number[n].text)), -n))


def _build_citations(
    checks: list[SentenceCheck], by_number: dict[int, RetrievedChunk]
) -> list[Citation]:
    claims: dict[int, list[str]] = {}
    for check in checks:
        for number in check.citations:
            claims.setdefault(number, []).append(check.text)

    citations = []
    for number in sorted(claims):
        source = by_number[number]
        page, bbox, snippet = _locate(source, " ".join(claims[number]))
        citations.append(
            Citation(
                number=number,
                chunk_id=source.chunk_id,
                doc_id=source.doc_id,
                filename=source.filename,
                title=source.title,
                section=source.section,
                page=page,
                page_start=source.page_start,
                page_end=source.page_end,
                snippet=snippet,
                bbox=bbox,
                has_ocr=source.has_ocr,
            )
        )
    return citations


def _locate(source: RetrievedChunk, claim: str) -> tuple[int, BBox | None, str]:
    """Finds the block inside a chunk that best matches the claim, for exact page attribution."""
    words = _content_words(claim)
    best_span = None
    best_score = -1.0
    for span in source.spans:
        fragment = source.text[span["start"] : span["end"]]
        fragment_words = _content_words(fragment)
        if not fragment_words:
            continue
        # Dice score: favours blocks that match the claim without simply being longer.
        score = 2 * len(words & fragment_words) / (len(words) + len(fragment_words) or 1)
        if score > best_score:
            best_score, best_span = score, span
    if best_span is None:
        return source.page_start, None, source.text[:280]
    snippet = source.text[best_span["start"] : best_span["end"]][:280]
    return best_span["page"], tuple(best_span["bbox"]), snippet
