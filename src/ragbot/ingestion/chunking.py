"""Deterministic sentence-aware chunking with token-accurate windows and full page provenance."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache

import py3langid
from tokenizers import Tokenizer

from ragbot.domain import BBox, Chunk, ChunkSpan, ExtractedDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_ABBREVIATION = re.compile(
    r"(?:\b(?:e\.g|i\.e|etc|vs|cf|fig|figs|eq|no|nos|vol|pp|p|sec|ch|dr|mr|mrs|ms|prof|inc|ltd|"
    r"co|corp|art|cl|amend|u\.s|v)|\b[A-Z])\.$",
    re.I,
)


@dataclass(slots=True)
class _Unit:
    text: str
    page: int
    bbox: BBox
    section: str
    tokens: int
    has_ocr: bool


class TokenCounter:
    """Counts tokens with the embedding model's own tokenizer, so windows match what it sees."""

    def __init__(self, tokenizer_name: str) -> None:
        self._tokenizer = _load_tokenizer(tokenizer_name)

    def count_many(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        encodings = self._tokenizer.encode_batch_fast(texts, add_special_tokens=False)
        return [len(encoding.ids) for encoding in encodings]

    def count(self, text: str) -> int:
        return self.count_many([text])[0]

    def split_to_limit(self, text: str, limit: int) -> list[str]:
        """Splits text at token boundaries into pieces of at most `limit` tokens each."""
        offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
        pieces: list[str] = []
        for start_index in range(0, len(offsets), limit):
            window = offsets[start_index : start_index + limit]
            start = window[0][0]
            end = (
                offsets[start_index + limit][0] if start_index + limit < len(offsets) else len(text)
            )
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
        return pieces


@lru_cache(maxsize=2)
def _load_tokenizer(name: str) -> Tokenizer:
    tokenizer = Tokenizer.from_pretrained(name)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def split_sentences(text: str) -> list[str]:
    """Regex splitter that re-joins fragments produced by common abbreviations."""
    sentences: list[str] = []
    for fragment in _SENTENCE_BOUNDARY.split(text):
        fragment = fragment.strip()
        if not fragment:
            continue
        if sentences and _ABBREVIATION.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {fragment}"
        else:
            sentences.append(fragment)
    return sentences


def build_chunks(
    document: ExtractedDocument,
    doc_id: str,
    counter: TokenCounter,
    chunk_tokens: int,
    overlap_ratio: float,
    min_tokens: int,
    fingerprint: str,
) -> list[Chunk]:
    """Greedy sentence windows of at most `chunk_tokens` tokens.

    Sentences longer than `overlap_ratio * chunk_tokens` are hard-split first. That bound means
    stepping back whole units always lands between 1x and 2x the overlap target, so overlap stays
    predictable. Windows do not overlap across section boundaries, where the context changes.
    """
    overlap_target = int(chunk_tokens * overlap_ratio)
    units = _build_units(document, counter, max(overlap_target, 1))

    chunks: list[Chunk] = []
    start = 0
    while start < len(units):
        section = units[start].section
        total = 0
        end = start
        while end < len(units) and total + units[end].tokens <= chunk_tokens:
            if end > start and units[end].section != section and total >= min_tokens:
                break
            total += units[end].tokens
            end += 1

        window_start = start
        if total < min_tokens and chunks:
            # A short tail borrows preceding context rather than becoming a sliver.
            while window_start > 0 and total < min_tokens:
                window_start -= 1
                total += units[window_start].tokens

        chunks.append(_make_chunk(units[window_start:end], doc_id, len(chunks), total, fingerprint))
        if end >= len(units):
            break

        if units[end].section != section:
            start = end
            continue
        step_back = end
        overlap = 0
        while step_back - 1 > start and overlap < overlap_target:
            step_back -= 1
            overlap += units[step_back].tokens
        start = step_back
    return chunks


def _build_units(
    document: ExtractedDocument, counter: TokenCounter, max_tokens: int
) -> list[_Unit]:
    pending: list[tuple[str, int, BBox, str, bool]] = []
    for page in document.pages:
        if page.skipped_reason:
            continue
        for block in page.blocks:
            is_ocr = block.source != "native"
            for sentence in split_sentences(block.text):
                pending.append((sentence, page.number, block.bbox, page.section, is_ocr))

    token_counts = counter.count_many([item[0] for item in pending])
    units: list[_Unit] = []
    for (text, page, bbox, section, is_ocr), tokens in zip(pending, token_counts, strict=True):
        if tokens <= max_tokens:
            units.append(_Unit(text, page, bbox, section, tokens, is_ocr))
            continue
        for piece in counter.split_to_limit(text, max_tokens):
            units.append(_Unit(piece, page, bbox, section, counter.count(piece), is_ocr))
    return [unit for unit in units if unit.tokens > 0]


def _make_chunk(
    units: list[_Unit], doc_id: str, ordinal: int, tokens: int, fingerprint: str
) -> Chunk:
    parts: list[str] = []
    spans: list[ChunkSpan] = []
    cursor = 0
    for unit in units:
        start = cursor
        end = start + len(unit.text)
        parts.append(unit.text)
        cursor = end + 1  # the joining space
        if spans and spans[-1].page == unit.page and spans[-1].bbox == unit.bbox:
            spans[-1].end = end
        else:
            spans.append(ChunkSpan(page=unit.page, start=start, end=end, bbox=unit.bbox))

    text = " ".join(parts)
    digest = hashlib.sha256(f"{doc_id}|{ordinal}|{fingerprint}".encode()).hexdigest()[:24]
    return Chunk(
        chunk_id=digest,
        doc_id=doc_id,
        ordinal=ordinal,
        text=text,
        section=units[0].section,
        page_start=min(unit.page for unit in units),
        page_end=max(unit.page for unit in units),
        token_count=tokens,
        language=py3langid.classify(text[:2000])[0],
        has_ocr=any(unit.has_ocr for unit in units),
        spans=spans,
    )
