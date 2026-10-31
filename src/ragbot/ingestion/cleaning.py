"""Text normalization, running header/footer removal, low-value page detection, language detection."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import py3langid

from ragbot.domain import ExtractedPage, TextBlock

_MARGIN_RATIO = 0.1
_MIN_REPEATS = 4
_LANG_SAMPLE_CHARS = 20_000

_HYPHEN_BREAK = re.compile(r"(?<=[a-z])[-\u00ad]\n(?=[a-z])")
_WHITESPACE = re.compile(r"[ \t\f\v\u00a0]+")
_NEWLINES = re.compile(r"\s*\n\s*")
_DIGITS = re.compile(r"\d+")
_PAGE_NUMBER = re.compile(
    r"^\W*(?:page\s+)?(?:\d{1,5}|[ivxlcdm]{1,7})(?:\s*(?:of|/)\s*\d+)?\W*$", re.I
)
_DOT_LEADER = re.compile(r"(?:\.\s?){4,}\s*\S{0,8}\s*$")
_PUA_CODEPOINTS = f"{chr(0xE000)}-{chr(0xF8FF)}"
_PUA_GLYPH = re.compile(f"[{_PUA_CODEPOINTS}]")
_WORD_WITH_GLYPHS = re.compile(f"[a-z{_PUA_CODEPOINTS}]+")
_LIGATURE_CANDIDATES = (
    "ti",
    "tt",
    "ft",
    "tf",
    "fi",
    "fl",
    "ff",
    "ffi",
    "ffl",
    "fj",
    "st",
    "ct",
    "th",
)
_TRAILING_NUMBER = re.compile(r"(?:,\s*|\s)\d{1,5}(?:[-\u2013]\d{1,5})?\s*$")


def normalize_text(text: str) -> str:
    """NFKC (expands ligatures), repairs line-end hyphenation and collapses layout whitespace."""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad\n", "").replace("\u00ad", "")
    text = _HYPHEN_BREAK.sub("", text)
    text = _NEWLINES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def repair_private_use_glyphs(pages: list[ExtractedPage]) -> dict[str, str]:
    """Restores letters from fonts that export ligatures as Private Use glyphs ("con?ngency").

    Fonts such as Calibri can map "ti" or "tt" to Private Use code points with no Unicode meaning.
    Each glyph is resolved from the document itself: the winning ligature is the one that turns
    damaged words into the most words the document also contains intact. Returns the mapping.
    """
    blocks = [block for page in pages for block in page.blocks]
    if not any(_PUA_GLYPH.search(block.text) for block in blocks):
        return {}

    words = Counter(
        word for block in blocks for word in _WORD_WITH_GLYPHS.findall(block.text.lower())
    )
    vocabulary = {word for word in words if not _PUA_GLYPH.search(word)}
    damaged = {word: count for word, count in words.items() if _PUA_GLYPH.search(word)}
    glyphs = {glyph for word in damaged for glyph in _PUA_GLYPH.findall(word)}

    mapping: dict[str, str] = {}
    for glyph in sorted(glyphs):
        scores = {
            candidate: sum(
                count
                for word, count in damaged.items()
                if glyph in word and word.replace(glyph, candidate) in vocabulary
            )
            for candidate in _LIGATURE_CANDIDATES
        }
        best = max(_LIGATURE_CANDIDATES, key=lambda candidate: scores[candidate])
        if scores[best] > 0:
            mapping[glyph] = best

    table = str.maketrans(mapping)
    for block in blocks:
        block.text = block.text.translate(table)
    return mapping


def clean_document(pages: list[ExtractedPage]) -> tuple[int, str]:
    """Clean pages in place. Returns (number of removed boilerplate blocks, document language)."""
    removed = _remove_repeating_margins(pages)
    for page in pages:
        page.skipped_reason = _low_value_reason(page)
        normalized = (
            TextBlock(normalize_text(block.text), block.bbox, block.source) for block in page.blocks
        )
        page.blocks = [block for block in normalized if block.text]
    return removed, detect_language(_language_sample(pages))


def detect_language(text: str) -> str:
    if len(text.strip()) < 20:
        return "unknown"
    language, _ = py3langid.classify(text)
    return language


def _language_sample(pages: list[ExtractedPage]) -> str:
    collected: list[str] = []
    size = 0
    for page in pages:
        for block in page.blocks:
            collected.append(block.text)
            size += len(block.text)
        if size >= _LANG_SAMPLE_CHARS:
            break
    return " ".join(collected)[:_LANG_SAMPLE_CHARS]


def _margin_key(text: str) -> str:
    return _DIGITS.sub("#", " ".join(text.lower().split()))


def _in_margin(block: TextBlock, page: ExtractedPage) -> bool:
    margin = page.height * _MARGIN_RATIO
    return block.bbox[3] <= margin or block.bbox[1] >= page.height - margin


def _remove_repeating_margins(pages: list[ExtractedPage]) -> int:
    """Drops page numbers and any margin text that recurs (ignoring digits) across pages."""
    counts = Counter(
        _margin_key(block.text)
        for page in pages
        for block in page.blocks
        if _in_margin(block, page)
    )
    removed = 0
    for page in pages:
        kept = []
        for block in page.blocks:
            if _in_margin(block, page) and (
                _PAGE_NUMBER.match(block.text.strip())
                or counts[_margin_key(block.text)] >= _MIN_REPEATS
            ):
                removed += 1
            else:
                kept.append(block)
        page.blocks = kept
    return removed


def _low_value_reason(page: ExtractedPage) -> str | None:
    """Tables of contents and back-of-book indexes attract false matches without answering much."""
    lines = [
        line.strip() for block in page.blocks for line in block.text.splitlines() if line.strip()
    ]
    if len(lines) < 8:
        return None
    dot_leaders = sum(bool(_DOT_LEADER.search(line)) for line in lines)
    if dot_leaders / len(lines) >= 0.3:
        return "table_of_contents"
    numbered = sum(bool(_TRAILING_NUMBER.search(line)) and len(line) < 70 for line in lines)
    if numbered / len(lines) >= 0.6:
        return "index_or_listing"
    return None
