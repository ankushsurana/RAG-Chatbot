"""OCR accuracy: the scanned demo document is a rasterized copy of a real PDF, so its original
text layer is exact ground truth for what OCR should have read."""

from __future__ import annotations

import json
import random
import re

import pymupdf

from ragbot.config import Settings
from ragbot.evaluation.metrics import percentile
from ragbot.ingestion.ocr import OcrEngine, group_lines_into_blocks

_WORD = re.compile(r"\w+")
_SEED = 7
_MIN_TRUTH_WORDS = 40


def run_ocr_evaluation(settings: Settings, scanned_filename: str, sample_pages: int) -> dict:
    engine = OcrEngine(settings.ocr_dpi, settings.ocr_threads, settings.ocr_min_confidence)
    scores: list[float] = []
    with (
        pymupdf.open(settings.raw_dir / scanned_filename) as truth,
        pymupdf.open(settings.pdf_dir / scanned_filename) as scanned,
    ):
        candidates = [
            index
            for index in range(truth.page_count)
            if len(_WORD.findall(truth[index].get_text())) >= _MIN_TRUTH_WORDS
        ]
        for index in sorted(random.Random(_SEED).sample(candidates, sample_pages)):
            expected = _words(truth[index].get_text())
            lines = engine.read_region(scanned[index])
            found = _words(" ".join(text for text, _ in group_lines_into_blocks(lines)))
            scores.append(_word_accuracy(expected, found))

    report = {
        "document": scanned_filename,
        "pages_sampled": len(scores),
        "mean_word_accuracy": round(sum(scores) / len(scores), 4),
        "p5_word_accuracy": round(percentile(scores, 5), 4),
    }
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "ocr_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(text)]


def _word_accuracy(expected: list[str], found: list[str]) -> float:
    """Multiset word recall: insensitive to reading order, which differs legitimately in layouts."""
    remaining: dict[str, int] = {}
    for word in found:
        remaining[word] = remaining.get(word, 0) + 1
    matched = 0
    for word in expected:
        if remaining.get(word, 0) > 0:
            remaining[word] -= 1
            matched += 1
    return matched / max(len(expected), 1)
