from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from ragbot.domain import RetrievedChunk


class FakeTokenCounter:
    """Whitespace token counter, so chunking tests need no model download."""

    def count_many(self, texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]

    def count(self, text: str) -> int:
        return len(text.split())

    def split_to_limit(self, text: str, limit: int) -> list[str]:
        words = text.split()
        return [" ".join(words[i : i + limit]) for i in range(0, len(words), limit)]


@pytest.fixture
def counter() -> FakeTokenCounter:
    return FakeTokenCounter()


@pytest.fixture
def make_source():
    def build(number: int, text: str, page: int = 7) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=f"c{number}",
            doc_id="d1",
            filename="manual.pdf",
            title="Manual",
            section="Intro",
            text=text,
            page_start=page,
            page_end=page,
            has_ocr=False,
            spans=[{"page": page, "start": 0, "end": len(text), "bbox": [10, 10, 100, 100]}],
            source_number=number,
            used_in_context=True,
        )

    return build


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """A small PDF with a running header, a numbered footer and body text on each page."""
    path = tmp_path / "sample.pdf"
    with pymupdf.open() as pdf:
        for number in range(1, 7):
            page = pdf.new_page(width=612, height=792)
            page.insert_text((72, 40), "ACME Operations Manual", fontsize=10)
            body = (
                f"Section {number} explains how the pump number {number} is serviced safely. " * 4
            )
            page.insert_textbox(pymupdf.Rect(72, 150, 540, 500), body, fontsize=11)
            page.insert_text((300, 760), str(number), fontsize=10)
        pdf.save(path)
    return path
