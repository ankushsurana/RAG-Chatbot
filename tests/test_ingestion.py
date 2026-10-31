from __future__ import annotations

from itertools import pairwise

import pymupdf
import pytest

from ragbot.config import Settings
from ragbot.domain import ExtractedDocument, ExtractedPage, TextBlock
from ragbot.ingestion.chunking import build_chunks, split_sentences
from ragbot.ingestion.cleaning import (
    clean_document,
    detect_language,
    normalize_text,
    repair_private_use_glyphs,
)
from ragbot.ingestion.corpus import rasterize_pdf
from ragbot.ingestion.extract import extract_document


def _document(page_texts: list[str], section: str = "Intro") -> ExtractedDocument:
    pages = [
        ExtractedPage(
            number=index + 1,
            width=612,
            height=792,
            section=section,
            blocks=[TextBlock(text, (72, 100, 540, 700))],
        )
        for index, text in enumerate(page_texts)
    ]
    return ExtractedDocument("doc.pdf", "Doc", len(pages), "en", pages)


def _long_text(start: int, sentences: int) -> str:
    return " ".join(
        f"Sentence {start + i} has exactly seven plain words here." for i in range(sentences)
    )


class TestNormalization:
    def test_repairs_hyphenation_and_ligatures(self):
        assert normalize_text("the of\ufb01ce exam-\nple\nnext") == "the office example next"

    def test_keeps_real_hyphens(self):
        assert normalize_text("well-known\nfact") == "well-known fact"

    def test_detects_language(self):
        assert (
            detect_language("The quick brown fox jumps over the lazy dog near the river.") == "en"
        )
        assert (
            detect_language("Der schnelle braune Fuchs springt \u00fcber den faulen Hund am Fluss.")
            == "de"
        )


class TestBoilerplate:
    def test_removes_repeating_margin_text_and_page_numbers(self):
        pages = [
            ExtractedPage(
                number=n,
                width=612,
                height=792,
                blocks=[
                    TextBlock(f"Chapter header page {n}", (72, 20, 400, 40)),
                    TextBlock(f"Unique body text number {n}.", (72, 200, 400, 300)),
                    TextBlock(str(n), (300, 760, 320, 780)),
                ],
            )
            for n in range(1, 7)
        ]
        removed, _ = clean_document(pages)
        assert removed == 12
        assert all(len(page.blocks) == 1 and "body" in page.blocks[0].text for page in pages)

    def test_margin_text_that_does_not_repeat_is_kept(self):
        page = ExtractedPage(
            number=1,
            width=612,
            height=792,
            blocks=[TextBlock("A one-off title", (72, 20, 400, 40))],
        )
        clean_document([page])
        assert page.blocks[0].text == "A one-off title"

    def test_flags_table_of_contents_pages(self):
        lines = "\n".join(f"Section {n} . . . . . . . . {n * 3}" for n in range(1, 12))
        page = ExtractedPage(
            number=1, width=612, height=792, blocks=[TextBlock(lines, (72, 100, 500, 700))]
        )
        clean_document([page])
        assert page.skipped_reason == "table_of_contents"


class TestGlyphRepair:
    def test_resolves_ligature_glyphs_from_document_vocabulary(self):
        ti, tt = chr(0xE000), chr(0xE001)
        damaged = f"The con{ti}ngency plan and informa{ti}on at a{tt}ached ma{tt}ers."
        intact = "CONTINGENCY INFORMATION ATTACHED MATTERS."
        pages = [
            ExtractedPage(
                number=1,
                width=612,
                height=792,
                blocks=[TextBlock(damaged, (0, 0, 1, 1)), TextBlock(intact, (0, 2, 1, 3))],
            )
        ]
        mapping = repair_private_use_glyphs(pages)
        assert mapping == {ti: "ti", tt: "tt"}
        assert (
            pages[0].blocks[0].text == "The contingency plan and information at attached matters."
        )

    def test_clean_text_is_untouched(self):
        pages = [
            ExtractedPage(
                number=1, width=612, height=792, blocks=[TextBlock("Clean.", (0, 0, 1, 1))]
            )
        ]
        assert repair_private_use_glyphs(pages) == {}
        assert pages[0].blocks[0].text == "Clean."


class TestSentences:
    def test_does_not_split_on_abbreviations(self):
        text = "See Fig. 3 for details, e.g. Table 2. The next sentence starts here."
        assert split_sentences(text) == [
            "See Fig. 3 for details, e.g. Table 2.",
            "The next sentence starts here.",
        ]


class TestChunking:
    def test_is_deterministic(self, counter):
        document = _document([_long_text(0, 200)])
        first = build_chunks(document, "d", counter, 100, 0.15, 10, "fp")
        second = build_chunks(document, "d", counter, 100, 0.15, 10, "fp")
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [c.text for c in first] == [c.text for c in second]

    def test_respects_size_limit_and_overlap_range(self, counter):
        document = _document([_long_text(0, 300)])
        chunks = build_chunks(document, "d", counter, 100, 0.15, 10, "fp")
        assert len(chunks) > 3
        assert all(chunk.token_count <= 100 for chunk in chunks)
        for current, following in pairwise(chunks):
            sentences = set(split_sentences(current.text))
            shared = sum(len(s.split()) for s in split_sentences(following.text) if s in sentences)
            assert 0.10 <= shared / current.token_count <= 0.30

    def test_tracks_pages_across_chunks(self, counter):
        document = _document([_long_text(0, 10), _long_text(100, 10), _long_text(200, 10)])
        chunks = build_chunks(document, "d", counter, 100, 0.15, 10, "fp")
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 2
        assert chunks[-1].page_end == 3
        for chunk in chunks:
            for span in chunk.spans:
                assert chunk.page_start <= span.page <= chunk.page_end
                assert chunk.text[span.start : span.end].startswith("Sentence")

    def test_oversized_sentence_is_split(self, counter):
        document = _document([" ".join(f"word{i}" for i in range(500))])
        chunks = build_chunks(document, "d", counter, 100, 0.15, 10, "fp")
        assert all(chunk.token_count <= 100 for chunk in chunks)
        joined = " ".join(chunk.text for chunk in chunks)
        assert all(f"word{i}" in joined for i in range(500))

    def test_new_section_starts_a_new_chunk_without_overlap(self, counter):
        first = _document([_long_text(0, 10)], "Alpha").pages
        second = _document([_long_text(100, 10)], "Beta").pages
        second[0].number = 2
        document = ExtractedDocument("d.pdf", "D", 2, "en", first + second)
        chunks = build_chunks(document, "d", counter, 200, 0.15, 10, "fp")
        assert [chunk.section for chunk in chunks] == ["Alpha", "Beta"]
        assert "Sentence 9 " not in chunks[1].text

    def test_skipped_pages_are_not_chunked(self, counter):
        document = _document(["Contents listing here.", _long_text(0, 10)])
        document.pages[0].skipped_reason = "table_of_contents"
        chunks = build_chunks(document, "d", counter, 100, 0.15, 10, "fp")
        assert "Contents" not in " ".join(chunk.text for chunk in chunks)


class TestExtraction:
    def test_native_pdf_strips_margins_and_keeps_pages(self, sample_pdf, tmp_path):
        settings = Settings(data_dir=tmp_path / "data", ocr_enabled=False)
        document = extract_document(sample_pdf, settings)
        assert document.page_count == 6
        assert document.ocr_pages == 0
        assert document.removed_boilerplate_blocks >= 6
        page_two = " ".join(block.text for block in document.pages[1].blocks)
        assert "pump number 2" in page_two
        assert "ACME Operations" not in page_two

    @pytest.mark.integration
    def test_scanned_pdf_is_read_with_ocr(self, sample_pdf, tmp_path):
        scanned = tmp_path / "scanned.pdf"
        rasterize_pdf(sample_pdf, scanned)
        with pymupdf.open(scanned) as pdf:
            assert not pdf[0].get_text().strip()
        settings = Settings(data_dir=tmp_path / "data")
        document = extract_document(scanned, settings)
        assert document.ocr_pages == 6
        text = " ".join(block.text for block in document.pages[2].blocks)
        assert "pump number 3" in text
        assert all(block.source == "ocr_page" for block in document.pages[2].blocks)
