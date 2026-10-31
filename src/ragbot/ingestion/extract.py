"""Page-level text extraction. Native text is preferred; OCR handles scans and embedded images."""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import asdict
from pathlib import Path

import pymupdf

from ragbot.config import Settings
from ragbot.domain import ExtractedDocument, ExtractedPage, TextBlock
from ragbot.ingestion.cleaning import clean_document, repair_private_use_glyphs
from ragbot.ingestion.ocr import OcrEngine, group_lines_into_blocks

_MIN_IMAGE_TEXT_CHARS = 30
_MIN_IMAGE_TEXT_ALPHA_RATIO = 0.6


def extract_document(path: Path, settings: Settings) -> ExtractedDocument:
    """Extract and clean one PDF. Importable at module level so it can run in a worker process."""
    started = time.perf_counter()
    pymupdf.TOOLS.mupdf_display_errors(False)
    with pymupdf.open(path) as pdf:
        if pdf.is_encrypted and not pdf.authenticate(""):
            raise ValueError(f"{path.name} is password protected")
        toc = pdf.get_toc(simple=True)
        title = _resolve_title(pdf, path)
        ocr = _build_ocr(pdf, settings)
        seen_images: set[int] = set()
        section_for_page = _section_lookup(toc)
        pages = [_extract_page(page, ocr, settings, seen_images, section_for_page) for page in pdf]
        page_count = pdf.page_count

    removed, language = clean_document(pages)
    return ExtractedDocument(
        filename=path.name,
        title=title,
        page_count=page_count,
        language=language,
        pages=pages,
        extract_seconds=time.perf_counter() - started,
        removed_boilerplate_blocks=removed,
    )


def load_or_extract(path: Path, sha256: str, settings: Settings) -> ExtractedDocument:
    """Extraction (especially OCR) is the slow step, so its output is cached by content hash."""
    cache_path = settings.cache_dir / f"{sha256[:24]}.{settings.extraction_fingerprint}.json.gz"
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            document = _document_from_dict(json.load(handle))
        document.filename = path.name
    else:
        document = extract_document(path, settings)
        temporary = cache_path.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
            json.dump(asdict(document), handle)
        temporary.replace(cache_path)
    repair_private_use_glyphs(document.pages)
    return document


def _build_ocr(pdf: pymupdf.Document, settings: Settings) -> OcrEngine | None:
    # Loading OCR models is skipped for PDFs that contain no images at all.
    if not settings.ocr_enabled:
        return None
    if not any(pdf.get_page_images(index, full=False) for index in range(pdf.page_count)):
        return None
    return OcrEngine(settings.ocr_dpi, settings.ocr_threads, settings.ocr_min_confidence)


def _extract_page(
    page: pymupdf.Page,
    ocr: OcrEngine | None,
    settings: Settings,
    seen_images: set[int],
    section_for_page,
) -> ExtractedPage:
    number = page.number + 1
    rect = page.rect
    extracted = ExtractedPage(
        number=number, width=rect.width, height=rect.height, section=section_for_page(number)
    )

    native = [
        TextBlock(text=block[4], bbox=tuple(block[:4]))
        for block in page.get_text("blocks")
        if block[6] == 0 and block[4].strip()
    ]
    native_text = "".join(block.text for block in native)
    images = page.get_image_info(xrefs=True) if ocr is not None else []
    page_area = max(rect.width * rect.height, 1.0)
    coverage = sum(_area(image["bbox"], rect) for image in images) / page_area

    needs_page_ocr = (
        ocr is not None
        and coverage >= settings.ocr_page_min_image_coverage
        and (
            len(native_text.strip()) < settings.ocr_page_min_native_chars
            or _is_garbled(native_text)
        )
    )
    if needs_page_ocr:
        extracted.ocr_used = True
        extracted.blocks = [
            TextBlock(text=text, bbox=bbox, source="ocr_page")
            for text, bbox in group_lines_into_blocks(ocr.read_region(page))
        ]
        return extracted

    extracted.blocks = native
    if ocr is None:
        return extracted

    for image in images:
        xref = image.get("xref", 0)
        if xref and xref in seen_images:
            continue  # logos and repeated artwork are read once
        if _area(image["bbox"], rect) / page_area < settings.ocr_image_min_coverage:
            continue
        seen_images.add(xref)
        clip = pymupdf.Rect(image["bbox"]) & rect
        lines = ocr.read_region(page, clip=clip)
        text = "\n".join(line.text for line in lines)
        if _looks_like_prose(text):
            bbox = tuple(clip)
            extracted.blocks.append(TextBlock(text=text, bbox=bbox, source="ocr_image"))
            extracted.image_ocr_blocks += 1
    return extracted


def _area(bbox, page_rect: pymupdf.Rect) -> float:
    clipped = pymupdf.Rect(bbox) & page_rect
    return 0.0 if clipped.is_empty else clipped.width * clipped.height


def _is_garbled(text: str) -> bool:
    """Detects broken text layers (bad font encodings) which should be OCRed instead."""
    sample = text[:4000]
    if not sample.strip():
        return False
    bad = sum(ch == "\ufffd" or (ord(ch) < 32 and ch not in "\n\t\r") for ch in sample)
    alpha = sum(ch.isalpha() for ch in sample)
    return bad / len(sample) > 0.05 or alpha / len(sample) < 0.25


def _looks_like_prose(text: str) -> bool:
    """Rejects OCR noise from charts and photos: axis ticks, stray glyphs and so on."""
    compact = "".join(text.split())
    if len(compact) < _MIN_IMAGE_TEXT_CHARS:
        return False
    return sum(ch.isalpha() for ch in compact) / len(compact) >= _MIN_IMAGE_TEXT_ALPHA_RATIO


def _resolve_title(pdf: pymupdf.Document, path: Path) -> str:
    # PDF title metadata is often junk ("Microsoft Word - draft3.doc"), so only trust clean values.
    title = (pdf.metadata or {}).get("title", "").strip()
    junk = ("untitled", "microsoft word", ".doc", ".pdf", ".indd", ".qxd")
    if 8 <= len(title) <= 150 and not any(marker in title.lower() for marker in junk):
        return title
    return path.stem.replace("_", " ")


def _section_lookup(toc: list[list]):
    """Returns a function mapping a page number to the deepest table-of-contents entry above it."""
    entries = sorted(
        ((entry[2], entry[1].strip()) for entry in toc if entry[2] > 0 and entry[1].strip()),
        key=lambda item: item[0],
    )

    def lookup(page_number: int) -> str:
        section = ""
        for start, title in entries:
            if start > page_number:
                break
            section = title
        return section[:160]

    return lookup


def _document_from_dict(raw: dict) -> ExtractedDocument:
    pages = [
        ExtractedPage(
            **{
                **page,
                "blocks": [
                    TextBlock(block["text"], tuple(block["bbox"]), block["source"])
                    for block in page["blocks"]
                ],
            }
        )
        for page in raw.pop("pages")
    ]
    return ExtractedDocument(pages=pages, **raw)
