"""Fetches the public demo corpus described by corpus/manifest.json."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pymupdf

from ragbot.config import Settings

MIN_PAGES = 200
_RASTER_DPI = 200
_JPEG_QUALITY = 80


@dataclass(slots=True)
class CorpusItem:
    filename: str
    pages: int
    size_mb: float
    scanned: bool


def download_corpus(
    settings: Settings, manifest_path: Path, log: Callable[[str], None] = print
) -> list[CorpusItem]:
    settings.ensure_dirs()
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]
    items: list[CorpusItem] = []
    for entry in entries:
        filename = entry["filename"]
        raw_path = settings.raw_dir / filename
        target = settings.pdf_dir / filename
        scanned = bool(entry.get("rasterize"))
        if not raw_path.exists():
            log(f"downloading {filename}")
            _download(entry["url"], raw_path)
        pages = _validate(raw_path)
        if not target.exists():
            if scanned:
                log(f"rasterizing {filename} to image-only pages")
                rasterize_pdf(raw_path, target)
            else:
                target.write_bytes(raw_path.read_bytes())
        items.append(CorpusItem(filename, pages, target.stat().st_size / 1e6, scanned))
    return items


def rasterize_pdf(source: Path, target: Path, dpi: int = _RASTER_DPI) -> None:
    """Writes a true image-only PDF, with no text layer, mimicking a scanned document."""
    with pymupdf.open(source) as pdf, pymupdf.open() as output:
        for page in pdf:
            pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
            new_page = output.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=pixmap.tobytes("jpg", _JPEG_QUALITY))
        output.save(target, garbage=3, deflate=True)


def _download(url: str, path: Path) -> None:
    temporary = path.with_suffix(".part")
    with (
        httpx.stream("GET", url, follow_redirects=True, timeout=120) as response,
        temporary.open("wb") as handle,
    ):
        response.raise_for_status()
        for block in response.iter_bytes(1 << 20):
            handle.write(block)
    temporary.replace(path)


def _validate(path: Path) -> int:
    with pymupdf.open(path) as pdf:
        if pdf.page_count < MIN_PAGES:
            raise ValueError(f"{path.name} has {pdf.page_count} pages, below {MIN_PAGES}")
        return pdf.page_count
