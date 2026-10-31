"""OCR for scanned pages and embedded images, using RapidOCR (Apache-2.0, PP-OCR models on ONNX)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymupdf
from rapidocr import RapidOCR

from ragbot.domain import BBox


@dataclass(slots=True)
class OcrLine:
    text: str
    bbox: BBox  # in PDF points


class OcrEngine:
    def __init__(self, dpi: int, threads: int, min_confidence: float) -> None:
        self._dpi = dpi
        self._min_confidence = min_confidence
        self._engine = RapidOCR(
            params={
                "Global.log_level": "error",
                "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )

    def read_region(self, page: pymupdf.Page, clip: pymupdf.Rect | None = None) -> list[OcrLine]:
        """Render a page (or a region of it) and return its recognized lines in PDF coordinates."""
        pixmap = page.get_pixmap(dpi=self._dpi, clip=clip, colorspace=pymupdf.csRGB, alpha=False)
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8)
        image = pixels.reshape(pixmap.height, pixmap.width, 3)[:, :, ::-1]  # RGB to BGR

        result = self._engine(image)
        if result.boxes is None or result.txts is None:
            return []

        scale = 72.0 / self._dpi
        origin_x, origin_y = (clip.x0, clip.y0) if clip is not None else (0.0, 0.0)
        lines: list[OcrLine] = []
        for box, text, score in zip(result.boxes, result.txts, result.scores, strict=True):
            if score < self._min_confidence or not text.strip():
                continue
            xs, ys = box[:, 0], box[:, 1]
            lines.append(
                OcrLine(
                    text=text.strip(),
                    bbox=(
                        origin_x + float(xs.min()) * scale,
                        origin_y + float(ys.min()) * scale,
                        origin_x + float(xs.max()) * scale,
                        origin_y + float(ys.max()) * scale,
                    ),
                )
            )
        return lines


def group_lines_into_blocks(lines: list[OcrLine]) -> list[tuple[str, BBox]]:
    """Merge OCR lines into paragraphs using vertical gaps, assuming top-to-bottom reading order."""
    if not lines:
        return []

    ordered = sorted(lines, key=lambda line: (round(line.bbox[1] / 4), line.bbox[0]))
    blocks: list[tuple[str, BBox]] = []
    current = [ordered[0]]
    for line in ordered[1:]:
        previous = current[-1]
        height = max(previous.bbox[3] - previous.bbox[1], 1.0)
        gap = line.bbox[1] - previous.bbox[3]
        same_row = abs(line.bbox[1] - previous.bbox[1]) < height * 0.5
        if same_row or gap <= height * 0.8:
            current.append(line)
        else:
            blocks.append(_merge(current))
            current = [line]
    blocks.append(_merge(current))
    return blocks


def _merge(lines: list[OcrLine]) -> tuple[str, BBox]:
    text = "\n".join(line.text for line in lines)
    bbox = (
        min(line.bbox[0] for line in lines),
        min(line.bbox[1] for line in lines),
        max(line.bbox[2] for line in lines),
        max(line.bbox[3] for line in lines),
    )
    return text, bbox
