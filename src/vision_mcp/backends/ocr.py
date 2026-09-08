"""Price-tag OCR backend, built on `python-doctr <https://github.com/mindee/doctr>`_.

Runs full-page OCR, then keeps only the text that parses as a price. The
``read_price_tags`` tool takes it from there — associating each price with the
nearest product from the detector.

The ``ocr`` extra is optional: without it, ``run`` returns
``status="not_implemented"`` with an install hint rather than crashing.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .base import BaseVisionModel, Detection, Result

_MISSING_DEP = (
    "Price-tag OCR needs the 'ocr' extra: `pip install \"shelf-auditor-mcp[ocr]\"` "
    "(or `uv sync --extra ocr`)."
)

_CUR = {"€": "EUR", "$": "USD", "£": "GBP", "EUR": "EUR", "USD": "USD", "GBP": "GBP", "KR": "SEK"}

# 12,99 | 1.19 | €5.40 | 3.50 EUR | £2 . A decimal part or a currency token is
# required, so bare integers (shelf counts, PLU codes) don't match.
_PRICE_RE = re.compile(
    r"(?P<sym>[€$£])?\s?(?P<int>\d{1,4})(?:[.,](?P<dec>\d{2}))?\s?(?P<word>EUR|USD|GBP|KR|€|\$|£)?",
    re.IGNORECASE,
)


def parse_price(text: str) -> tuple[float, str | None] | None:
    """``"€5,40"`` -> ``(5.40, "EUR")``. Returns None if it isn't a price."""

    cleaned = re.sub(r"\s+", " ", text.strip())
    m = _PRICE_RE.fullmatch(cleaned) or _PRICE_RE.search(cleaned)
    if not m:
        return None

    currency = None
    for token in (m.group("sym"), m.group("word")):
        if token and token.upper() in _CUR:
            currency = _CUR[token.upper()]
            break

    if m.group("dec") is None and currency is None:
        return None  # a bare integer with no currency is just a number

    value = float(m.group("int"))
    if m.group("dec") is not None:
        value += int(m.group("dec")) / 100
    if not 0.01 <= value <= 9999:
        return None
    return round(value, 2), currency


class OcrBackend(BaseVisionModel):
    name = "doctr-ocr"

    def _load(self) -> Any:
        try:
            from doctr.models import ocr_predictor
        except ImportError:
            return None
        return ocr_predictor(pretrained=True)

    def _infer(self, image: np.ndarray, params: dict[str, Any]) -> Result:
        if self._model is None:
            return Result(status="not_implemented", message=_MISSING_DEP)

        h, w = image.shape[:2]
        page = self._model([image]).pages[0]

        detections: list[Detection] = []
        parsed: list[dict[str, Any]] = []

        def _add(text: str, conf: float, geom: Any) -> bool:
            price = parse_price(text)
            if price is None:
                return False
            (gx1, gy1), (gx2, gy2) = geom
            detections.append(
                Detection(
                    label=text,
                    confidence=round(float(conf), 4),
                    bbox=(gx1 * w, gy1 * h, gx2 * w, gy2 * h),
                )
            )
            parsed.append({"value": price[0], "currency": price[1]})
            return True

        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join(word.value for word in line.words)
                line_conf = min((word.confidence for word in line.words), default=0.0)
                if not _add(line_text, line_conf, line.geometry):
                    for word in line.words:  # line didn't parse — try single words
                        _add(word.value, word.confidence, word.geometry)

        return Result(status="ok", detections=detections, meta={"parsed": parsed})


_ocr: OcrBackend | None = None


def get_ocr() -> OcrBackend:
    global _ocr
    if _ocr is None:
        _ocr = OcrBackend()
    return _ocr
