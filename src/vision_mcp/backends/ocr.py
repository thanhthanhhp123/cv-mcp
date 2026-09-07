"""Price-tag OCR backend — STUB.

The interface is complete so ``read_price_tags`` and ``shelf_report`` can be
built and tested end-to-end. Wiring a real engine (``python-doctr`` or PaddleOCR)
means filling in ``_load`` / ``_infer`` here and adding the dep — nothing in the
MCP layer changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseVisionModel, Result

_STUB_MESSAGE = (
    "OCR backend is not implemented yet. Install the 'ocr' extra and implement "
    "vision_mcp.backends.ocr.OcrBackend._load/_infer (python-doctr recommended)."
)


class OcrBackend(BaseVisionModel):
    name = "ocr-stub"

    def _load(self) -> Any:
        return object()  # nothing to load for the stub

    def _infer(self, image: np.ndarray, params: dict[str, Any]) -> Result:
        return Result(status="not_implemented", message=_STUB_MESSAGE)


_ocr: OcrBackend | None = None


def get_ocr() -> OcrBackend:
    global _ocr
    if _ocr is None:
        _ocr = OcrBackend()
    return _ocr
