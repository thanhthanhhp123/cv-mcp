"""OCR backend tests.

``parse_price`` is a pure function — always tested. The doctr-backed path only
runs when the optional ``ocr`` extra is installed.
"""

from __future__ import annotations

import pytest

from vision_mcp.backends.ocr import parse_price


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("€5,40", (5.40, "EUR")),
        ("1.19", (1.19, None)),
        ("7.65", (7.65, None)),
        ("3.50 EUR", (3.50, "EUR")),
        ("$2.00", (2.00, "USD")),
        ("£12,99", (12.99, "GBP")),
        ("  0,99 € ", (0.99, "EUR")),
    ],
)
def test_parse_price_hits(text, expected):
    assert parse_price(text) == expected


@pytest.mark.parametrize("text", ["", "Riesling 2007", "PLU 935", "42", "aisle 6", "600 g"])
def test_parse_price_misses(text):
    assert parse_price(text) is None


def test_ocr_backend_not_implemented_without_extra():
    from vision_mcp.backends.ocr import OcrBackend

    try:
        import doctr  # noqa: F401
    except ImportError:
        import numpy as np

        res = OcrBackend().run(np.zeros((32, 32, 3), dtype=np.uint8))
        assert res.status == "not_implemented"
        assert "ocr" in (res.message or "").lower()
    else:
        pytest.skip("ocr extra installed")
