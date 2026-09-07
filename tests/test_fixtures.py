"""Integration tests on real CC-licensed shelf photos.

Skipped unless both the detector weights and the fixture images are present:

    uv run python scripts/download_models.py           # or fetch the SKU-110K fine-tune
    uv run python tests/fixtures/download_fixtures.py

With the SKU-110K fine-tune in models/ (scripts/hpc/), the detector finds packaged
goods COCO cannot; the thresholds below hold for either weight set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vision_mcp.config import get_settings
from vision_mcp.schemas import ImageInput
from vision_mcp.tools.count_products import count_products
from vision_mcp.tools.detect_gaps import detect_gaps

FIXTURES = Path(__file__).parent / "fixtures"
_MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
_ITEMS = {item["file"]: item for item in _MANIFEST["fixtures"]}


def _need(file: str) -> Path:
    s = get_settings()
    if not s.detector_path.is_file() and s.detector_weights.startswith("yolov8s_sku110k"):
        # falls back to auto-downloaded yolov8n.pt, which is fine for these tests
        pass
    path = FIXTURES / file
    if not path.is_file():
        pytest.skip(f"{file} not downloaded (run tests/fixtures/download_fixtures.py)")
    return path


def test_count_products_on_wine_shelf():
    _need("wine_shelf.jpg")
    out = count_products(ImageInput(image=str(FIXTURES / "wine_shelf.jpg")))
    assert out.total >= _ITEMS["wine_shelf.jpg"]["min_bottles"]
    assert out.counts == {"product": out.total}
    for item in out.items:
        x1, y1, x2, y2 = item.bbox
        assert 0 <= x1 < x2 <= out.image.width
        assert 0 <= y1 < y2 <= out.image.height
        assert all(0.0 <= v <= 1.0 for v in item.bbox_normalized)
    assert out.annotated is not None


@pytest.mark.parametrize("file", ["wine_shelf.jpg", "bread_shelf.jpg", "empty_shelf.jpg"])
def test_tools_never_raise_on_real_photos(file):
    path = _need(file)
    params = ImageInput(image=str(path), annotate=False)
    c = count_products(params)
    g = detect_gaps(params)
    assert c.image.width > 0 and g.image.width == c.image.width
    assert g.gap_count == len(g.gaps)
