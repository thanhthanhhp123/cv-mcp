"""Shared fixtures. Synthetic images keep the core suite fast and dependency-free;
detector-backed tests skip themselves when weights are absent.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from vision_mcp.config import get_settings
from vision_mcp.imaging import encode_png

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def synthetic_shelf() -> np.ndarray:
    """A 600x800 RGB image: two rows of pale blocks with one obvious gap in the
    top row. Not a real photo — enough for geometry / plumbing tests."""

    img = np.full((600, 800, 3), 235, dtype=np.uint8)
    xs_top = [40, 160, 280, 520, 640]  # missing block around x=400 -> a gap
    xs_bot = [40, 160, 280, 400, 520, 640]
    for y0 in (60, 340):
        xs = xs_top if y0 == 60 else xs_bot
        for x0 in xs:
            img[y0 : y0 + 180, x0 : x0 + 100] = (120, 150, 200)
    return img


@pytest.fixture
def synthetic_shelf_datauri(synthetic_shelf: np.ndarray) -> str:
    b64 = base64.b64encode(encode_png(synthetic_shelf)).decode("ascii")
    return f"data:image/png;base64,{b64}"


@pytest.fixture
def detector_available() -> bool:
    return get_settings().detector_path.is_file()


@pytest.fixture
def require_detector() -> None:
    if not get_settings().detector_path.is_file():
        pytest.skip("detector weights not downloaded (run scripts/download_models.py)")
