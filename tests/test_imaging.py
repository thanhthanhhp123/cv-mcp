from __future__ import annotations

import base64

import numpy as np
import pytest

from vision_mcp.imaging import (
    ImageLoadError,
    encode_png,
    load_image,
    resize_max_edge,
    to_base64_png,
)


def test_load_from_path(tmp_path, synthetic_shelf):
    p = tmp_path / "shelf.png"
    p.write_bytes(encode_png(synthetic_shelf))
    out = load_image(str(p))
    assert out.shape == synthetic_shelf.shape


def test_load_from_data_uri(synthetic_shelf_datauri, synthetic_shelf):
    out = load_image(synthetic_shelf_datauri)
    assert out.shape == synthetic_shelf.shape


def test_load_from_bare_base64(synthetic_shelf):
    raw = base64.b64encode(encode_png(synthetic_shelf)).decode("ascii")
    out = load_image(raw)
    assert out.shape[2] == 3


def test_load_missing_file():
    with pytest.raises(ImageLoadError):
        load_image("no/such/file.png")


def test_load_bad_base64():
    with pytest.raises(ImageLoadError):
        load_image("data:image/png;base64,not-valid-!!!")


def test_resize_max_edge_downscales():
    big = np.zeros((3000, 1500, 3), dtype=np.uint8)
    out, scale = resize_max_edge(big, 1920)
    assert max(out.shape[:2]) == 1920
    assert scale > 1.0


def test_resize_noop_when_small(synthetic_shelf):
    out, scale = resize_max_edge(synthetic_shelf, 1920)
    assert scale == 1.0
    assert out is synthetic_shelf


def test_to_base64_png_roundtrips(synthetic_shelf):
    b64 = to_base64_png(synthetic_shelf)
    assert load_image(f"data:image/png;base64,{b64}").shape == synthetic_shelf.shape
