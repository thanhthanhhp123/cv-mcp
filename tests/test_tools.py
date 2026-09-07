"""Tool-level tests.

Detector-backed tools use a fake detector injected via monkeypatch so the
geometry / packaging logic is tested without downloading weights. One optional
test exercises the real YOLO model when weights are present.
"""

from __future__ import annotations

import pytest

from vision_mcp.backends.base import Detection, Result
from vision_mcp.schemas import ImageInput, PlanogramInput, PlanogramSlot, ShelfReportInput
from vision_mcp.tools import _common
from vision_mcp.tools.check_planogram import check_planogram
from vision_mcp.tools.count_products import count_products
from vision_mcp.tools.detect_gaps import detect_gaps
from vision_mcp.tools.read_price_tags import read_price_tags
from vision_mcp.tools.shelf_report import shelf_report


@pytest.fixture
def fake_detector(monkeypatch, synthetic_shelf):
    """Detections matching the synthetic_shelf fixture (top row has a gap)."""

    top = [(40, 60), (160, 60), (280, 60), (520, 60), (640, 60)]
    bot = [(40, 340), (160, 340), (280, 340), (400, 340), (520, 340), (640, 340)]
    dets = [
        Detection("product", 0.9, (x, y, x + 100, y + 180)) for x, y in (*top, *bot)
    ]

    def _run(original, conf):
        return dets, Result(status="ok", detections=dets, meta={"notes": []})

    monkeypatch.setattr(_common, "run_detector", _run)
    # tools import run_detector by name into their own module namespace
    for mod in ("count_products", "detect_gaps", "check_planogram"):
        monkeypatch.setattr(f"vision_mcp.tools.{mod}.run_detector", _run)
    return dets


def test_count_products(fake_detector, synthetic_shelf_datauri):
    out = count_products(ImageInput(image=synthetic_shelf_datauri))
    assert out.total == 11
    assert out.counts == {"product": 11}
    assert out.items[0].bbox_normalized[0] == pytest.approx(40 / 800)
    assert out.annotated.kind == "inline"


def test_detect_gaps_finds_top_row_gap(fake_detector, synthetic_shelf_datauri):
    out = detect_gaps(ImageInput(image=synthetic_shelf_datauri))
    assert out.rows_detected == 2
    assert out.gap_count >= 1
    assert any(g.row == 0 for g in out.gaps)


def test_check_planogram_missing_slot(fake_detector, synthetic_shelf_datauri):
    slots = [
        PlanogramSlot(label="present", region=(0.02, 0.08, 0.18, 0.42), order=0),
        PlanogramSlot(label="absent", region=(0.45, 0.02, 0.6, 0.18), order=1),
    ]
    out = check_planogram(
        PlanogramInput(image=synthetic_shelf_datauri, slots=slots, iou_threshold=0.1)
    )
    assert not out.compliant
    assert any(d.type == "missing" and d.label == "absent" for d in out.deviations)


def test_read_price_tags_is_stub(synthetic_shelf_datauri):
    out = read_price_tags(ImageInput(image=synthetic_shelf_datauri))
    assert out.status == "not_implemented"
    assert out.tags == []
    assert out.notes


def test_shelf_report_combines(fake_detector, synthetic_shelf_datauri):
    out = shelf_report(
        ShelfReportInput(image=synthetic_shelf_datauri, include_price_tags=True, format="both")
    )
    assert out.report["count_products"]["total"] == 11
    assert "detect_gaps" in out.report
    assert out.report["read_price_tags"]["status"] == "not_implemented"
    assert out.markdown.startswith("# Shelf audit report")


def test_bad_image_raises_tool_input_error(fake_detector):
    from vision_mcp.tools._common import ToolInputError

    with pytest.raises(ToolInputError):
        count_products(ImageInput(image="no/such/file.png"))


def test_real_detector_smoke(require_detector, synthetic_shelf_datauri):
    out = count_products(ImageInput(image=synthetic_shelf_datauri))
    assert out.total >= 0  # model runs without raising
