"""Per-tool input / output models. Strict Pydantic v2 — bad input becomes a clean
MCP error, never a raw exception across the protocol.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BBox = tuple[float, float, float, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- shared -----------------------------------------------------------
class ImageInput(_Strict):
    image: str = Field(..., description="Local path, http(s) URL, or base64 data URI.")
    conf: float | None = Field(None, ge=0.0, le=1.0, description="Detector confidence override.")
    annotate: bool = Field(True, description="Return an annotated image (inline or by reference).")


class DetectedItem(_Strict):
    label: str
    confidence: float
    bbox: BBox = Field(..., description="Absolute pixels [x1,y1,x2,y2].")
    bbox_normalized: BBox = Field(..., description="0..1 [x1,y1,x2,y2].")


class AnnotatedImage(_Strict):
    kind: Literal["inline", "reference"]
    png_base64: str | None = None
    uri: str | None = None


class ImageMeta(_Strict):
    width: int
    height: int
    image_id: str


# --- count_products -------------------------------------------------
class CountProductsOutput(_Strict):
    total: int
    counts: dict[str, int]
    items: list[DetectedItem]
    image: ImageMeta
    annotated: AnnotatedImage | None = None
    notes: list[str] = Field(default_factory=list)


# --- detect_gaps ---------------------------------------------------
class ShelfGap(_Strict):
    bbox: BBox
    bbox_normalized: BBox
    row: int
    width_ratio: float = Field(..., description="Gap width / mean product width in that row.")
    severity: Literal["minor", "moderate", "major"]


class DetectGapsOutput(_Strict):
    gap_count: int
    gaps: list[ShelfGap]
    rows_detected: int
    image: ImageMeta
    annotated: AnnotatedImage | None = None
    notes: list[str] = Field(default_factory=list)


# --- read_price_tags (stub) ---------------------------------------
class PriceTag(_Strict):
    text: str
    price_value: float | None
    currency: str | None
    bbox: BBox
    bbox_normalized: BBox
    linked_product_bbox: BBox | None = None


class ReadPriceTagsOutput(_Strict):
    status: Literal["ok", "not_implemented"]
    tags: list[PriceTag] = Field(default_factory=list)
    image: ImageMeta | None = None
    notes: list[str] = Field(default_factory=list)


# --- check_planogram --------------------------------------------
class PlanogramSlot(_Strict):
    label: str
    region: BBox = Field(..., description="Expected region, normalized 0..1 [x1,y1,x2,y2].")
    order: int | None = None


class PlanogramInput(ImageInput):
    slots: list[PlanogramSlot] = Field(..., min_length=1)
    iou_threshold: float = Field(0.1, ge=0.0, le=1.0)


class PlanogramDeviation(_Strict):
    type: Literal["missing", "misplaced", "extra", "wrong_order"]
    label: str
    expected_region: BBox | None = None
    observed_bbox: BBox | None = None
    detail: str


class CheckPlanogramOutput(_Strict):
    compliant: bool
    deviation_count: int
    deviations: list[PlanogramDeviation]
    image: ImageMeta
    annotated: AnnotatedImage | None = None
    notes: list[str] = Field(default_factory=list)


# --- shelf_report -----------------------------------------------
class ShelfReportInput(ImageInput):
    slots: list[PlanogramSlot] | None = Field(
        None, description="If given, the report includes a planogram check."
    )
    include_price_tags: bool = Field(False, description="Run the (stubbed) price-tag reader.")
    format: Literal["json", "markdown", "both"] = "both"


class ShelfReportOutput(_Strict):
    report: dict
    markdown: str | None = None
    image: ImageMeta
    annotated: AnnotatedImage | None = None
