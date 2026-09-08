"""``read_price_tags`` — OCR price labels and link each to the nearest product.

OCR (python-doctr) finds every price on the shelf; the detector finds the
products; this tool pairs each price with the product most likely to own it
(closest facing, preferring the one directly above the tag).
"""

from __future__ import annotations

from ..backends.matching import center
from ..backends.ocr import get_ocr
from ..schemas import AnnotatedImage, ImageInput, PriceTag, ReadPriceTagsOutput
from ._common import ToolInputError, package_annotated, prepare_image, run_detector

BBox = tuple[float, float, float, float]


def _nearest_product(tag: BBox, products: list[BBox]) -> BBox | None:
    """Product owning this price tag: closest facing, preferring one sitting
    above the tag (shelf-edge tags hang below their product)."""

    if not products:
        return None
    tx, ty = center(tag)
    tag_w = max(tag[2] - tag[0], 1.0)

    def score(p: BBox) -> tuple[int, float]:
        px, py = center(p)
        sits_above = 0 if p[3] <= ty else 1
        return sits_above, abs(px - tx) + 0.25 * abs(py - ty)

    best = min(products, key=score)
    return best if abs(center(best)[0] - tx) <= 6 * tag_w else None


def read_price_tags(params: ImageInput) -> ReadPriceTagsOutput:
    original, meta = prepare_image(params.image)

    ocr = get_ocr().run(original)
    if ocr.status == "not_implemented":
        return ReadPriceTagsOutput(
            status="not_implemented", image=meta, notes=[ocr.message or "OCR backend unavailable."]
        )
    if ocr.status == "error":
        raise ToolInputError(ocr.message or "OCR backend failed")

    products = [d.bbox for d in run_detector(original, params.conf)[0]]

    tags: list[PriceTag] = []
    parsed_meta = ocr.meta.get("parsed", [])
    for det, price in zip(ocr.detections, parsed_meta, strict=False):
        linked = _nearest_product(det.bbox, products)
        tags.append(
            PriceTag(
                text=det.label,
                price_value=price.get("value"),
                currency=price.get("currency"),
                confidence=det.confidence,
                bbox=det.bbox,
                bbox_normalized=(
                    det.bbox[0] / meta.width,
                    det.bbox[1] / meta.height,
                    det.bbox[2] / meta.width,
                    det.bbox[3] / meta.height,
                ),
                linked_product_bbox=linked,
            )
        )

    notes = [f"OCR kept {len(tags)} price-like strings out of the page."]
    if not tags:
        notes.append("No prices parsed — try a closer / sharper photo of the shelf edge.")

    annotated: AnnotatedImage | None = None
    if params.annotate and tags:
        annotated, _ = package_annotated(
            original,
            [t.bbox for t in tags],
            [
                (f"{t.price_value:.2f} {t.currency or ''}".strip() if t.price_value else t.text)
                for t in tags
            ],
            color_idx=3,
        )

    return ReadPriceTagsOutput(
        status="ok", tags=tags, image=meta, annotated=annotated, notes=notes
    )
