"""``count_products`` — count items per type with positions and confidence."""

from __future__ import annotations

from collections import Counter

from ..schemas import CountProductsOutput, ImageInput
from ._common import ToolInputError, package_annotated, prepare_image, run_detector, to_items


def count_products(params: ImageInput) -> CountProductsOutput:
    original, meta = prepare_image(params.image)
    dets, result = run_detector(original, params.conf)
    if result.status == "error":
        raise ToolInputError(result.message or "detector failed")

    items = to_items(dets, meta.width, meta.height)
    counts = Counter(i.label for i in items)
    notes = list(result.meta.get("notes", []))
    notes.append(
        "Detector runs class-agnostic; every box is labelled 'product'. "
        "Fine-tune on SKU-110K for per-SKU labels."
    )

    annotated = None
    if params.annotate:
        annotated, _ = package_annotated(
            original,
            [i.bbox for i in items],
            [f"{i.label} {i.confidence:.2f}" for i in items],
        )

    return CountProductsOutput(
        total=len(items),
        counts=dict(counts),
        items=items,
        image=meta,
        annotated=annotated,
        notes=notes,
    )
