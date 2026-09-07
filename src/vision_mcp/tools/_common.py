"""Plumbing shared by the tool modules: load an image, run the detector, and
package annotated output within the payload limit.
"""

from __future__ import annotations

import uuid

import numpy as np

from ..backends.base import Detection, Result
from ..backends.detector import get_detector
from ..config import get_settings
from ..imaging import (
    ImageLoadError,
    annotate,
    encode_png,
    load_image,
    resize_max_edge,
    to_base64_png,
)
from ..schemas import AnnotatedImage, DetectedItem, ImageMeta
from ..storage import get_storage


class ToolInputError(ValueError):
    """Bad tool input — the server converts this into a clean MCP error."""


def prepare_image(spec: str) -> tuple[np.ndarray, ImageMeta]:
    """Load, cache the original, and return an inference-ready (resized) copy
    plus metadata. Boxes from inference are in *resized* pixels; callers scale
    back with the returned scale via :func:`run_detector`."""

    try:
        original = load_image(spec)
    except ImageLoadError as exc:
        raise ToolInputError(str(exc)) from exc

    storage = get_storage()
    image_id = storage.put_image(original)
    h, w = original.shape[:2]
    return original, ImageMeta(width=w, height=h, image_id=image_id)


def run_detector(original: np.ndarray, conf: float | None) -> tuple[list[Detection], Result]:
    """Run the detector on a resized copy and rescale boxes to original pixels."""

    resized, scale = resize_max_edge(original)
    result = get_detector().run(resized, {"conf": conf})
    if scale != 1.0:
        result.detections = [
            Detection(
                label=d.label,
                confidence=d.confidence,
                bbox=(d.bbox[0] * scale, d.bbox[1] * scale, d.bbox[2] * scale, d.bbox[3] * scale),
            )
            for d in result.detections
        ]
    return result.detections, result


def to_items(dets: list[Detection], width: int, height: int) -> list[DetectedItem]:
    return [
        DetectedItem(
            label=d.label,
            confidence=d.confidence,
            bbox=d.bbox,
            bbox_normalized=d.normalized(width, height),
        )
        for d in dets
    ]


def package_annotated(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    labels: list[str] | None,
    *,
    color_idx: int = 0,
) -> tuple[AnnotatedImage, str]:
    """Return an inline base64 image, or spill to disk and return a reference
    when it exceeds ``max_inline_image_bytes``. Second item is the job_id used
    for any ``annotated://`` reference."""

    job_id = uuid.uuid4().hex
    canvas = annotate(image, boxes, labels, color_idx=color_idx)
    png = encode_png(canvas)
    if len(png) <= get_settings().max_inline_image_bytes:
        return AnnotatedImage(kind="inline", png_base64=to_base64_png(canvas)), job_id

    uri = get_storage().put_annotated(job_id, canvas)
    return AnnotatedImage(kind="reference", uri=uri), job_id
