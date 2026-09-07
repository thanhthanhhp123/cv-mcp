"""``detect_gaps`` — locate empty regions on the shelf (out-of-stock).

Heuristic: cluster detections into rows, then within each row scan the
horizontal span for gaps between consecutive products that are wide relative to
the row's mean product width.
"""

from __future__ import annotations

from typing import Literal

from ..backends.matching import cluster_rows
from ..schemas import DetectGapsOutput, ImageInput, ShelfGap
from ._common import ToolInputError, package_annotated, prepare_image, run_detector

_MINOR, _MODERATE = 0.6, 1.5


def _severity(width_ratio: float) -> Literal["minor", "moderate", "major"]:
    if width_ratio >= _MODERATE:
        return "major"
    if width_ratio >= _MINOR:
        return "moderate"
    return "minor"


def detect_gaps(params: ImageInput) -> DetectGapsOutput:
    original, meta = prepare_image(params.image)
    dets, result = run_detector(original, params.conf)
    if result.status == "error":
        raise ToolInputError(result.message or "detector failed")

    rows = cluster_rows(dets)
    gaps: list[ShelfGap] = []
    for row_idx, row in enumerate(rows):
        if len(row) < 2:
            continue
        widths = [d.bbox[2] - d.bbox[0] for d in row]
        mean_w = sum(widths) / len(widths)
        min_gap = mean_w * _MINOR
        for left, right in zip(row, row[1:], strict=False):
            gap_w = right.bbox[0] - left.bbox[2]
            if gap_w < min_gap:
                continue
            y1 = min(left.bbox[1], right.bbox[1])
            y2 = max(left.bbox[3], right.bbox[3])
            bbox = (left.bbox[2], y1, right.bbox[0], y2)
            ratio = gap_w / mean_w
            gaps.append(
                ShelfGap(
                    bbox=bbox,
                    bbox_normalized=(
                        bbox[0] / meta.width,
                        bbox[1] / meta.height,
                        bbox[2] / meta.width,
                        bbox[3] / meta.height,
                    ),
                    row=row_idx,
                    width_ratio=round(ratio, 2),
                    severity=_severity(ratio),
                )
            )

    notes = list(result.meta.get("notes", []))
    if not dets:
        notes.append("No products detected — cannot infer shelf rows or gaps.")

    annotated = None
    if params.annotate:
        annotated, _ = package_annotated(
            original,
            [g.bbox for g in gaps],
            [f"gap r{g.row} {g.severity}" for g in gaps],
            color_idx=1,
        )

    return DetectGapsOutput(
        gap_count=len(gaps),
        gaps=gaps,
        rows_detected=len(rows),
        image=meta,
        annotated=annotated,
        notes=notes,
    )
