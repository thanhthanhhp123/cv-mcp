"""Geometry helpers shared by detect_gaps and check_planogram."""

from __future__ import annotations

from ..backends.base import Detection

BBox = tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter)


def center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def cluster_rows(dets: list[Detection], tol_ratio: float = 0.6) -> list[list[Detection]]:
    """Group detections into shelf rows by vertical center.

    ``tol_ratio`` is a fraction of the median box height: centers within that
    band of a row's running mean join the row.
    """

    if not dets:
        return []
    heights = sorted((d.bbox[3] - d.bbox[1]) for d in dets)
    median_h = heights[len(heights) // 2] or 1.0
    tol = median_h * tol_ratio

    ordered = sorted(dets, key=lambda d: center(d.bbox)[1])
    rows: list[list[Detection]] = []
    row_means: list[float] = []
    for d in ordered:
        cy = center(d.bbox)[1]
        placed = False
        for i, mean in enumerate(row_means):
            if abs(cy - mean) <= tol:
                rows[i].append(d)
                row_means[i] = sum(center(x.bbox)[1] for x in rows[i]) / len(rows[i])
                placed = True
                break
        if not placed:
            rows.append([d])
            row_means.append(cy)

    order = sorted(range(len(rows)), key=lambda i: row_means[i])
    return [sorted(rows[i], key=lambda d: d.bbox[0]) for i in order]
