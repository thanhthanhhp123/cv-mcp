from __future__ import annotations

from vision_mcp.backends.base import Detection
from vision_mcp.backends.matching import cluster_rows, iou


def _det(x1, y1, x2, y2):
    return Detection(label="product", confidence=0.9, bbox=(x1, y1, x2, y2))


def test_iou_identical():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_cluster_rows_splits_two_rows():
    dets = [
        _det(0, 0, 20, 40),
        _det(30, 2, 50, 42),
        _det(0, 200, 20, 240),
        _det(30, 202, 50, 242),
    ]
    rows = cluster_rows(dets)
    assert len(rows) == 2
    assert all(len(r) == 2 for r in rows)
    # rows are ordered top-to-bottom, items left-to-right
    assert rows[0][0].bbox[0] < rows[0][1].bbox[0]


def test_cluster_rows_empty():
    assert cluster_rows([]) == []
