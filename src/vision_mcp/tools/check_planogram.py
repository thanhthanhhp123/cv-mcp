"""``check_planogram`` — compare the actual layout to a reference spec.

The spec is a list of slots, each with an expected normalized region and an
optional order index. We detect products, match each slot to the best
overlapping detection, and report missing / misplaced / extra / wrong-order
deviations.
"""

from __future__ import annotations

from ..backends.matching import center, iou
from ..schemas import CheckPlanogramOutput, PlanogramDeviation, PlanogramInput
from ._common import ToolInputError, package_annotated, prepare_image, run_detector


def check_planogram(params: PlanogramInput) -> CheckPlanogramOutput:
    original, meta = prepare_image(params.image)
    dets, result = run_detector(original, params.conf)
    if result.status == "error":
        raise ToolInputError(result.message or "detector failed")

    w, h = meta.width, meta.height
    deviations: list[PlanogramDeviation] = []
    matched_idx: set[int] = set()
    slot_match_x: list[tuple[int, float]] = []  # (order, matched center x) for order check

    for slot in params.slots:
        exp_px = (
            slot.region[0] * w,
            slot.region[1] * h,
            slot.region[2] * w,
            slot.region[3] * h,
        )
        best_i, best_iou = -1, 0.0
        for i, d in enumerate(dets):
            if i in matched_idx:
                continue
            score = iou(exp_px, d.bbox)
            if score > best_iou:
                best_i, best_iou = i, score

        if best_i < 0 or best_iou < params.iou_threshold:
            deviations.append(
                PlanogramDeviation(
                    type="missing",
                    label=slot.label,
                    expected_region=slot.region,
                    detail=f"No product overlaps the expected region (best IoU {best_iou:.2f}).",
                )
            )
            continue

        matched_idx.add(best_i)
        d = dets[best_i]
        if slot.order is not None:
            slot_match_x.append((slot.order, center(d.bbox)[0]))
        # "misplaced": matched, but centre well outside the expected region.
        cx, cy = center(d.bbox)
        if not (exp_px[0] <= cx <= exp_px[2] and exp_px[1] <= cy <= exp_px[3]):
            deviations.append(
                PlanogramDeviation(
                    type="misplaced",
                    label=slot.label,
                    expected_region=slot.region,
                    observed_bbox=d.bbox,
                    detail="Matched product centre falls outside the expected region.",
                )
            )

    for i, d in enumerate(dets):
        if i not in matched_idx:
            deviations.append(
                PlanogramDeviation(
                    type="extra",
                    label="product",
                    observed_bbox=d.bbox,
                    detail="Detected product not accounted for by any planogram slot.",
                )
            )

    # wrong_order: expected order vs. left-to-right position of matched centres.
    ordered = [x for _, x in sorted(slot_match_x)]
    if any(a > b for a, b in zip(ordered, ordered[1:], strict=False)):
        deviations.append(
            PlanogramDeviation(
                type="wrong_order",
                label="(sequence)",
                detail="Left-to-right order of matched products does not follow slot order.",
            )
        )

    annotated = None
    if params.annotate:
        annotated, _ = package_annotated(
            original,
            [d.observed_bbox for d in deviations if d.observed_bbox],
            [d.type for d in deviations if d.observed_bbox],
            color_idx=2,
        )

    return CheckPlanogramOutput(
        compliant=not deviations,
        deviation_count=len(deviations),
        deviations=deviations,
        image=meta,
        annotated=annotated,
        notes=list(result.meta.get("notes", [])),
    )
