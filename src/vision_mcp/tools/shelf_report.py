"""``shelf_report`` — meta-tool combining the others into one report."""

from __future__ import annotations

from ..schemas import (
    ImageInput,
    PlanogramInput,
    ShelfReportInput,
    ShelfReportOutput,
)
from .check_planogram import check_planogram
from .count_products import count_products
from .detect_gaps import detect_gaps
from .read_price_tags import read_price_tags


def _markdown(report: dict) -> str:
    lines = ["# Shelf audit report", ""]
    c = report["count_products"]
    lines += [f"**Products detected:** {c['total']}", ""]
    if c["counts"]:
        lines.append("| Label | Count |")
        lines.append("|---|---|")
        lines += [f"| {k} | {v} |" for k, v in c["counts"].items()]
        lines.append("")

    g = report["detect_gaps"]
    lines += [f"**Shelf gaps:** {g['gap_count']} across {g['rows_detected']} row(s)"]
    for gap in g["gaps"]:
        lines.append(f"- row {gap['row']}: {gap['severity']} (×{gap['width_ratio']} mean width)")
    lines.append("")

    if report.get("check_planogram"):
        p = report["check_planogram"]
        status = "compliant ✅" if p["compliant"] else f"{p['deviation_count']} deviation(s) ⚠️"
        lines += [f"**Planogram:** {status}"]
        for d in p["deviations"]:
            lines.append(f"- {d['type']}: {d['label']} — {d['detail']}")
        lines.append("")

    if report.get("read_price_tags"):
        pt = report["read_price_tags"]
        if pt["status"] == "ok":
            lines.append(f"**Price tags:** {len(pt['tags'])} read")
            for t in pt["tags"][:10]:
                val = f"{t['price_value']:.2f} {t['currency'] or ''}".strip()
                lines.append(f"- {val or t['text']}")
        else:
            lines.append(f"**Price tags:** {pt['status']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def shelf_report(params: ShelfReportInput) -> ShelfReportOutput:
    base = ImageInput(image=params.image, conf=params.conf, annotate=params.annotate)

    count_out = count_products(base)
    gaps_out = detect_gaps(base.model_copy(update={"annotate": False}))

    report: dict = {
        "count_products": count_out.model_dump(exclude={"annotated"}),
        "detect_gaps": gaps_out.model_dump(exclude={"annotated"}),
    }

    if params.slots:
        plano_out = check_planogram(
            PlanogramInput(
                image=params.image,
                conf=params.conf,
                annotate=False,
                slots=params.slots,
            )
        )
        report["check_planogram"] = plano_out.model_dump(exclude={"annotated"})

    if params.include_price_tags:
        pt_out = read_price_tags(base.model_copy(update={"annotate": False}))
        report["read_price_tags"] = pt_out.model_dump()

    markdown = _markdown(report) if params.format in ("markdown", "both") else None
    if params.format == "markdown":
        report = {}

    return ShelfReportOutput(
        report=report,
        markdown=markdown,
        image=count_out.image,
        annotated=count_out.annotated,
    )
