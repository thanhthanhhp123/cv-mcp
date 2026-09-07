"""Run the tools on one image and write annotated PNGs you can open.

    uv run python scripts/try_image.py path/to/shelf.jpg
    uv run python scripts/try_image.py tests/fixtures/wine_shelf.jpg --slots slots.json

Outputs go to ``out/`` next to the repo root:
    out/<stem>.count.png    out/<stem>.gaps.png    out/<stem>.planogram.png
plus a JSON dump of each tool's result.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from vision_mcp.schemas import ImageInput, PlanogramInput, PlanogramSlot, ShelfReportInput
from vision_mcp.storage import get_storage
from vision_mcp.tools.check_planogram import check_planogram
from vision_mcp.tools.count_products import count_products
from vision_mcp.tools.detect_gaps import detect_gaps
from vision_mcp.tools.shelf_report import shelf_report


def _save_annotated(annotated, path: Path) -> None:
    if annotated is None:
        return
    if annotated.kind == "inline" and annotated.png_base64:
        path.write_bytes(base64.b64decode(annotated.png_base64))
    elif annotated.uri:
        job_id = annotated.uri.removeprefix("annotated://")
        png = get_storage().get_annotated(job_id)
        if png is None:
            print(f"  annotated reference {annotated.uri} not in storage")
            return
        path.write_bytes(png)
    print(f"  wrote {path}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument(
        "--slots",
        help="JSON file: list of {label, region:[x1,y1,x2,y2] in 0..1, order?}",
    )
    ap.add_argument("--conf", type=float, default=None)
    args = ap.parse_args(argv)

    out = Path(__file__).resolve().parents[1] / "out"
    out.mkdir(exist_ok=True)
    stem = Path(args.image).stem

    base = ImageInput(image=args.image, conf=args.conf)

    print("count_products:")
    c = count_products(base)
    print(f"  total={c.total} counts={c.counts}")
    _save_annotated(c.annotated, out / f"{stem}.count.png")
    (out / f"{stem}.count.json").write_text(c.model_dump_json(indent=2, exclude={"annotated"}))

    print("detect_gaps:")
    g = detect_gaps(base)
    print(f"  gap_count={g.gap_count} rows={g.rows_detected}")
    _save_annotated(g.annotated, out / f"{stem}.gaps.png")
    (out / f"{stem}.gaps.json").write_text(g.model_dump_json(indent=2, exclude={"annotated"}))

    slots = None
    if args.slots:
        raw = json.loads(Path(args.slots).read_text())
        slots = [PlanogramSlot(**s) for s in raw]
        print("check_planogram:")
        p = check_planogram(
            PlanogramInput(image=args.image, conf=args.conf, slots=slots)
        )
        print(f"  compliant={p.compliant} deviations={p.deviation_count}")
        _save_annotated(p.annotated, out / f"{stem}.planogram.png")
        (out / f"{stem}.planogram.json").write_text(
            p.model_dump_json(indent=2, exclude={"annotated"})
        )

    print("shelf_report (markdown):")
    r = shelf_report(ShelfReportInput(image=args.image, conf=args.conf, slots=slots, format="both"))
    (out / f"{stem}.report.md").write_text(r.markdown or "")
    print(r.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
