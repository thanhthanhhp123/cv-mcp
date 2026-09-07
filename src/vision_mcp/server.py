"""FastMCP server: registers the shelf-auditor tools, resources, and prompts.

Run over stdio with ``python -m vision_mcp.server``.
"""

from __future__ import annotations

import base64
import json

from mcp.server.fastmcp import FastMCP

from .config import get_settings
from .jobs import get_job_manager
from .schemas import (
    CheckPlanogramOutput,
    CountProductsOutput,
    DetectGapsOutput,
    ImageInput,
    PlanogramInput,
    ReadPriceTagsOutput,
    ShelfReportInput,
    ShelfReportOutput,
)
from .storage import get_storage
from .tools._common import ToolInputError
from .tools.check_planogram import check_planogram as _check_planogram
from .tools.count_products import count_products as _count_products
from .tools.detect_gaps import detect_gaps as _detect_gaps
from .tools.read_price_tags import read_price_tags as _read_price_tags
from .tools.shelf_report import shelf_report as _shelf_report

mcp = FastMCP("shelf-auditor")


def _guard(fn, params):  # type: ignore[no-untyped-def]
    try:
        return fn(params)
    except ToolInputError as exc:
        raise ValueError(f"invalid input: {exc}") from None


# --- tools ---------------------------------------------------------
@mcp.tool()
def count_products(params: ImageInput) -> CountProductsOutput:
    """Count products on a shelf photo. Returns per-label counts, per-item boxes
    (absolute + normalized) with confidence, and an annotated image."""
    return _guard(_count_products, params)


@mcp.tool()
def detect_gaps(params: ImageInput) -> DetectGapsOutput:
    """Find empty (out-of-stock) regions on the shelf, grouped by row, each with
    a width ratio versus the row's mean product width and a severity."""
    return _guard(_detect_gaps, params)


@mcp.tool()
def read_price_tags(params: ImageInput) -> ReadPriceTagsOutput:
    """Read price labels and link each to the nearest product. NOTE: the OCR
    backend is currently a stub and returns status 'not_implemented'."""
    return _guard(_read_price_tags, params)


@mcp.tool()
def check_planogram(params: PlanogramInput) -> CheckPlanogramOutput:
    """Compare the shelf against a planogram spec (list of slots with expected
    normalized regions + optional order). Lists missing / misplaced / extra /
    wrong-order deviations."""
    return _guard(_check_planogram, params)


@mcp.tool()
def shelf_report(params: ShelfReportInput) -> ShelfReportOutput:
    """Run count, gap detection, and (optionally) a planogram check on one
    image and return a combined JSON and/or markdown report."""
    return _guard(_shelf_report, params)


@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """Status of an async job created by a heavy tool call."""
    status = get_job_manager().status(job_id)
    if status is None:
        raise ValueError(f"unknown job_id: {job_id}")
    return status


@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """Result of a finished async job."""
    return get_job_manager().result(job_id)


# --- resources ---------------------------------------------------
@mcp.resource("models://available")
def models_available() -> str:
    s = get_settings()
    return json.dumps(
        {
            "detector": {
                "weights": s.detector_weights,
                "present": s.detector_path.is_file(),
                "device": s.device,
                "class_agnostic": True,
            },
            "ocr": {"backend": "stub", "status": "not_implemented"},
        },
        indent=2,
    )


@mcp.resource("image://{image_id}")
def image_resource(image_id: str) -> str:
    img = get_storage().get_image(image_id)
    if img is None:
        raise ValueError(f"no cached image: {image_id}")
    from .imaging import to_base64_png

    return to_base64_png(img)


@mcp.resource("results://{job_id}")
def results_resource(job_id: str) -> str:
    res = get_storage().get_result(job_id)
    if res is None:
        raise ValueError(f"no cached result: {job_id}")
    return json.dumps(res, indent=2)


@mcp.resource("annotated://{job_id}")
def annotated_resource(job_id: str) -> str:
    png = get_storage().get_annotated(job_id)
    if png is None:
        raise ValueError(f"no annotated image: {job_id}")
    return base64.b64encode(png).decode("ascii")


# --- prompts ---------------------------------------------------
@mcp.prompt("count-items")
def prompt_count_items(image: str) -> str:
    return f"Use count_products on this shelf photo and summarise the counts:\n{image}"


@mcp.prompt("detect-gaps")
def prompt_detect_gaps(image: str) -> str:
    return f"Use detect_gaps on this shelf photo and list every out-of-stock region:\n{image}"


@mcp.prompt("read-shelf")
def prompt_read_shelf(image: str) -> str:
    return (
        "Use read_price_tags on this shelf photo and list each price with its "
        f"product. If the tool reports it is not implemented, say so plainly:\n{image}"
    )


@mcp.prompt("planogram-check")
def prompt_planogram_check(image: str) -> str:
    return (
        "Use check_planogram to compare this shelf photo against the planogram "
        f"the user provides, then summarise the deviations:\n{image}"
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
