from __future__ import annotations

import json

import pytest

from vision_mcp import server


async def test_tools_registered():
    tools = {t.name for t in await server.mcp.list_tools()}
    assert {
        "count_products",
        "detect_gaps",
        "read_price_tags",
        "check_planogram",
        "shelf_report",
        "get_job_status",
        "get_job_result",
    } <= tools


async def test_prompts_registered():
    names = {p.name for p in await server.mcp.list_prompts()}
    assert {"count-items", "detect-gaps", "read-shelf", "planogram-check"} <= names


def test_models_available_resource():
    payload = json.loads(server.models_available())
    assert payload["ocr"]["status"] == "not_implemented"
    assert "detector" in payload


def test_guard_wraps_tool_input_error():
    from vision_mcp.tools._common import ToolInputError

    def boom(_):
        raise ToolInputError("bad path")

    with pytest.raises(ValueError, match="invalid input: bad path"):
        server._guard(boom, None)
