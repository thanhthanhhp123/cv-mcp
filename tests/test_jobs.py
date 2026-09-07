from __future__ import annotations

import asyncio

from vision_mcp.jobs import JobManager


async def test_job_success():
    mgr = JobManager()

    async def work():
        await asyncio.sleep(0)
        return {"value": 42}

    job_id = mgr.submit(work)
    for _ in range(100):
        if mgr.status(job_id)["state"] == "done":
            break
        await asyncio.sleep(0.01)
    res = mgr.result(job_id)
    assert res["state"] == "done"
    assert res["result"] == {"value": 42}


async def test_job_error():
    mgr = JobManager()

    async def boom():
        raise RuntimeError("nope")

    job_id = mgr.submit(boom)
    for _ in range(100):
        if mgr.status(job_id)["state"] in ("done", "error"):
            break
        await asyncio.sleep(0.01)
    res = mgr.result(job_id)
    assert res["state"] == "error"
    assert "nope" in res["error"]


def test_unknown_job():
    assert JobManager().status("missing") is None
    assert JobManager().result("missing")["state"] == "error"
