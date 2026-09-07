"""Async job manager for heavy calls.

Fast single-image tools run synchronously and never touch this. Batch /
segmentation work (post-MVP) submits a coroutine here, gets a ``job_id`` back,
and the agent polls ``get_job_status`` / ``get_job_result``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

JobState = Literal["pending", "running", "done", "error"]


@dataclass
class Job:
    id: str
    state: JobState = "pending"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: Any = None
    error: str | None = None

    def as_status(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "state": self.state,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    def submit(self, coro_factory: Callable[[], Awaitable[Any]]) -> str:
        job = Job(id=uuid.uuid4().hex)
        self._jobs[job.id] = job

        async def _run() -> None:
            job.state = "running"
            try:
                job.result = await coro_factory()
                job.state = "done"
            except Exception as exc:  # noqa: BLE001
                job.state = "error"
                job.error = str(exc)
            finally:
                job.finished_at = time.time()

        self._tasks[job.id] = asyncio.ensure_future(_run())
        return job.id

    def status(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return job.as_status() if job else None

    def result(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            return {"job_id": job_id, "state": "error", "error": "unknown job_id"}
        if job.state in ("pending", "running"):
            return {"job_id": job_id, "state": job.state}
        if job.state == "error":
            return {"job_id": job_id, "state": "error", "error": job.error}
        return {"job_id": job_id, "state": "done", "result": job.result}


_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
