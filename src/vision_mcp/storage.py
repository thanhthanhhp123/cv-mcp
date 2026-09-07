"""In-memory cache for input images and job results, with a small disk spill for
annotated PNGs so they can be served via the ``annotated://{job_id}`` resource.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any

import numpy as np

from .config import get_settings
from .imaging import encode_png


class _LRU(OrderedDict[str, Any]):
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity

    def put(self, key: str, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        self[key] = value
        while len(self) > self.capacity:
            self.popitem(last=False)


class Storage:
    def __init__(self, image_capacity: int = 32, result_capacity: int = 128) -> None:
        self._lock = threading.Lock()
        self._images: _LRU = _LRU(image_capacity)
        self._results: _LRU = _LRU(result_capacity)
        self._annotated: _LRU = _LRU(result_capacity)
        self._dir = get_settings().storage_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    # --- images --------------------------------------------------------
    def put_image(self, image: np.ndarray) -> str:
        image_id = uuid.uuid4().hex
        with self._lock:
            self._images.put(image_id, image)
        return image_id

    def get_image(self, image_id: str) -> np.ndarray | None:
        with self._lock:
            img = self._images.get(image_id)
        return img

    # --- results ------------------------------------------------------
    def put_result(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._results.put(job_id, result)

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._results.get(job_id)

    # --- annotated images -------------------------------------------
    def put_annotated(self, job_id: str, image: np.ndarray) -> str:
        png = encode_png(image)
        path = self._dir / f"{job_id}.png"
        path.write_bytes(png)
        with self._lock:
            self._annotated.put(job_id, png)
        return f"annotated://{job_id}"

    def get_annotated(self, job_id: str) -> bytes | None:
        with self._lock:
            cached = self._annotated.get(job_id)
        if cached is not None:
            return cached
        path = self._dir / f"{job_id}.png"
        return path.read_bytes() if path.is_file() else None


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage
