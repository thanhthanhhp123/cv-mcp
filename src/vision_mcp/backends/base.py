"""The contract every vision backend obeys.

The MCP layer only ever touches ``BaseVisionModel.run`` and the ``Result`` it
returns. Swapping YOLOv8 for RT-DETR, or a local model for a cloud API, means
writing a new subclass here and nothing else.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Detection:
    """One detected box in absolute pixel coordinates."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)

    def normalized(self, width: int, height: int) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 / width, y1 / height, x2 / width, y2 / height)


@dataclass
class Result:
    """Backend output. ``status`` lets a backend report "not implemented" or a
    soft failure without raising across the MCP protocol."""

    status: str = "ok"  # "ok" | "not_implemented" | "error"
    detections: list[Detection] = field(default_factory=list)
    message: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class BaseVisionModel(abc.ABC):
    """Lazy-loading, cached vision model.

    Subclasses implement ``_load`` (called once, on first ``run``) and
    ``_infer``. ``run`` handles the caching and wraps failures into a
    ``Result(status="error")``.
    """

    name: str = "base"

    def __init__(self) -> None:
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> None:
        if self._model is None:
            self._model = self._load()

    @abc.abstractmethod
    def _load(self) -> Any:
        """Load and return the underlying model object."""

    @abc.abstractmethod
    def _infer(self, image: np.ndarray, params: dict[str, Any]) -> Result:
        """Run inference on an RGB uint8 HxWx3 array."""

    def run(self, image: np.ndarray, params: dict[str, Any] | None = None) -> Result:
        params = params or {}
        try:
            self.ensure_loaded()
            return self._infer(image, params)
        except Exception as exc:  # noqa: BLE001 - surfaced as a Result, not raised
            return Result(status="error", message=f"{self.name}: {exc}")
