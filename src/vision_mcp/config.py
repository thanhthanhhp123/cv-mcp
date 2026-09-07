"""Runtime configuration.

All tunables live here so the MCP layer and the vision backends never hard-code
paths, device choices, or payload limits. Override any field with an env var
prefixed ``VISION_MCP_`` (e.g. ``VISION_MCP_DEVICE=cuda``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISION_MCP_", env_file=".env", extra="ignore")

    # --- models -------------------------------------------------------------
    models_dir: Path = _REPO_ROOT / "models"
    # SKU-110K fine-tune (scripts/hpc/). Falls back to COCO yolov8n.pt when the
    # file is absent — see YoloDetector._load.
    detector_weights: str = "yolov8s_sku110k.pt"
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # --- inference --------------------------------------------------------
    # Lower than the YOLO 0.25 default: retail shelves are dense and the SKU-110K
    # fine-tune scores real items lower on out-of-distribution store photos.
    detector_conf: float = Field(0.2, ge=0.0, le=1.0)
    detector_iou: float = Field(0.45, ge=0.0, le=1.0)
    # YOLO inference resolution. Retail shelves are dense with small items, so the
    # 640 default under-detects; 960 matches the SKU-110K fine-tune. Raise for
    # very dense shelves at the cost of speed.
    detector_imgsz: int = Field(960, gt=0)
    max_edge_px: int = Field(
        1920, gt=0, description="Resize longest image edge to this before inference."
    )

    # --- payload limits -------------------------------------------------
    max_input_bytes: int = Field(20 * 1024 * 1024, gt=0)
    # Above this, an annotated image is returned as an annotated://{job_id} ref
    # instead of inline base64.
    max_inline_image_bytes: int = Field(4 * 1024 * 1024, gt=0)

    # --- storage --------------------------------------------------------
    storage_dir: Path = _REPO_ROOT / ".cache"

    @property
    def detector_path(self) -> Path:
        return self.models_dir / self.detector_weights


@lru_cache
def get_settings() -> Settings:
    return Settings()
