"""YOLO (ultralytics) object detector.

COCO-pretrained YOLO has no "retail product" class, so for shelf work we run it
**class-agnostic**: every box becomes the label ``"product"``. Fine-tuning on
SKU-110K is the documented next step. Model loads lazily on first ``run`` and is
cached; CPU is used when CUDA is unavailable, with a note in the ``Result``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import get_settings
from .base import BaseVisionModel, Detection, Result


class YoloDetector(BaseVisionModel):
    name = "yolo-detector"

    def __init__(self, class_agnostic: bool = True) -> None:
        super().__init__()
        self.class_agnostic = class_agnostic
        self._device: str = "cpu"
        self._weights_note: str | None = None

    def _resolve_device(self) -> str:
        settings = get_settings()
        if settings.device != "auto":
            return settings.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def _load(self) -> Any:
        from ultralytics import YOLO

        settings = get_settings()
        weights = settings.detector_path
        if weights.is_file():
            source = str(weights)
            self._weights_note = None
        else:
            # The default points at the SKU-110K fine-tune; when it hasn't been
            # fetched, fall back to a COCO checkpoint ultralytics can download.
            source = "yolov8n.pt"
            self._weights_note = (
                f"{settings.detector_weights} not found in {settings.models_dir}; "
                "using COCO yolov8n.pt, which misses non-drink retail items. "
                "Fine-tune with scripts/hpc/ for real shelf accuracy."
            )
        model = YOLO(source)
        self._device = self._resolve_device()
        return model

    def _infer(self, image: np.ndarray, params: dict[str, Any]) -> Result:
        settings = get_settings()
        conf = float(params.get("conf") or settings.detector_conf)
        iou = float(params.get("iou") or settings.detector_iou)

        preds = self._model.predict(  # type: ignore[union-attr]
            image,
            conf=conf,
            iou=iou,
            imgsz=settings.detector_imgsz,
            device=self._device,
            verbose=False,
        )
        detections: list[Detection] = []
        names = getattr(self._model, "names", {})
        for pred in preds:
            for box in pred.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                label = "product" if self.class_agnostic else str(names.get(cls_id, cls_id))
                detections.append(
                    Detection(
                        label=label,
                        confidence=round(float(box.conf[0]), 4),
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    )
                )

        notes = []
        if self._device == "cpu":
            notes.append("Running on CPU — inference may be slow (seconds per image).")
        if self._weights_note:
            notes.append(self._weights_note)

        return Result(
            status="ok",
            detections=detections,
            meta={"device": self._device, "conf": conf, "iou": iou, "notes": notes},
        )


_detector: YoloDetector | None = None


def get_detector() -> YoloDetector:
    global _detector
    if _detector is None:
        _detector = YoloDetector()
    return _detector
