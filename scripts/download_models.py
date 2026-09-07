"""Download the COCO fallback detector weights into ``models/``.

The default detector is the SKU-110K fine-tune (``yolov8s_sku110k.pt``), which is
not on the ultralytics CDN — fetch it with ``scripts/hpc/`` or copy it in. This
script grabs the COCO checkpoint the detector falls back to, so the server runs
before the fine-tune is in place.

    python scripts/download_models.py            # yolov8n.pt (the fallback)
    python scripts/download_models.py yolov8s.pt
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from vision_mcp.config import get_settings

_DEFAULT = "yolov8n.pt"


def main(argv: list[str]) -> int:
    settings = get_settings()
    weights = argv[1] if len(argv) > 1 else _DEFAULT
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    target = settings.models_dir / weights

    if target.is_file():
        print(f"already present: {target}")
        return 0

    from ultralytics import YOLO

    print(f"fetching {weights} ...")
    YOLO(weights)  # triggers download into the ultralytics cache / cwd
    for candidate in (Path(weights), Path.cwd() / weights):
        if candidate.is_file():
            shutil.move(str(candidate), target)
            print(f"saved: {target}")
            return 0

    print(f"downloaded {weights} but could not locate the file to move it", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
