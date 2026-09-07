"""[HEAD NODE — has internet] Download and convert SKU-110K into YOLO format.

Uses ultralytics' own SKU-110K dataset spec, which downloads
``SKU110K_fixed.tar.gz`` (~13 GB) and converts the CSV annotations to YOLO
labels + ``train/val/test.txt`` lists. The offline training job then just points
``data=SKU-110K.yaml`` and finds this local copy.

    ~/shelf/env/bin/python ~/shelf/jobs/prepare_sku110k.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("SHELF_ROOT", Path.home() / "shelf"))
DATA = ROOT / "data"


def _free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 2**30


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    free = _free_gb(DATA)
    print(f"free in {DATA}: {free:.0f} GB")
    if free < 35 and not (DATA / "SKU-110K").exists():
        print("FATAL: need ~35 GB free for the SKU-110K download + extract", file=sys.stderr)
        return 1

    os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch"))

    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import SETTINGS

    # make sure ultralytics resolves the dataset under ~/shelf/data
    SETTINGS.update({"datasets_dir": str(DATA)})

    info = check_det_dataset("SKU-110K.yaml", autodownload=True)
    print("dataset ready:")
    for k in ("path", "train", "val", "test", "nc", "names"):
        print(f"  {k}: {info.get(k)}")

    # drop the tarball if it survived extraction
    for tar in DATA.glob("SKU110K*fixed*.tar*"):
        print(f"removing {tar.name} ({tar.stat().st_size / 2**30:.1f} GB)")
        tar.unlink()

    ds = Path(info["path"])
    for split in ("train", "val", "test"):
        lst = ds / f"{split}.txt"
        n = len(lst.read_text().splitlines()) if lst.is_file() else 0
        print(f"  {split}: {n} images")
    print(f"free in {DATA} now: {_free_gb(DATA):.0f} GB")
    print(">> prepare done. Next: run.sh push && run.sh submit train_sku110k.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
