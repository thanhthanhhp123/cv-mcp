#!/bin/bash
# [HEAD NODE] Create ~/$ROOT, a conda env with ultralytics, and stage the base
# YOLO weights. Idempotent — safe to re-run. ~10 min, ~8 GB on first run.
set -eu
ROOT="${ROOT:-shelf}"
BASE="$HOME/$ROOT"

mkdir -p "$BASE"/{data,jobs,runs,models,logs,.torch,.ultralytics}

df -h "$HOME" | tail -1
FREE_G=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
[ "$FREE_G" -ge 40 ] || { echo "FATAL: only ${FREE_G}G free in \$HOME; need ~40G"; exit 1; }

ENV_DIR="$BASE/env"
if [ -x "$ENV_DIR/bin/python" ] && "$ENV_DIR/bin/python" -c "import ultralytics" 2>/dev/null; then
  echo ">> env already good: $ENV_DIR"
else
  CONDA=""
  for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$(command -v conda || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && { CONDA="$c"; break; }
  done
  [ -n "$CONDA" ] || { echo "FATAL: no conda found"; exit 1; }
  echo ">> conda: $CONDA"
  [ -x "$ENV_DIR/bin/python" ] || "$CONDA" create -y -p "$ENV_DIR" python=3.10
  "$ENV_DIR/bin/pip" install -q -U pip
  # driver 595 runs any cu12x wheel; cu124 is the tested pairing with ultralytics
  "$ENV_DIR/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
  # polars is required by ultralytics' SKU-110K CSV->YOLO conversion script
  "$ENV_DIR/bin/pip" install -q ultralytics onnx onnxruntime onnxslim pyyaml polars
fi

"$ENV_DIR/bin/python" - <<'PY'
import torch, ultralytics
print("torch", torch.__version__, "| ultralytics", ultralytics.__version__)
PY

# point ultralytics at our dataset dir so `prepare` and the offline job agree
"$ENV_DIR/bin/yolo" settings datasets_dir="$BASE/data" runs_dir="$BASE/runs" \
  weights_dir="$BASE/models" >/dev/null 2>&1 || true

# stage base weights (head node has internet; compute nodes do not)
cd "$BASE/models"
for w in yolov8s.pt yolov8n.pt; do
  [ -f "$w" ] || { echo ">> fetch $w"; \
    "$ENV_DIR/bin/python" -c "from ultralytics import YOLO; YOLO('$w')"; \
    [ -f "$w" ] || mv "$HOME/$w" . 2>/dev/null || true; }
done
ls -lh "$BASE/models"
echo ">> setup done. Next: run.sh prepare"
