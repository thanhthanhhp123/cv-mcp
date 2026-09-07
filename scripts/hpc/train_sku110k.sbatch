#!/bin/bash
# Fine-tune the shelf detector on SKU-110K, then export ONNX.
# Prereqs (all on the head node, which has internet):
#   run.sh setup     -> ~/shelf/env + ~/shelf/models/yolov8s.pt
#   run.sh prepare   -> ~/shelf/data/SKU-110K/  (images + YOLO labels + *.txt)
#
# Single GPU + Ultralytics auto-batch (batch=-1) so it adapts to whatever card
# SLURM gives us (A40 46 GB, L40 48 GB, RTX 6000 24 GB all seen on main-gpu).
#
#SBATCH --job-name=shelf-yolo
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/%u/shelf/logs/slurm-%j.out

set -euo pipefail
ROOT="$HOME/shelf"
PY="$ROOT/env/bin/python"
YOLO="$ROOT/env/bin/yolo"
[ -x "$PY" ] || { PY="$HOME/mct/env/bin/python"; YOLO="$HOME/mct/env/bin/yolo"; }

MODEL="${MODEL:-$ROOT/models/yolov8s.pt}"
DATA="${DATA:-SKU-110K.yaml}"          # resolved via ultralytics + datasets_dir
EPOCHS="${EPOCHS:-30}"
IMGSZ="${IMGSZ:-960}"                  # SKU-110K objects are tiny + dense; >640 helps
BATCH="${BATCH:--1}"                   # -1 = auto (probe VRAM); set an int to pin
NAME="${NAME:-yolov8s_sku110k}"

export ULTRALYTICS_OFFLINE=1 HF_HUB_OFFLINE=1 YOLO_OFFLINE=1
export TORCH_HOME="$ROOT/.torch"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

echo "node=$(hostname)  job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "model=$MODEL data=$DATA epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH"
[ -f "$MODEL" ] || { echo "FATAL: missing $MODEL (run.sh setup first)"; exit 1; }

"$YOLO" detect train \
  model="$MODEL" data="$DATA" epochs="$EPOCHS" imgsz="$IMGSZ" batch="$BATCH" \
  device=0 workers=12 project="$ROOT/runs" name="$NAME" exist_ok=True \
  patience=10 cos_lr=True

BEST="$ROOT/runs/$NAME/weights/best.pt"
[ -f "$BEST" ] || { echo "FATAL: no $BEST"; exit 1; }

echo "=== validate best.pt on val split ==="
"$YOLO" detect val model="$BEST" data="$DATA" imgsz="$IMGSZ" \
  project="$ROOT/runs" name="${NAME}_val" exist_ok=True || true

echo "=== export ONNX ==="
"$YOLO" export model="$BEST" format=onnx opset=12 simplify=True \
  dynamic=False imgsz="$IMGSZ"

mkdir -p "$ROOT/models"
cp "$BEST" "$ROOT/models/$NAME.pt"
cp "${BEST%.pt}.onnx" "$ROOT/models/$NAME.onnx"
ls -lh "$ROOT/models/$NAME."*
echo "=== fetch back:  bash scripts/hpc/run.sh fetch shelf/models/$NAME.pt models/$NAME.pt ==="
