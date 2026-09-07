#!/bin/bash
# Prove the ~/shelf conda env reaches a GPU on a compute node. Run once.
#SBATCH --job-name=shelf-smoke
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/%u/shelf/logs/slurm-%j.out

set -euo pipefail
ROOT="$HOME/shelf"
PY="$ROOT/env/bin/python"
[ -x "$PY" ] || PY="$HOME/mct/env/bin/python"

echo "node=$(hostname)  job=$SLURM_JOB_ID  gpus=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

export ULTRALYTICS_OFFLINE=1 HF_HUB_OFFLINE=1 YOLO_OFFLINE=1
export TORCH_HOME="$ROOT/.torch"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
a = torch.randn(4096, 4096, device="cuda")
(a @ a).sum().item()
print("matmul ok, vram_GB=%.1f" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
import ultralytics; print("ultralytics", ultralytics.__version__)
from ultralytics.data.utils import check_det_dataset
info = check_det_dataset("SKU-110K.yaml", autodownload=False)
print("SKU-110K path:", info["path"], "| names:", info["names"])
PY
