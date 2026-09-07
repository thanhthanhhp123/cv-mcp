# Fine-tuning the detector on ut-hpc

Fine-tune the shelf detector on **SKU-110K** on the UT Twente SLURM cluster
(`ut-hpc`), then pull the weights back and point `config.py` at them.

Driven by the `ut-hpc` skill in the sibling `multi-cam-tracking` repo — same
cluster facts apply: **compute nodes have no internet**, so all downloads happen
on the head node; GPU work goes through `sbatch --partition=main-gpu`.

Remote layout (under `$HOME` on the cluster), kept separate from the skill's
`~/mct/`:

```
~/shelf/
  env/     conda env (or reuses ~/mct/env if it already has ultralytics)
  data/    SKU-110K/  (downloaded + converted by ultralytics on the head node)
  jobs/    *.sbatch pushed from here
  runs/    ultralytics training output
  models/  yolov8s.pt (start point) + fine-tuned best.pt / .onnx
  logs/    slurm-<id>.out
```

> **Licensing:** SKU-110K is released for **academic / non-commercial** use.
> A model fine-tuned on it inherits that restriction — fine for the portfolio
> demo and benchmarking, not for shipping to a paying client. For commercial
> work, retrain on licensed or self-collected data with the same pipeline.

## Steps

```bash
S=scripts/hpc/run.sh

bash $S status                 # connectivity, GPU queue, home disk
bash $S setup                  # [head] make ~/shelf, conda env, stage yolov8s.pt
bash $S prepare                # [head] download + convert SKU-110K (~13 GB, 20-40 min)
bash $S push                   # upload scripts/hpc/*.sbatch -> ~/shelf/jobs/
bash $S submit smoke.sbatch    # [job] prove env reaches a GPU
bash $S watch <jobid>
bash $S submit train_sku110k.sbatch   # [job] the fine-tune (hours)
bash $S watch <jobid>
bash $S fetch shelf/models/yolov8s_sku110k.pt models/yolov8s_sku110k.pt
```

Then wire it in (the `.pt` lands in `models/`, which is gitignored):

```bash
export VISION_MCP_DETECTOR_WEIGHTS=yolov8s_sku110k.pt
# or persist it in .env:  echo 'VISION_MCP_DETECTOR_WEIGHTS=yolov8s_sku110k.pt' >> .env
```

`config.detector_imgsz` is already 960 to match the fine-tune. The detector runs
class-agnostic, so the single `object` class is surfaced as `"product"` — no code
change needed.

Tune the run via `sbatch --export`: `EPOCHS`, `IMGSZ`, `BATCH`, `MODEL`, `NAME`.

## Progress log

| Date | Job | Result |
|------|-----|--------|
| 2026-09-07 | setup / prepare | env ready (torch 2.6+cu124, ultralytics 8.4.142); SKU-110K 8219/588/2936 converted |
| 2026-09-07 | 582977 smoke | OK — env reaches GPU, dataset resolves |
| 2026-09-07 | 582978 | FAILED — bad CLI (`python -m ultralytics.cfg`); fixed to `yolo detect train` |
| 2026-09-07 | 583005 | **done.** yolov8s, imgsz 960, batch=-1 (AutoBatch → 12; SKU-110K images are large + very dense), 30 epochs, 1× L40, **42 min**. val mAP50 **0.938**, mAP50-95 **0.595**, P 0.916 / R 0.886. → `models/yolov8s_sku110k.pt` (22 MB) + `.onnx` (43 MB) |

Artifacts in `artifacts/`: `results.png` (curves), `results.csv`, `args.yaml`,
`confusion_matrix_normalized.png`, and `<fixture>__{coco,sku110k}_yolov8s.jpg`
before/after crops.

### Reproduce / iterate

```bash
bash scripts/hpc/run.sh submit train_sku110k.sbatch --export=EPOCHS=50,IMGSZ=1280,BATCH=8
```

Higher `IMGSZ` (1280) helps the tiny dense boxes further, at ~2× epoch time.
AutoBatch only used 59% of the L40 at imgsz 960 — pin `BATCH=24` to go faster.

Cluster state after job 583005: `~/shelf/data` holds SKU-110K (~15 GB), env ~8 GB;
home was at 93 %. If space gets tight, `ssh ut-hpc 'rm -rf ~/shelf/data/SKU-110K'`
and re-run `run.sh prepare` before the next training job.

