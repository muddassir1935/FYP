# Training Run Log — YOLOv8s Weapon Detector (FR-02)

Model weights are **never** committed to Git (repo rule). Every run is recorded
here with its configuration, held-out **test split** metrics, and the weights
file's SHA-256 hash plus Google Drive location, so any result in the report
can be traced back to the exact weights that produced it.

- Drive root: `SENTINEL_FYP/runs/<run_name>/` (weights, `results.csv`,
  `run_summary.json`, confusion matrix)
- Training script: `scripts/training/train_weapon_detector.py`
- Notebook: `notebooks/sentinel_weapon_training.ipynb`
- Dataset: weapon YOLO — 6,762 images (train 4,731 / val 1,351 / test 680),
  classes Gun (0), Knife (1); verified in Colab before every run

| # | Run | Date | Config | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 | Gun AP50 | Knife AP50 |
|---|-----|------|--------|---------|--------------|---|---|----|----------|------------|
| 0 | flowtest_yolov8s_20261108_1021 | 2026-11-08 | pipeline check only (coco8, 5 epochs) | — | — | — | — | — | — | — |
| 1 | weapon_yolov8s_smoke_20261108_1759 | 2026-11-08 | 10 epochs, imgsz 640, batch 16, seed 42, T4 | 0.653 | 0.3335 | 0.7128 | 0.5827 | 0.6412 | 0.5272 | 0.7787 |

## Run notes

### #0 — Flow test (2026-11-08)
End-to-end pipeline verification only (GPU, Drive mount/save, mini-train,
metrics, demo predict). Weights `flowtest_yolov8s_20261108_1021_best.pt`
(22.58 MB, SHA-256 `586c46150c74938ec9340d8d57cb473be0592ffc387ab66acf70b3e8b060dd29`)
in `SENTINEL_FYP/test_run/`. Metrics are meaningless by design.

### #1 — Smoke run (2026-11-08)
First run on the real dataset; purpose was pipeline validation on real data
before committing to a long baseline run. Dataset verification passed
(6,762 images, label class ids clean).

- Weights: `weapon_yolov8s_smoke_20261108_1759_best.pt` (22.49 MB)
- SHA-256: `96dba97b15d735d36a7611f162cf7abe4a3adf4007491e3516fc5d3015d8d1af`
- **Finding:** Gun is the weak class (AP50 0.527) despite having ~3.3x more
  images than Knife (0.779) — the opposite of the pre-run expectation. Gun
  images come from more diverse sources and vary more in scale/quality;
  Knife images are visually consistent. Watch Gun AP/recall in the baseline
  run; if it stays low, the next lever is a gun label-quality/diversity
  inspection and more epochs, not more knife data.

### Next
Baseline run — 50 epochs, same configuration (only the epoch count changes),
then a tuning decision from its test metrics.
