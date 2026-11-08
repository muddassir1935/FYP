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
| 2 | weapon_yolov8s_baseline_20261108_1837 | 2026-11-08 | 50 epochs, imgsz 640, batch 16, seed 42, T4 | 0.8066 | 0.484 | 0.8326 | 0.7615 | 0.7955 | 0.7088 | 0.9045 |

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

### #2 — Baseline run (2026-11-08)
First full run; only the epoch count changed vs the smoke run (10 → 50).
Training took ~85 minutes on a free-tier T4 (runtime capped at 2.5 h that
session). Adopted as the current working model (v1) for the FYP-I pipeline
and demo.

- Weights: `weapon_yolov8s_baseline_20261108_1837_best.pt` (22.50 MB)
- SHA-256: `5bb7e92824cb1f9fd976d0e4d5a1cf45d6f9d24b96c58606ce2be785975febad`
- **Result:** mAP@0.5 0.653 → **0.8066** vs smoke. Knife is now strong
  (AP50 0.9045); Gun improved the most in absolute terms (0.527 → 0.709) but
  remains the weaker class, gap narrowed from 0.25 to ~0.20. Recall 0.7615
  is the metric to watch for a surveillance use case — the FR-05 temporal
  consistency buffer is the designed mitigation for single-frame misses and
  false alarms downstream.

### Next
Pipeline build (capture + weapon detector + temporal buffer + demo runner)
on this baseline model. One optional tuning run (100 epochs) is queued for
later in Phase 3 if the schedule allows; any Gun-focused data work comes
after a label-quality inspection, not before.
