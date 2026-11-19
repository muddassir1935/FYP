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
| 3 | weapon_yolov8s_v2_20261119_1059 | 2026-11-19 | 100 epochs, imgsz 640, batch 32, seed 42, 2x T4 DDP (Kaggle), dataset v2 + Knife 2x oversampling | 0.8621 | 0.5543 | 0.8319 | 0.8278 | 0.8298 | 0.7813 | 0.943 |

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

### #3 — Dataset v2 retrain (2026-11-19)
First run on dataset v2 (train 21,809 images after the quality sweep, with
Knife 2x oversampling of 1,889 knife images → 23,698; val 1,351; **test 680
FROZEN, the same split as v1, so the metrics are directly comparable**).
Trained on Kaggle (2x T4, DDP), 100 epochs in ~6.6 h, patience 20 (no early
stop). Adopted as the current working model (v2).

- Weights: `weapon_yolov8s_v2_20261119_1059_best.pt` (22.53 MB)
- SHA-256: `1c6ef07711287d62a8d010ffe9d9d7a7b558f632f285e91e1b4f66cd9645c847`
- Drive: `SENTINEL_FYP/runs/weapon_yolov8s_v2_20261119_1059/`
  (best.pt id `1z2baegmmqL1frskwzWQu5c6ffwDwj604`)
- **Result:** mAP@0.5 0.8066 → **0.8621** (+5.6 pts). The biggest gains are
  exactly where v1 was weakest: Gun AP50 0.7088 → **0.7813** (+7.3 pts) and
  recall 0.7615 → **0.8278** (+6.6 pts). Knife AP50 0.9045 → **0.943**.
  Precision is flat (0.8326 → 0.8319) — the gains came from finding more
  true weapons, not from fewer false alarms.
- **Post-training crash (recovered, no retrain needed):** after all 100
  epochs, the training script crashed at `results.save_dir` — Kaggle's
  newer Ultralytics returns a plain metrics dict from `model.train()`,
  not the trainer object. The test evaluation and `run_summary.json`
  never ran on Kaggle. The weights were safe (YOLO saves them during
  training); `best.pt` + result files were downloaded via the Kaggle
  session-output API and the test metrics above were evaluated locally
  on CPU from the downloaded weights. Fixed in
  `scripts/training/train_weapon_detector.py` (`resolve_train_output`
  handles both return types).
- **Next:** retest on the same bullet video + laptop webcam (user) — the
  real-world check that val/test numbers cannot give.

### Error analysis — baseline v1 (2026-11-09, from the val confusion matrix)

- Gun<->Knife cross-confusion is negligible (12 Gun predicted as Knife,
  3 Knife predicted as Gun): the two classes are not confused with each
  other.
- The dominant errors are background false alarms at the matrix's 0.25
  threshold (22,256 Gun / 1,765 Knife unmatched predictions accumulated
  across the val split — the matrix counts per image, so read these as
  relative magnitudes, not absolute rates) and Gun misses (120 of 1,209
  true Guns predicted background). The headline P/R/F1 are measured at
  the model's best operating threshold — which is why the demo's default
  confidence was raised to 0.40.
- Field observation (user test, 2026-11-09): bullets held in a hand are
  confirmed as Knife at up to 0.62 confidence. The scene is near-static,
  so the false detection persists and the FR-05 buffer (which validates
  persistence, not correctness) cannot filter it. Suspected hand-grip
  context bias in the knife training images. Planned fix: dataset v2
  with hard negatives (hands holding bullets/coins/pens/tools, ammunition
  close-ups — all as background) plus retraining.

### Next
Pipeline build (capture + weapon detector + temporal buffer + demo runner)
on this baseline model. One optional tuning run (100 epochs) is queued for
later in Phase 3 if the schedule allows; any Gun-focused data work comes
after a label-quality inspection, not before.
