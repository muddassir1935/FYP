# Training Run Log — YOLOv8s Weapon Detector (FR-02)

Model weights are **never** committed to Git (repo rule). Every run is recorded
here with its configuration, held-out **test split** metrics, and the weights
file's SHA-256 hash plus Google Drive location, so any result in the report
can be traced back to the exact weights that produced it.

> **⚠️ LEAKAGE — read before citing any number below.** The frozen 680-image
> test set leaks into training: 169/680 test images (25%) have a verified
> near-duplicate in train, including 18 byte-identical files (Stage-1 audit,
> 21 Nov 2026 — `audit-v4/audit_stage1_report.md`). Every frozen-test mAP50
> in this log (v1 0.8066, v2 0.8621, v2.1 0.8108, v2.2 0.7955, v3 0.8666,
> v4ft 0.751, v4 0.8293) is therefore **leakage-inflated** — the honest
> series is the cleaned-511 test column below (v2 0.8264, v3 0.8378,
> v4ft 0.709, v4 0.8081). Quote the cleaned numbers in the report; the
> frozen numbers are comparable across runs only as a (biased) relative
> series.

- Drive root: `SENTINEL_FYP/runs/<run_name>/` (weights, `results.csv`,
  `run_summary.json`, confusion matrix)
- Training script: `scripts/training/train_weapon_detector.py`
- Notebook: `notebooks/sentinel_weapon_training.ipynb`
- Dataset: weapon YOLO, classes Gun (0), Knife (1). v1 trained on 6,762
  images (train 4,731 / val 1,351 / test 680); v2–v4 trained on Kaggle
  (not Colab) on 13,242–24,146 train images — see the per-entry dataset
  lines below. Val 1,351 / test 680 frozen across all runs.
- **Durations** are Kaggle session wall-clock, including setup and eval;
  pure training time from `results.csv` (final `time` column): v2 5.99 h,
  v3 8.01 h, v4 8.36 h.
- **RQ3 (temporal buffer):** see `docs/rq3_temporal_buffer.md` — the
  FR-05 buffer measurement (v2 95.0% / v3 98.1% FP-frame reduction) is a
  component-level result, not a demo claim.
- **`best.pt`** = the checkpoint from the best **mAP50-95** epoch
  (Ultralytics fitness = mAP50-95 in 8.4.215), not necessarily the
  best-mAP50 epoch. Each run's `run_summary.json` now records the
  best-mAP50 epoch alongside the best-fitness epoch and the mAP50 gap
  between them (`best_epoch_metrics`). If the gap ever exceeds ~0.005,
  evaluate both checkpoints on the cleaned test set before adopting.
  (v3: best-fitness epoch 100, mAP50 0.8684; best-mAP50 epoch 98,
  mAP50 0.8688 — gap 0.0003, harmless. v4: both epochs coincide at
  epoch 100.)
- **Augmentation trade-offs kept with the frozen v3 recipe (v5):**
  `degrees=20` — rotation re-boxes each rotated object with an
  axis-aligned box, which systematically loosens thin-knife labels (a
  ~10:1 knife rotated 20° grows to ~4.2× its area, mostly background).
  Kept because rotation fixed the vertical-knife→gun failure, and v3's
  Knife AP50 (0.9493 frozen test) shows the trade-off did not prevent
  strong knife detection. `erasing=0.4` — random erasing blanks a
  rectangle on ~40% of images and can delete a small knife from the
  image while its label stays (pure label noise for small objects).
  Risk compounds with mosaic, which shrinks objects 4× for the first
  90 epochs. Kept with the risk noted; v3's frozen-test numbers were
  still strong.
- **Mosaic trade-off:** `mosaic=1.0` composites 4 images into 1, so every
  object is rendered at half resolution (4× smaller area) for the first
  90 epochs; `close_mosaic=10` disables it only for the final 10
  epochs. This is a deliberate small-object trade-off kept with the
  frozen v3 recipe. If v5 ever early-stops, check whether mosaic was
  still on at the stop epoch — `close_mosaic` counts back from the
  *planned* total, so a stop before epoch 90 means mosaic was never
  closed.

| # | Run | Date | Config | mAP@0.5 (frozen 680) | Cleaned mAP@0.5 (511) | mAP@0.5:0.95 | P | R | F1 | Gun AP50 | Knife AP50 |
|---|-----|------|--------|----------------------|-----------------------|--------------|---|---|----|----------|------------|
| 0 | flowtest_yolov8s_20261108_1021 | 2026-11-08 | pipeline check only (coco8, 5 epochs) | — | — | — | — | — | — | — |
| 1 | weapon_yolov8s_smoke_20261108_1759 | 2026-11-08 | 10 epochs, imgsz 640, batch 16, seed 42, T4 | 0.653 | not measured | 0.3335 | 0.7128 | 0.5827 | 0.6412 | 0.5272 | 0.7787 |
| 2 | weapon_yolov8s_baseline_20261108_1837 | 2026-11-08 | 50 epochs, imgsz 640, batch 16, seed 42, T4 | 0.8066 | not measured | 0.484 | 0.8326 | 0.7615 | 0.7955 | 0.7088 | 0.9045 |
| 3 | weapon_yolov8s_v2_20261119_1059 | 2026-11-19 | 100 epochs, imgsz 640, batch 32, seed 42, 2x T4 DDP (Kaggle), dataset v2 + Knife 2x oversampling | 0.8621 | 0.8264 | 0.5543 | 0.8319 | 0.8278 | 0.8298 | 0.7813 | 0.943 |
| 4 | weapon_yolov8s_v21_20261120_1836 | 2026-11-20 | 25 epochs from v2 best.pt, imgsz 640, batch 32, seed 42, 2x T4 DDP (Kaggle), hard-mined dataset v2.1 (train 13,242) | 0.8108 | 0.7641 | 0.4581 | 0.8353 | 0.7647 | 0.7984 | 0.7416 | 0.8799 |
| 5 | weapon_yolov8s_v22_20261120_2045 | 2026-11-20 | 40 epochs (early stop @14) from v2 best.pt, lr0 0.0005, warmup 5, patience 12, 2x T4 DDP (Kaggle), dataset v2.1 | 0.7955 | 0.7557 | 0.3946 | 0.8415 | 0.7194 | 0.7757 | 0.6981 | 0.8928 |
| 6 | weapon_yolov8s_v3_20261120_2215 | 2026-11-21 | 100 epochs from COCO yolov8s.pt, imgsz 640, batch 32, seed 42, 2x T4 DDP (Kaggle), combined dataset v3 (train 24,146) + Knife 2x oversampling | 0.8666 | 0.8378 | 0.5377 | 0.8484 | 0.8164 | 0.8321 | 0.7839 | 0.9493 |
| 7 | weapon_yolov8s_v4ft_20261121_1230 | 2026-11-21 | 8 epochs from v3 best.pt, lr0 0.001, imgsz 640, batch 32, seed 42, 2x T4 DDP (Kaggle), dataset v4ft (train 4,145) — GATE FAIL | 0.751 | 0.709 | 0.3906 | 0.7931 | 0.7341 | 0.7625 | 0.6741 | 0.828 |
| 8 | weapon_yolov8s_v4_20261121_1350 | 2026-11-22 | 100 epochs from COCO yolov8s.pt, imgsz 640, batch 32, seed 42, 2x T4 DDP (Kaggle), audit-cleaned dataset v4 (train 23,268) + Knife 2x oversampling — GATE MIXED, v3 stays | 0.8293 | 0.8081 | 0.5031 | 0.8391 | 0.7864 | 0.8119 | 0.7543 | 0.9043 |

## Run notes

### #0 — Flow test (2026-11-08)
End-to-end pipeline verification only (GPU, Drive mount/save, mini-train,
metrics, demo predict). Weights `flowtest_yolov8s_20261108_1021_best.pt`
(22.58 MB, SHA-256 `586c46150c74938ec9340d8d57cb473be0592ffc387ab66acf70b3e8b060dd29`)
in `SENTINEL_FYP/test_run/`. Metrics are meaningless by design.
Provenance: the local run outputs (`run_summary.json`, `results.csv`)
were never downloaded for this run; the Drive copy is the sole source.

### #1 — Smoke run (2026-11-08)
First run on the real dataset; purpose was pipeline validation on real data
before committing to a long baseline run. Dataset verification passed
(6,762 images, label class ids clean).

- Weights: `weapon_yolov8s_smoke_20261108_1759_best.pt` (22.49 MB)
- SHA-256: `96dba97b15d735d36a7611f162cf7abe4a3adf4007491e3516fc5d3015d8d1af`
- Provenance: the local run outputs (`run_summary.json`, `results.csv`)
  were never downloaded for this run; the Drive copy is the sole source.
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
- Provenance: the local run outputs (`run_summary.json`, `results.csv`)
  were never downloaded for this run; the Drive copy is the sole source.
- **Leakage caveat:** the 0.8066 below is measured on the frozen test
  set, 25% of which leaks into training — the honest cleaned-test
  number for v1 was never measured.
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
- **Leakage caveat:** "directly comparable" above means same split, not
  honest — the frozen test leaks 25% into train, so 0.8621 is
  leakage-inflated; the honest cleaned-test mAP50 is **0.8264**.
- **Reproducibility:** Kaggle dataset `k230784/sentinel-weapon-v2`
  (version not recorded); kernel slug/version not recorded (user-run on
  Kaggle); local zip `out/weapon_yolo_v2.zip`, MD5
  `02cd71ae0ab617da7de0653e6c7522bf` (also split into 5 Drive parts, MD5s
  in `out/parts.md5`); seed 42.
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

### #4 — v2.1 hard-example fine-tune (2026-11-20)
25-epoch fine-tune from v2 best.pt on the hard-mined dataset (train 13,242:
10,905 v2 replay at 0.5 + 855 pixel-verified mined frames + 1,177 Roboflow
knife + 305 weather variants; val 1,351 / test 680 frozen). The goal was
real-video robustness. Clean test REGRESSED (0.8621 → 0.8108, gate FAIL):
fine-tuning on the hard mix shifted the decision boundary — the model got
more conservative (precision up, recall down) and forgot clean-test
behaviour. BUT on 1,456 audited hard frames: F1 0.363 → **0.805**, false
positives 313 → 33 (−89%), and the unseen split (616 frames never trained
on) improved F1 0.142 → 0.674 — the improvement generalizes, it is not
memorization. Failure modes eliminated: phone→Gun 19/50 → 0/50,
knife→Gun 22/63 → 0/63, UCF debris false alarms 122/434 → 0/434.

- Weights: `weapon_yolov8s_v21_20261120_1836_best.pt` (22.53 MB)
- SHA-256: `6262e00391689011fdbdd2ef69a8a22fd0aa2c460405feb97f66258576f107b5`
- **Cleaned-test (511) measurement (23 Nov 2026, fix session):** mAP50
  **0.7641** (P 0.7886 / R 0.7359, Gun AP 0.7034 / Knife AP 0.8247) — the
  regression holds on the clean set too (v3: 0.8378). Recorded in
  `audit-v4/clean_test_eval.json`.
- Drive: not uploaded (superseded; local copy only). Kept as the fallback
  model.
- **Reproducibility:** Kaggle dataset `k230784/sentinel-weapon-v21`
  (version not recorded); kernel slug/version not recorded (user-run on
  Kaggle); local zip `out/weapon_yolo_v21.zip`, MD5
  `8be99d786ed8481163e90957d419c3e8`; seed 42.
- **Lesson:** fine-tuning trades clean-test accuracy for robustness — the
  fix is a full retrain on combined data, not a gentler recipe.

### #5 — v2.2 fine-tune, gentler recipe (2026-11-20)
Second fine-tune on the same v2.1 data with a gentler recipe (lr0 0.0005,
warmup 5 epochs, 40 epochs, patience 12) to test whether the v2.1
regression was a recipe problem. It was not: training stopped early at
epoch 14 (never beat epoch-2 fitness) and the val crater was DEEPER than
v2.1's (v2.2: 0.776 → 0.570 at epoch 5; v2.1's low was 0.605). Test
mAP@0.5 0.7955, hard-frame F1 0.477 — worse than v2.1 on both. Abandoned.

- Weights: `weapon_yolov8s_v22_20261120_2045_best.pt` (22.53 MB)
- SHA-256: `59c079bf38ed017bc1d534f9ea6155916771175f102cb212c34d2b9cdfca821e`
- **Cleaned-test (511) measurement (23 Nov 2026, fix session):** mAP50
  **0.7557** (P 0.8133 / R 0.6919, Gun AP 0.6523 / Knife AP 0.8590).
  Recorded in `audit-v4/clean_test_eval.json`.
- Drive: not uploaded (superseded; local copy only).
- **Reproducibility:** same dataset as #4 (`k230784/sentinel-weapon-v21`,
  version not recorded); kernel slug/version not recorded (user-run on
  Kaggle); local zip `out/weapon_yolo_v21.zip`, MD5
  `8be99d786ed8481163e90957d419c3e8`; seed 42.
- **Lesson:** the clean-test regression is caused by the DATA MIX, not the
  recipe — both recipes cratered. Fine-tuning has hit its ceiling for
  this problem.

### #6 — v3 full retrain (2026-11-21) — ADOPTED
100-epoch full retrain from COCO yolov8s.pt (no prior model to forget) on
the combined dataset: full v2 train (21,809) + 855 mined + 1,177 Roboflow
+ 305 weather variants = train 24,146; Knife 2x oversampling ON; val 1,351
/ test 680 FROZEN (same split as v1/v2, directly comparable). Trained on
Kaggle (2x T4, DDP) in ~8.4 h; the run was triggered via the Kaggle API
overnight (SaveKernel without QUICK_SAVE) and monitored every 30 min.
Adopted as the current working model (v3).

- Weights: `weapon_yolov8s_v3_20261120_2215_best.pt` (22.53 MB)
- SHA-256: `2a4663ee6b9e54dfefc1b1ca8bd45f7a184f58ea682d0a1b469edb086ad95a3c`
- Drive: `SENTINEL_FYP/runs/weapon_yolov8s_v3_20261120_2215/`
  (best.pt id `1bZvKdlak3oSwIx-QApR-_B8qhMGiE4Xj`)
- **Leakage caveat:** 0.8666 is leakage-inflated — the frozen test leaks
  25% into train; the honest cleaned-test mAP50 is **0.8378**.
- **Reproducibility:** Kaggle dataset `k230784/sentinel-weapon-v3`
  (version not recorded); kernel `k230784/sentinel-weapon-v3-full-retrain`
  (version not recorded); local zip `out/weapon_yolo_v3.zip`, MD5
  `f0a66e81770ed4fc6fe403fe5b12387a`; seed 42.
- **Result:** mAP@0.5 0.8621 → **0.8666** (gate PASS — beats v2); Knife
  AP50 0.943 → **0.9493**; Gun AP50 0.7813 → **0.7839**. Precision up
  (0.8319 → 0.8484), recall slightly down (0.8278 → 0.8164). Caveat:
  mAP@0.5:0.95 slightly below v2 (0.5377 vs 0.5543). The clean-test
  margin over v2 (+1.14 pts; frozen +0.45 pts) is within run-to-run
  variance (~3 pts on the 511), so adoption rests on the combined
  evidence (hard frames + failure-mode eliminations), not the headline
  margin alone.
- **Hard frames (1,456, conf 0.40):** F1 **0.860** — beats v2.1 (0.805)
  as well as v2 (0.363). Knife F1 0.944 (recall 0.937 vs v2.1's 0.698).
  Unseen split (616 frames): tied with v2.1 (0.667 vs 0.674 — noise).
  ALL v2.1 failure-mode eliminations preserved; alarm precision 95%
  (v2.1: 88%, v2: 3%). v3 is the "both" model: v2's clean-test strength
  + v2.1's real-video robustness in one.
- **Not fixed:** UCF tiny/dark guns (TP 6/17, 11/16 positive frames still
  missed — slightly better than v2.1's 5, still bad).
- **Next:** field re-test on the same bullet/webcam videos + alarm count
  (before: 177 alarms / 15 min); demo prep for the 5 Dec review.

### #7 — v4ft fine-tune (2026-11-21) — GATE FAIL
8-epoch fine-tune from v3 best.pt (lr0 0.001) on dataset v4ft: train 4,145
(4,000 clean v3 replay (sampled after the 897 Stage-1 exclusions) + 145
user-reviewed Test8/Test9 hard frames — 13 Gun + 17 Knife positives, 115
negatives). The fine-tune hypothesis was re-tested after the v3 full
retrain: v3 was already trained on mixed data, so fine-tuning it on a
small addition should be a much smaller distribution shift than v2.1/v2.2
faced. It still failed. Clean frozen test REGRESSED (0.8666 → 0.751);
cleaned test 0.8378 → **0.709** (−12.9 pts); hard frames F1 0.860 → 0.737
— worse than v2.1's 0.805 too. Third confirmation that fine-tuning on a
small new set + replay trades clean accuracy for hard-frame robustness:
the new frames were 27% of each epoch's exposures (x10 oversample) and
115/145 of them were negatives, so the model learned "fire less overall".
BUT the new data itself works: on the Test8/Test9 frames v4ft scored
F1 **0.949** vs v3's 0.344 (1 FP on 126 negatives vs v3's 47) — the
recipe was wrong, not the data.

- Weights: `weapon_yolov8s_v4ft_20261121_1230_best.pt` (22.52 MB)
- SHA-256: `0a29654153c9d4c92c7b64101c0f210525fcd84a4eb2294e23917e9857e0adce`
- Drive: not uploaded (superseded; local copy only).
- **Reproducibility:** Kaggle dataset `k230784/sentinel-weapon-v4ft`
  (version not recorded); kernel `k230784/sentinel-weapon-v4ft-fine-tune`
  (version not recorded); local zip `out/weapon_yolo_v4ft.zip`, MD5
  `81a7fe131f73fe0444e1708dddf269b7`; seed 42.
- **Lesson:** fine-tuning is done for this problem — three recipes, three
  failures (v2.1, v2.2, v4ft). The Test8/Test9 data goes into the next
  FULL retrain at natural frequency instead.

### #8 — v4 full retrain (2026-11-22) — GATE MIXED, v3 stays
100-epoch full retrain from COCO yolov8s.pt on the audit-cleaned dataset
v4: train 23,268 = 23,123 clean v3 replay (after the Stage-1 + Stage-2
audits: 897 + 139 drops with 13 overlap, 25 flips, 26 adds, 5 reboxes)
+ 145 Test8/Test9 frames at natural frequency; Knife 2x oversampling ON;
val 1,351 / test 680 FROZEN. Trained on Kaggle (2x T4, DDP) in ~8.9 h
session wall-clock (8.36 h pure training). The gate came back MIXED, so
**v3 stays the working model** — and the live re-test (22 Nov 2026,
user-run) confirmed it: v4 did NOT win live, so **v4 is NOT adopted,
v3 stays** (see the live re-test bullet below).

- Weights: `weapon_yolov8s_v4_20261121_1350_best.pt` (22.53 MB)
- SHA-256: `8de382e8bc8e8b6959b52a9c08a6dd962042d0f3a7a11ce2b32995d1fcbba20f`
- Drive: `SENTINEL_FYP/runs/weapon_yolov8s_v4_20261121_1350/`
  (best.pt id `1dpd5lBcx-VE17JxGV5RVagNTQHIby2IB`)
- **Result:** cleaned test mAP@0.5 **0.8081** vs v3's 0.8378 (−3.0 pts,
  recall-driven: R 0.7616 vs 0.7869, P slightly up 0.8272 vs 0.8222).
  Gun AP50 0.7335 vs 0.7558; Knife AP50 0.8827 vs 0.9198. The frozen-test
  comparison (0.8293 vs 0.8666) is NOT honest — 25% of that test set
  leaks into v3's training data, and v4 never saw the leaked images.
- **Hard frames (1,456 old, conf 0.40):** F1 **0.863** vs v3's 0.860 —
  TIED. Unseen split (618 frames): 0.644 vs 0.667 — slightly worse, no
  generalization gain. New Test8/Test9 frames (156): F1 0.794 vs 0.344,
  FPs 47 → 11 — but v4 TRAINED on 145 of these, so this is memorization
  + learning, not proof. Test9 detail: Gun FPs 17 → 0; bullet FPs 9 → 3
  (reduced, not eliminated).
- **Per-slice (cleaned test, conf 0.25):** overall F1 0.756 vs 0.773.
  Gun recall drops in EVERY slice. Bright-light Knife precision is
  IDENTICAL to v3 (0.340, same 31 FPs) — v4 did not fix it. Small-Gun
  recall 0.603 vs 0.637 — worse. Thresholds re-tuned for v4 (Gun 0.36,
  Knife 0.37, val mAP50 0.8482) but the demo stays at 0.40: val has no
  hard negatives, so val-optimal is not field-optimal. Note: the
  standalone `model.val` re-run by the threshold script gives a slightly
  different val mAP50 (0.8482) than the in-training val in
  `run_summary.json` (0.8418) — a separate invocation, not an error;
  the run-summary value is the canonical training record.
- **Live re-test (22 Nov 2026, 15:32–16:12, user-run, v4 OpenVINO,
  laptop webcam):** 215 snapshots / 235 confirmed boxes — TP 43 /
  FP 190, confirmed-alarm precision **18.5%**, Gun precision **0/25**
  (every Gun alarm a phone). Phone (49 FPs) and scissors (22) were
  NOT fixed live. Verdict: v4 NOT adopted, v3 stays. Source:
  `field-test-v4/session1/tally.md`.
- **Reproducibility:** Kaggle dataset `k230784/sentinel-weapon-v4`
  (version not recorded); kernel `k230784/sentinel-weapon-v4-full-retrain`
  (version not recorded); local zip `out/weapon_yolo_v4.zip`, MD5
  `3fbe9e6220a93e324028d20755d53255`; seed 42.
- **Not fixed:** UCF tiny/dark guns (TP 5/17 vs v3's 6/17); bright-light
  Knife FPs; bullet FPs (3 remain).
- **Lesson:** the decontamination cost is now quantified — removing 1,023
  train images (audit drops) cost ~3 pts of cleaned-test mAP, while
  hard-frame performance held. The next improvement round is dataset +
  strategy work (small-gun data, bright-light negatives, imgsz sweep),
  not another same-recipe retrain.

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
v3 stays the working model — v4 is NOT adopted (mixed offline gate +
live re-test loss, 22 Nov 2026, 18.5% confirmed-alarm precision; see
#8). Next improvement round is dataset + strategy work — small-gun
data, bright-light hard negatives, and the imgsz sweep — then a v5
retrain on next week's Kaggle quota. After that: live re-test, demo prep
and rehearsal for the 5 Dec FYP-I review, then the FYP-I report (Ch 6
holds the weapon-module results).
