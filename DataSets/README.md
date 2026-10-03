# 📦 SENTINEL — Datasets Directory Structure

> **IMPORTANT**: Raw video files, image sets, and extracted `.npy` keypoint arrays are **ignored by Git** (configured in `.gitignore`) to prevent bloating the GitHub repository.

## 📁 Directory Structure
```
DataSets/
├── README.md               # This instruction file
├── raw/                    # Raw untouched datasets downloaded from sources
│   ├── weapons/            # Kaggle Weapon datasets (zip/extracted)
│   ├── ucf_crime/          # UCF-Crime filtered videos (Fighting, Robbery, Shooting)
│   └── rwf_2000/           # RWF-2000 video dataset (Fight, NonFight)
└── curated/                # Preprocessed and standardized datasets ready for training
    ├── weapon_yolo/        # Standardized YOLO format (images/ & labels/ for train/val/test)
    │   ├── images/
    │   └── labels/
    └── behavior_lstm/      # Extracted keypoints for 30-frame window sequences
        ├── fighting/
        ├── robbery/
        ├── shooting/
        └── normal/
```

## 📥 Dataset Sources & Download Links

### 1. Weapon Detection
* **Sources**:
  * Kaggle: Weapon Dataset for YOLOv5 / Weapon Detection for YOLO
  * Target Classes: `0: Gun`, `1: Knife`
  * Includes negative background samples to prevent false alarms on everyday items.

### 2. Behavioral Crime Recognition
* **UCF-Crime**:
  * Filtered subset: Only `Fighting`, `Robbery`, `Shooting` video clips.
* **RWF-2000**:
  * `Fight` clips mapped to `Fighting`.
  * `Non-Fight` clips mapped to `Normal Activity`.
