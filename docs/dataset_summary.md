# 📊 SENTINEL — Curated Dataset Preprocessing & Verification Report

> **Generated Automatically**: 2026-10-17 16:09:30  
> **Status**: Verified & Synchronized with FYP-I Master Proposal

---

## 1. 🗡️ Weapon Detection Dataset Summary (YOLOv8s)

* **Schema**: 2-Class Architecture (`0: Gun`, `1: Knife`)
* **Format**: Ultralytics YOLO (`<class_id> <x_center> <y_center> <width> <height>`)
* **Total Images Processed**: **6,762**
* **Negative Background Samples**: **1,014** (15.0% ratio for false-alarm suppression)
* **Configuration File**: [`DataSets/curated/weapon_yolo/dataset.yaml`](../DataSets/curated/weapon_yolo/dataset.yaml)

### Split & Class Distribution

| Split | Gun (0) | Knife (1) | Background (Negative) | Split Total | Percentage |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Train (70%)** | 3,078 | 944 | 709 | **4,731** | 70.0% |
| **Val (20%)** | 879 | 270 | 202 | **1,351** | 20.0% |
| **Test (10%)** | 441 | 136 | 103 | **680** | 10.1% |
| **Total** | **4,398** | **1,350** | **1,014** | **6,762** | **100.0%** |

![Weapon Bounding Box Previews](./previews/weapon_samples.png)

---

## 2. 🥋 Behavior & Action Recognition Dataset Summary (YOLOv8m-Pose + 2-Layer LSTM)

* **Architecture**: 17 Keypoints × 3 = **51 dimensions** per frame
* **Window Size**: **30 consecutive frames** (Sliding Window, shape: `(30, 51)`)
* **Classes**: 4 Canonical Crime Categories
  * `0: Normal Activity` (RWF-2000 NonFight)
  * `1: Fighting` (RWF-2000 Fight + UCF-Crime Fighting)
  * `2: Robbery` (UCF-Crime Robbery)
  * `3: Shooting` (UCF-Crime Shooting)
* **Anti-Data-Leakage**: Partitioned at video level **prior** to temporal window extraction.
* **Integrity Audit**: **0** NaN/Inf array corruption incidents.
* **Total `(30, 51)` Sequences Extracted**: **3,579**

### Split & Sequence Distribution

| Split | Normal (0) | Fighting (1) | Robbery (2) | Shooting (3) | Total Sequences |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Train (70%)** | 490 | 490 | 759 | 738 | **2,477** |
| **Val (20%)** | 140 | 174 | 228 | 228 | **770** |
| **Test (10%)** | 70 | 70 | 114 | 78 | **332** |
| **Total** | **700** | **734** | **1,101** | **1,044** | **3,579** |

![Skeletal Vectors Preview](./previews/pose_samples.png)

---

## 3. 🛡️ Quality Assurance & Engineering Integrity

* [x] **Path Independence**: All preprocessing scripts use dynamic `pathlib.Path` resolution.
* [x] **Zero Data Leakage**: Video streams partitioned strictly prior to sequence windowing.
* [x] **Coordinate Normalization**: All bounding boxes clamped to `[0.0, 1.0]`; degenerate boxes discarded.
* [x] **Corrupted File Quarantine**: Quarantined items tracked in [`DataSets/raw/quarantine_log.txt`](../DataSets/raw/quarantine_log.txt).
* [x] **Git Cleanliness**: Raw datasets, videos, images, and `.npy` binaries are ignored in [`.gitignore`](../.gitignore).
