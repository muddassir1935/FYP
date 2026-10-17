# 📓 SENTINEL — Lead Computer Vision Engineering Logbook

This logbook records technical decisions, mathematical formulations, experimental pipelines, and quantitative dataset evolutions for the **SENTINEL** Real-Time Intelligent Surveillance System.

---

## [Entry 2026-10-17] Dataset Scaling, Class Balancing & Bi-Lateral Skeletal Symmetry Augmentation

* **Author**: Lead Computer Vision Engineer (SENTINEL FYP Team)
* **Branch**: `feature/dataset-expansion-and-balancing`
* **Status**: Tested, Verified & Merged into Master Pipeline

---

### 1. Context & Architectural Rationale

An initial audit of our preprocessed datasets revealed two severe bottlenecks:
1. **Weapon Imbalance (Spatial YOLO Stream)**:
   The baseline distribution had **4,398 Guns** vs only **161 Knives** (3.5% knife representation). An object detector trained on this distribution exhibits a high risk of catastrophic knife false-negatives in edge surveillance deployments.
   * **Target**: Scale Knife instances to **1,200 – 1,500** (~20-25% of positive detections) while preserving ~15% negative background sampling to suppress false alarms on everyday carry items (phones, wallets, keys).
2. **Action Sequence Scarcity (Temporal LSTM Stream)**:
   The baseline feature extraction yielded only **128 total sequences** across 4 behavior classes (`normal`, `fighting`, `robbery`, `shooting`), with as few as 16 training sequences per class. A 2-layer LSTM cannot converge or generalize on such limited sequential data.
   * **Target**: Scale total temporal sequences to **2,500+ sequences** (~600 to 750 balanced sequences per class).

---

### 2. Phase 2: Supplementary Knife Ingestion & Spatial Transforms

#### Ingestion & Schema Harmonization
* Supplementary knife imagery from `DataSets/raw/weapons/extra_knives/` (1,000 annotated image/label pairs) was pooled with Source A knife samples (161 samples), yielding **1,161 raw knife instances**.
* All annotations were strictly mapped to canonical **`Class 1: Knife`**.
* Coordinate bounding boxes were strictly clamped to $[0.0, 1.0]$; degenerate boxes with $w \le 0$ or $h \le 0$ were quarantined.

#### Geometric & Photometric Augmentation Pipeline
To span the deficit from 1,161 to the target of 1,350 knife instances, an offline augmentation engine was implemented in [`scripts/data_preparation/prepare_weapon_dataset.py`](../scripts/data_preparation/prepare_weapon_dataset.py):
1. **Horizontal Flipping**:
   $$\tilde{x}_c = 1.0 - x_c, \quad \tilde{y}_c = y_c, \quad \tilde{w} = w, \quad \tilde{h} = h$$
2. **Affine Random Rotation ($\pm 10^\circ$ to $\pm 15^\circ$)**:
   Given center rotation matrix $M \in \mathbb{R}^{2 \times 3}$:
   $$\begin{bmatrix} x' \\ y' \end{bmatrix} = M \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
   The 4 corners of each bounding box are transformed through $M$, and the tight enclosing bounding box is re-projected and normalized.
3. **Photometric Jitter & Surveillance Blur**:
   * HSV saturation and brightness randomly scaled by factor $\alpha \in [0.7, 1.3]$:
     $$S' = \text{clip}(S \cdot \alpha_S, 0, 255), \quad V' = \text{clip}(V \cdot \alpha_V, 0, 255)$$
   * Dynamic Gaussian blur ($\sigma \in [0.6, 1.4]$) and directional motion blur kernels simulating CCTV camera motion.
4. **False-Alarm Mitigation**:
   Maintained a strict **15.0% negative background ratio** ($N_{\text{neg}} = 1,014$ samples with empty annotation files) across Train, Val, and Test splits.

---

### 3. Phase 3: Action Recognition Full-Batch Extraction (Stride = 5)

In [`scripts/data_preparation/extract_pose_keypoints.py`](../scripts/data_preparation/extract_pose_keypoints.py):
* Updated temporal sliding window stride from **15 frames to 5 frames** with window size $W = 30$ consecutive frames:
  $$N_{\text{windows}} = \left\lfloor \frac{T - W}{\text{stride}} \right\rfloor + 1$$
* For a standard 90-frame video clip, `stride=5` extracts 13 overlapping observation windows instead of only 4 windows (an immediate **$3.25\times$ data multiplier**).
* Preserved strict **video-level anti-data-leakage partitioning**: clips are partitioned into 70% Train, 20% Val, 10% Test *prior* to temporal window slicing.
* Feature dimensions per frame: $17 \text{ joints} \times 3 = 51$ values $(x_{\text{norm}}, y_{\text{norm}}, \text{confidence})$.

---

### 4. Phase 4: Bi-Lateral Skeletal Joint Symmetry Mirroring & Jitter

To expand temporal sequences beyond 2,500 without incurring massive raw video storage footprints, an anatomically valid skeletal symmetry algorithm was implemented in [`scripts/data_preparation/augment_pose_sequences.py`](../scripts/data_preparation/augment_pose_sequences.py).

#### COCO 17-Keypoint Bi-Lateral Symmetry
Each frame $t \in [0, 29]$ in array $A \in \mathbb{R}^{30 \times 51}$ consists of 17 keypoints $j \in \{0, \dots, 16\}$:
$$P_j = \left( x_j, y_j, c_j \right)$$

1. **Horizontal Inversion**:
   $$\tilde{x}_j = 1.0 - x_j \quad (\text{if } c_j > 0)$$
2. **Anatomical Swap Pairs**:
   $$\begin{aligned}
   \text{Left Eye (1)} &\longleftrightarrow \text{Right Eye (2)} \\
   \text{Left Ear (3)} &\longleftrightarrow \text{Right Ear (4)} \\
   \text{Left Shoulder (5)} &\longleftrightarrow \text{Right Shoulder (6)} \\
   \text{Left Elbow (7)} &\longleftrightarrow \text{Right Elbow (8)} \\
   \text{Left Wrist (9)} &\longleftrightarrow \text{Right Wrist (10)} \\
   \text{Left Hip (11)} &\longleftrightarrow \text{Right Hip (12)} \\
   \text{Left Knee (13)} &\longleftrightarrow \text{Right Knee (14)} \\
   \text{Left Ankle (15)} &\longleftrightarrow \text{Right Ankle (16)} \\
   \text{Nose (0)} &\longleftrightarrow \text{Nose (0)}
   \end{aligned}$$
* **Anatomical Validity**: A right-handed strike, mugging motion, or weapon-drawing gesture is mirrored into an anatomically authentic left-handed execution, effectively doubling temporal samples while maintaining biomechanical validity.

#### Gaussian Coordinate Jitter
Micro-noise was added to all detected keypoints:
$$\tilde{x}_j = \text{clip}(x_j + \epsilon_x, 0.0, 1.0), \quad \tilde{y}_j = \text{clip}(y_j + \epsilon_y, 0.0, 1.0), \quad \epsilon \sim \mathcal{N}(0, 0.005^2)$$
This provides robustness against CCTV sensor noise, perspective compression, and person scale variations.

---

### 5. Final Dataset Metrics Summary

#### Spatial Weapon Detection Dataset (YOLOv8s)
| Split | Gun (0) | Knife (1) | Background (Negative) | Total Images | Positive % Knife |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Train (70%)** | 3,078 | 944 | 709 | **4,731** | 23.5% |
| **Val (20%)** | 879 | 270 | 202 | **1,351** | 23.5% |
| **Test (10%)** | 441 | 136 | 103 | **680** | 23.6% |
| **Total** | **4,398** | **1,350** | **1,014** | **6,762** | **23.5%** |

#### Temporal Behavior Recognition Dataset (YOLOv8m-Pose + 2-Layer LSTM)
| Split | Normal (0) | Fighting (1) | Robbery (2) | Shooting (3) | Split Total | Percentage |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train (70%)** | 490 | 490 | 759 | 738 | **2,477** | 69.2% |
| **Val (20%)** | 140 | 174 | 228 | 228 | **770** | 21.5% |
| **Test (10%)** | 70 | 70 | 114 | 78 | **332** | 9.3% |
| **Total** | **700** | **734** | **1,101** | **1,044** | **3,579** | **100.0%** |

* **Zero NaN/Inf Anomalies**: All 3,579 `.npy` files verified with $100\%$ finite float values.
* **Format & Serialization**: Standardized `(30, 51)` float32 arrays saved as `seq_*.npy` (base), `seq_*_flipped.npy` (mirrored), and `seq_*_jitter.npy` (perturbed).

