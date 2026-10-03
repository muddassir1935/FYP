# 🛡️ SENTINEL — Master Project Context & Decision Record

> **Notice for Teammates & AI Coding Agents (Cursor, Claude, Copilot, Antigravity, etc.)**:  
> This file is the single source of truth for the entire **SENTINEL** Final Year Project (FYP). Before proposing architecture changes, adding libraries, or writing code, read this document thoroughly to align with all finalized design decisions, scope boundaries, and dataset specifications.

---

## 1. 📋 Project Metadata

* **Project Title**: SENTINEL: Real-Time AI-Powered Surveillance System for Weapon Detection and Behavioral Crime Recognition
* **Domain**: Computer Vision, Deep Learning, Real-Time Intelligent Edge Surveillance
* **Project Type**: 10-Month Final Year Project (FYP) — Fall 2026 to Spring 2027
* **Target Hardware**: Edge / Consumer GPU (e.g., NVIDIA GeForce RTX 3060 12GB / RTX 3050 Laptop)
* **Target Performance**: Continuous Real-Time Processing at **≥ 25 FPS**
* **Repository**: [https://github.com/muddassir1935/FYP.git](https://github.com/muddassir1935/FYP.git)
* **Team Members & Core Roles**:
  * **Muddassir Rizwan (Lead / ML & AI Models)**: Dataset unification, YOLOv8s weapon detection, YOLOv8m-Pose skeleton extraction, 2-layer LSTM temporal action classifier, model optimization.
  * **Faiz Hussain (Backend & Systems Pipeline)**: OpenCV threaded video ingestion, Temporal Consistency Buffer implementation, FastAPI asynchronous backend, WebSocket frame/metadata streaming, model inference pipeline.
  * **Qaim Mehdi (Frontend & UI Dashboard)**: React + Vite surveillance dashboard, HTML5 Canvas GPU-accelerated overlay rendering, audio siren & visual alert banner, SQLite incident history & analytics.

---

## 2. 🎯 Core Problem Statement & The SENTINEL Solution

### The Core Problem
Conventional surveillance heavily relies on human operators monitoring multi-monitor video walls. Cognitive psychology research shows that human operators miss up to **90% of critical security events after just 20 minutes** due to vigilance decrement and fatigue. 

Existing automated surveillance solutions suffer from two distinct failure modes:
1. **Static Object Detectors Only**: They detect weapons (e.g., guns, knives) but are blind to unarmed violence, muggings, brawls, and physical assaults.
2. **Heavy 3D-CNN Action Recognizers**: Models like C3D, I3D, or SlowFast process raw spatio-temporal video volumes. They require immense datacenter compute, suffer slow inference (<5-10 FPS on consumer hardware), lack interpretability, and are prone to false positives caused by camera motion, background clutter, and lighting variations.

### The SENTINEL Solution
SENTINEL implements a **Dual-Stream Lightweight AI Pipeline**:
1. **Spatial Stream (Weapon Detection)**: Runs a fine-tuned **YOLOv8s** model to detect Handguns, Rifles, and Knives with high bounding-box precision and fast inference.
2. **Temporal Stream (Behavioral Crime Recognition)**: Uses **YOLOv8m-Pose** to extract 17 anatomical body keypoints (51 coordinates: X, Y, Confidence) per person. A lightweight **2-Layer LSTM (Long Short-Term Memory)** network processes sliding sequences of 30 frames to classify complex human dynamics into crime categories without processing raw video pixels through heavy 3D convolutions.
3. **Decision & False-Alarm Suppression Logic**: Rather than running an expensive secondary verification CNN (which degrades FPS), a pure algorithmic **Temporal Consistency Buffer** tracks detection persistence over sequential frames via Intersection-over-Union (IoU) sliding windows.
4. **Edge-Ready Dashboard**: A decoupled **FastAPI backend** streams low-latency frames over **WebSockets** to a **React + Vite** frontend where bounding boxes and skeleton vectors are rendered directly onto an **HTML5 Canvas**, eliminating browser DOM bottlenecking.

---

## 3. 🛡️ Strict Scope Boundaries (MoSCoW Framework)

To prevent scope creep and guarantee high-quality real-time execution, strict boundaries have been approved:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SENTINEL SCOPE BOUNDARIES                       │
├────────────────────────────────────────────────────────────────────────┤
│  ✅ IN SCOPE (Commitment — Must Have)                                  │
│  • Single-Stream Video Ingestion (CCTV RTSP feed, Webcam, video files) │
│  • Weapon Detection: Handguns, Rifles, Knives (Standardized 2 classes) │
│  • Behavioral Crime Recognition: Fighting, Robbery, Shooting, Normal   │
│  • Pure Algorithmic Temporal Consistency Buffer (False alarm filter)   │
│  • Real-Time Web Dashboard (Live Video + GPU Canvas Overlays)          │
│  • SQLite Incident Logging (Snapshots, timestamps, crime tags)         │
│  • Multi-Channel Alerting (Audible Siren + Visual Flash + Email Alert) │
│  • Real-Time Performance Target: ≥ 25 FPS on consumer GPUs             │
├────────────────────────────────────────────────────────────────────────┤
│  ❌ OUT OF SCOPE (Strictly Won't Have — Do NOT Implement)              │
│  • Multi-camera network switching & cross-camera Re-ID                 │
│  • Facial recognition or suspect identity watchlist matching           │
│  • Concealed weapon detection under clothing (requires mmWave / IR)    │
│  • Acoustic / Audio-based gunshot sound detection                      │
│  • Heavy 3D-CNN spatio-temporal backbones (C3D, I3D, SlowFast)         │
└────────────────────────────────────────────────────────────────────────┘
```

> ⚠️ **Directive for AI Agents**: Never recommend or implement Face Recognition or 3D-CNNs. These were formally rejected during proposal defense due to low CCTV resolution, privacy restrictions, and severe latency penalties.

---

## 4. 🧠 Detailed Architectural Decisions & Trade-Off Matrix

### 4.1 Model Architecture Decisions

| Component | Final Choice | Alternatives Considered | Decisive Rationale |
|---|---|---|---|
| **Weapon Detection** | **Ultralytics YOLOv8s (Fine-Tuned)** | YOLOv5, YOLOv8x, Faster R-CNN, SSD | YOLOv8s provides the optimal sweet spot between mAP and inference speed (~8ms per frame). YOLOv8x is too heavy; Faster R-CNN is too slow for real-time edge streaming. |
| **Pose Estimation** | **YOLOv8m-Pose** | MediaPipe, OpenPose | MediaPipe struggles with multi-person surveillance crowds; OpenPose is computationally prohibitive; YOLOv8m-Pose natively detects multiple people and extracts 17 COCO keypoints accurately under partial occlusion. |
| **Action Recognition** | **2-Layer LSTM on Pose Sequences** | C3D, I3D, SlowFast, Video Vision Transformers (ViViT) | LSTMs operate on compact 51-dimensional vectors rather than million-pixel volumes. They are invariant to background changes, illumination shifts, and camera color grading while consuming negligible GPU VRAM. |
| **False-Alarm Filter** | **Pure Temporal Consistency Buffer** | Secondary Verification CNN, Bayesian Filtering | A secondary CNN halves FPS (<15 FPS). A pure deque-based sliding window tracking threats across 10 frames with an IoU threshold requires 0.0ms extra GPU compute while eliminating single-frame flicker alarms. |

### 4.2 System & Infrastructure Decisions

| Component | Final Choice | Alternatives Considered | Decisive Rationale |
|---|---|---|---|
| **Deep Learning Framework** | **PyTorch 2.x** | TensorFlow / Keras | Native ecosystem for Ultralytics YOLOv8 and dynamic LSTM sequence handling. |
| **Backend Framework** | **FastAPI (Python)** | Flask, Django, Node.js | Asynchronous ASGI runtime (`asyncio`), high-performance request handling, native WebSocket support, seamless integration with Python AI pipelines. |
| **Video Ingestion** | **OpenCV (`cv2.VideoCapture`)** | PyAV, GStreamer | Threaded OpenCV buffer reader provides simple, robust RTSP/webcam ingestion across all operating systems. |
| **Live Streaming Protocol** | **WebSockets (`ws://`)** | WebRTC, HLS, RTMP | HLS has high latency (2-5s); WebRTC is complex to negotiate; WebSockets provide ultra-low latency (<50ms) direct binary/JSON push to the browser. |
| **Frontend Framework** | **React + Vite** | Next.js, Plain HTML/JS | Instant HMR development, componentized UI state management, lightweight SPA bundle. |
| **Frontend Styling** | **Vanilla CSS (Modern Dark Mode)** | TailwindCSS, Bootstrap | Full creative control over animations, glassmorphism, responsive grids, and zero dependency overhead. |
| **Video Overlay Rendering** | **HTML5 Canvas (`<canvas>`)** | DOM `<div>` / SVG elements | Rendering 20+ dynamic bounding boxes and skeleton bones at 30 FPS inside browser DOM causes severe layout thrashing; HTML5 Canvas renders directly via GPU rasterization with zero lag. |
| **Database** | **SQLite + SQLAlchemy** | PostgreSQL, MongoDB | Zero configuration, self-contained single-file storage, completely portable across development machines, ideal for FYP incident logging. |

---

## 5. 📊 Target Classes & Dataset Architecture

### 5.1 Target Classes

#### Spatial Stream (Weapon Detection — 2 Unified Classes)
* `Class 0`: **Gun** (Encompasses Handguns, Pistols, Revolvers, Rifles, Shotguns)
* `Class 1`: **Knife** (Encompasses Daggers, Kitchen Knives, Machetes, Blades)
* *Negative Samples*: Frames containing everyday objects (smartphones, wallets, keys, water bottles) labeled as empty `.txt` annotations to train the model to avoid false positives.

#### Temporal Stream (Behavioral Crime Recognition — 4 Unified Classes)
* `Class 0`: **Normal Activity** (Walking, standing, talking, shopping, benign interactions)
* `Class 1`: **Fighting** (Assault, brawling, punching, kicking, wrestling)
* `Class 2`: **Robbery** (Sudden mugging, snatching, threatening postures followed by rapid retreat)
* `Class 3`: **Shooting** (Person holding weapon in aiming stance + sudden recoil / violent aiming dynamic)

---

### 5.2 Datasets & Preparation Strategy

#### A. Weapon Detection Datasets
* **Sources**: Merged Kaggle repositories:
  1. *Weapon Dataset for YOLOv5*
  2. *Weapon Detection for YOLO*
* **Standardization**: Converted into standard Ultralytics YOLO format (`images/` and `labels/` with normalized bounding boxes: `class_id center_x center_y width height`).
* **Negative Sample Integration**: ~15% background-only images with empty annotation files.

#### B. Behavioral Action Datasets
* **Sources**:
  1. **UCF-Crime Dataset**: Filtered down strictly to 3 relevant subsets: `Fighting`, `Robbery`, `Shooting`. (All non-relevant classes such as Arson, Burglary, Explosion, Shoplifting, Vandalism are excluded).
  2. **RWF-2000 Dataset**: 2,000 surveillance video clips:
     * `Fight` clips mapped into `Fighting` class.
     * `Non-Fight` clips mapped into `Normal Activity` class.
* **Feature Extraction Pipeline**:
  * Videos are trimmed to salient crime segments.
  * Frames are sampled at consistent frame rates.
  * `YOLOv8m-Pose` is run **once offline** to extract 17 keypoints (51 values) per frame.
  * Formatted into sliding windows of 30 frames `(30, 51)` and stored as pre-extracted `.npy` arrays.
  * *Why offline extraction?* Training the LSTM directly on pre-extracted coordinate arrays takes seconds per epoch instead of hours of repeated video decoding and pose inference.

---

## 6. 📁 Repository Directory Structure

```
FYP/
├── .gitignore                      # Production gitignore (excludes datasets, weights, venv)
├── README.md                       # Public-facing repository documentation
├── PROJECT_CONTEXT.md              # THIS FILE — Master technical context & decision record
├── AGENTS.md                       # Operational directives for AI coding agents
│
├── docs/                           # Detailed technical documentation
│   ├── architecture_deep_dive.md   # Component-by-component data flow analysis
│   ├── project_decisions.md        # Complete decision log & trade-off rationale
│   ├── roadmap_wbs.md              # 10-Month WBS, milestone schedule & dates
│   ├── capability_matrix.md        # Comparison with literature review research papers
│   └── team_responsibilities.md    # Task assignment matrix for all 3 members
│
├── DataSets/                       # Local data storage (Directory tracked, files gitignored)
│   ├── README.md                   # Instructions on downloading & placing datasets
│   ├── raw/                        # Untouched downloaded datasets (Kaggle, UCF, RWF)
│   └── curated/                    # Standardized YOLO images & extracted LSTM keypoints
│
├── models/                         # Model weights and definitions
│   ├── README.md                   # Instructions on model weights & download links
│   └── weights/                    # Saved .pt / .onnx checkpoints (gitignored)
│
├── scripts/                        # Utility & operational automation scripts
│   ├── data_preparation/          # Scripts for merging, cleaning, and extracting keypoints
│   ├── training/                   # Model training and fine-tuning scripts
│   └── inference/                  # Standalone benchmark and verification scripts
│
├── src/                            # Production source code
│   ├── backend/                    # FastAPI application, database models, WebSockets
│   ├── frontend/                   # React + Vite surveillance dashboard
│   └── pipeline/                   # Video capture, YOLOv8, Pose, LSTM, Buffer modules
│
└── Literature Review/              # Academic deliverables & foundational papers
    ├── Documentation/              # Approved FYP-I proposal, thesis chapters, presentation
    └── Research_Papers/            # 5 Foundational IEEE/Springer reference papers
```

---

## 7. ⏱️ 10-Month Milestone Roadmap (Aug 2026 – Jun 2027)

* **Phase 1: Project Initiation & Research (Aug 17 – Sep 13)** ✅ *COMPLETED*
  * Proposal drafted, reviewed, approved, and officially submitted.
* **Phase 2: Dataset Collection & Preparation (Sep 28 – Oct 25)** 🔄 *CURRENT PHASE*
  * Downloading, merging Kaggle weapon sources, filtering UCF-Crime and RWF-2000, offline keypoint extraction.
* **Phase 3: Weapon Detection Model Training (Nov 9 – Dec 6)**
  * Fine-tuning YOLOv8s on unified weapon dataset, mAP evaluation on test split.
* **Phase 4: Action Recognition Model Training (Jan 15 – Feb 18)**
  * Training 2-layer LSTM on pre-extracted `(30, 51)` keypoint sequences, F1-score evaluation.
* **Phase 5: Backend & Systems Pipeline (Feb 19 – Mar 25)**
  * FastAPI asynchronous server, OpenCV RTSP streaming, Temporal Consistency Buffer, WebSockets.
* **Phase 6: Frontend Dashboard (Mar 26 – Apr 22)**
  * React + Vite dark mode UI, HTML5 Canvas GPU rendering, audio sirens, SQLite log view.
* **Phase 7: System Integration & Live Testing (Apr 23 – May 20)**
  * Full end-to-end integration, FPS benchmarking, webcam live crime simulation demo.
* **Phase 8: Documentation & Final FYP-II Defense (May 21 – Jun 15)**
  * Final thesis chapters, video demo recording, defense presentation.

---

## 8. 🤖 Guidelines for AI Agents Working in this Repository

When working on this repository, any AI assistant (Cursor, Claude, Copilot, Antigravity) must strictly obey these directives:

1. **Adhere to the Dual-Stream Architecture**: Do not replace the YOLOv8-Pose + LSTM pipeline with 3D-CNNs (C3D, I3D, SlowFast, VideoMAE).
2. **Respect Scope Boundaries**: Never introduce Face Recognition, facial embeddings, multi-camera tracking, or audio-based gunshot models.
3. **Protect Git Cleanliness**: Never commit `.pt`, `.pth`, `.mp4`, `.avi`, `.zip`, `.npy`, or image datasets to Git. Always verify `.gitignore`.
4. **Preserve Low Latency (≥25 FPS)**: Ensure all frame-processing pipelines in Python are non-blocking and asynchronous. Never introduce heavy synchronous blocking operations into the FastAPI frame loop.
5. **UI Technology Rules**: Use React + Vite. Use Vanilla CSS with tailored design tokens. Do NOT install TailwindCSS unless specifically asked.
6. **Canvas Rendering**: Ensure all real-time video bounding boxes and pose skeletons are drawn using the HTML5 Canvas 2D rendering context over the video element, rather than creating hundreds of DOM elements.
7. **Type Hints & Clean Code**: All backend Python code must use PEP 484 type hints and structured docstrings.
