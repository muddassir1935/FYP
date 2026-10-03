# 🛡️ SENTINEL — Master Project Context & Decision Record

> **Notice for Teammates & AI Coding Agents (Cursor, Claude, Copilot, Antigravity, etc.)**:  
> This file is the single source of truth for the entire **SENTINEL** Final Year Project (FYP). It is 100% synchronized with the finalized, approved **FYP-I Project Proposal** ([`FYP-I-Project-Proposal.docx`](./Literature%20Review/Documentation/Proposal/FYP-I-Project-Proposal.docx)) submitted to the Department of Computer Science, FAST-NUCES Karachi Campus. Before proposing architecture changes, adding libraries, or writing code, read this document thoroughly.

---

## 1. 📋 Project Metadata

* **Project Title**: SENTINEL: Real-Time AI-Powered Surveillance System for Weapon Detection and Behavioral Crime Recognition
* **Academic Institution**: National University of Computer and Emerging Sciences (FAST-NUCES), Karachi Campus
* **Department**: Department of Computer Science
* **Academic Session**: Fall 2026 – Spring 2027 (10-Month FYP)
* **Date of Approval**: 18th September 2026
* **Project Nature**: Traditional, Research & Development (R&D)
* **Sustainable Development Goals (SDGs)**: **SDG 9: Industry, Innovation, and Infrastructure**
* **Target Hardware**: Consumer-Grade Edge GPU (e.g., NVIDIA GeForce RTX 3060 12GB / RTX 3050 Laptop)
* **Target Performance**: Continuous Real-Time Processing at **≥ 25 FPS (Target: 30 FPS)**
* **Repository**: [https://github.com/muddassir1935/FYP.git](https://github.com/muddassir1935/FYP.git)
* **Project Group Members**:
  * **Muddassir Rizwan** (`23K-0784`, CGPA 3.24, `k230784@nu.edu.pk`) — *Lead / ML & AI Model Development*
  * **Faiz Hussain** (`23K-2061`, CGPA 2.92, `k232061@nu.edu.pk`) — *Backend & Systems Pipeline*
  * **Qaim Mehdi** (`23K-2075`, CGPA 2.37, `k232075@nu.edu.pk`) — *Frontend & UI Dashboard*

---

## 2. 🎯 Core Problem, Research Questions & SENTINEL Solution

### The Core Problem
Conventional surveillance heavily relies on human operators monitoring multi-monitor video walls. Cognitive psychology research shows that human operators experience a **vigilance decrement of ~45% within the first 20 minutes**, missing over **90% of real-time security threats** during extended shifts. 

Existing automated surveillance solutions suffer from two distinct deficiencies:
1. **Static Object Detectors Only**: They detect weapons (e.g., guns, knives) in isolated frames but are architecturally blind to unarmed violence, muggings, brawls, and physical assaults.
2. **Heavy 3D-CNN Action Recognizers**: Models like C3D, I3D, or SlowFast process raw spatio-temporal video volumes. They require immense datacenter compute, suffer slow inference (<5-10 FPS on consumer hardware), lack interpretability, and are prone to false positives caused by camera motion, background clutter, and lighting variations.

### Research Questions (Formally Defined in Proposal)
* **RQ1**: How effectively can a unified pipeline combining weapon detection with pose-based action recognition classify violent crimes in real-time surveillance footage?
* **RQ2**: To what extent do skeletal key point features improve temporal crime classification accuracy compared to traditional bounding-box-based approaches?
* **RQ3**: How does the implementation of a sliding-window temporal buffer impact false-positive suppression rates in real-time threat detection?
* **RQ4**: Can a dual-model AI inference pipeline sustain real-time throughput (≥25 FPS) on consumer-grade GPU hardware?

### The SENTINEL Solution
SENTINEL implements a **Dual-Stream Lightweight AI Pipeline**:
1. **Spatial Stream (Weapon Detection)**: Runs a fine-tuned **YOLOv8s** (Small) model to detect Handguns, Rifles, and Knives with high bounding-box precision and fast inference (~8ms).
2. **Temporal Stream (Behavioral Crime Recognition)**: Uses pre-trained **YOLOv8m-Pose** (Medium) to extract 17 anatomical body keypoints (51 coordinates: X, Y, Confidence) per person. A lightweight **2-Layer LSTM** network processes sliding sequences of 30 frames to classify complex human dynamics into crime categories without processing raw video pixels through heavy 3D convolutions.
3. **Decision & False-Alarm Suppression Logic**: Rather than running an expensive secondary verification CNN (which degrades FPS), a pure algorithmic **Temporal Consistency Buffer** tracks detection persistence over sequential frames via Intersection-over-Union (IoU) sliding windows.
4. **Operational Delivery**: A decoupled **FastAPI backend** streams low-latency frames over **WebSockets** to a **React + Vite** frontend where bounding boxes and skeleton vectors are rendered directly onto an **HTML5 Canvas**, eliminating browser DOM bottlenecking. Multi-channel alerts (audio siren, email with image snapshot via SMTP, and SQLite audit logging) notify operators instantly.

---

## 3. 🛡️ Formal Requirements Specification (From Approved Proposal)

### Functional Requirements (FR)
| ID | Requirement Description | Implementation Mapping |
|---|---|---|
| **FR-01** | System shall ingest real-time video feeds. | OpenCV threaded RTSP / Webcam capture |
| **FR-02** | System shall detect and localize guns and knives with bounding boxes and confidence scores. | Ultralytics YOLOv8s fine-tuned on curated dataset |
| **FR-03** | System shall extract human skeletal key points (17 joints) for each detected person. | Pre-trained YOLOv8m-Pose (51-dim feature vector) |
| **FR-04** | System shall classify behavioral anomalies into 3 crime categories: Fighting, Robbery, Shooting (+ Normal). | 2-Layer PyTorch LSTM on 30-frame temporal window |
| **FR-05** | System shall apply temporal consistency filtering before confirming threat alerts. | Sliding-window IoU tracking buffer (zero compute) |
| **FR-06** | System shall display live video feed with real-time bounding box overlays on a web dashboard. | React + Vite with HTML5 Canvas GPU rendering |
| **FR-07** | System shall trigger an audible alarm and visual alert banner upon confirmed threat detection. | Web Audio API siren + dynamic React alert banner |
| **FR-08** | System shall log every confirmed incident to the database with timestamp, snapshot image, threat class, and confidence. | SQLite + SQLAlchemy ORM incident repository |
| **FR-09** | System shall dispatch automated email alerts with incident image attachment upon threat confirmation. | Python `smtplib` / Gmail SMTP integration |
| **FR-10** | System shall provide an incident history view with search and filter functionality. | React search/filter table querying FastAPI REST endpoints |

### Non-Functional Requirements (NFR)
* **NFR-01**: System shall process video at **≥ 25 FPS (Target: 30 FPS)** for smooth real-time operation.
* **NFR-02**: System shall operate on consumer-grade GPU hardware (e.g., RTX 3060).
* **NFR-03**: Dashboard UI shall be accessible via modern web browsers with sub-second alert latency.

---

## 4. ⛔ Strict Scope Boundaries (MoSCoW Framework)

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
│  • Edge microcontroller hardware (Jetson Nano / Raspberry Pi)          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 📊 Target Classes & Dataset Architecture

### 5.1 Target Classes
* **Weapons (Spatial — 2 Classes)**:
  * `Class 0`: **Gun** (Handguns, Pistols, Revolvers, Rifles, Shotguns)
  * `Class 1`: **Knife** (Daggers, Kitchen Knives, Machetes, Blades)
  * *Negative Samples*: Normal objects (phones, wallets, keys) with empty annotation files.
* **Behaviors (Temporal — 4 Classes)**:
  * `Class 0`: **Normal Activity** (Walking, standing, benign everyday interactions)
  * `Class 1`: **Fighting** (Assault, brawling, punching, kicking, wrestling)
  * `Class 2`: **Robbery** (Sudden mugging, snatching, threatening postures followed by rapid retreat)
  * `Class 3`: **Shooting** (Person holding weapon in aiming stance + sudden recoil dynamic)

### 5.2 Datasets & Preparation Strategy
* **Weapon Detection**:
  * Merged Kaggle repositories: *Weapon Dataset for YOLOv5* + *Weapon Detection for YOLO*.
  * Standardized into Ultralytics YOLO format (`images/` and `labels/`).
  * ~15% background-only images for negative sampling.
* **Behavioral Action**:
  * **UCF-Crime**: Filtered strictly to `Fighting`, `Robbery`, `Shooting`.
  * **RWF-2000**: `Fight` mapped to `Fighting`, `Non-Fight` mapped to `Normal Activity`.
  * **Offline Feature Extraction**: Videos are processed **once offline** through YOLOv8m-Pose. Extracted 17 keypoints (51 values) over 30 consecutive frames are saved as `.npy` arrays with shape `(30, 51)`.

---

## 6. 🌿 Git Branching & Version Control Protocol

### 6.1 "Never Work on Main"
* The `main` branch is protected. Direct commits are forbidden.
* Every feature, script, or fix must have its own branch:
  * `feature/<name>` (e.g., `feature/weapon-yolo-unification`, `feature/fastapi-websocket-stream`)
  * `fix/<name>` (e.g., `fix/buffer-iou-calculation`)
  * `docs/<name>` (e.g., `docs/proposal-sync`)

### 6.2 Conventional Commits (Zero AI Fluff)
Commit messages must follow the senior engineer standard:
```
<type>(<scope>): <imperative summary in lowercase, max 72 chars>
```
* Allowed types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`
* Allowed scopes: `(data)`, `(models)`, `(pipeline)`, `(backend)`, `(ui)`, `(buffer)`, `(docs)`
* Example: `feat(pipeline): implement deque sliding window for temporal consistency buffer`

---

## 7. 🛡️ Defensive Engineering Directives

1. **Path Independence**: Use `pathlib.Path(__file__).resolve().parent`. Never use hardcoded machine paths (`C:\...`).
2. **Async Event Loop Protection**: In FastAPI, heavy inference calls must run via `await loop.run_in_executor(executor, ...)` so WebSocket frame streaming stays ≥25 FPS.
3. **Dynamic Hardware Detection**: Auto-detect `cuda` vs `cpu`.
4. **Stream Resilience**: Implement auto-reconnection for dropped camera/RTSP feeds.
5. **No TailwindCSS**: Vanilla CSS only for the React dashboard.
6. **Canvas GPU Rendering**: Real-time bounding boxes and skeletons must be drawn on `<canvas>`, not via thousands of DOM elements.
