# SENTINEL Project — Work Breakdown Structure (WBS)

**Total Duration:** 10 Months (~300 Days)
**Methodology:** Joint Ownership (Cross-functional assignments to ensure all members get AI, Backend, and Frontend experience).
**Team Members:** 
- Muddassir Rizwan (MR)
- Faiz Hussain (FH)
- Qaim Mehdi (QM)

---

## Important Note on Scheduling
- **Task Durations:** The days mentioned are "Working Days". 
- **Exam Buffers:** I have strategically placed 3 "Buffer Periods" (10 days each) throughout the 10 months. You can shift these buffer blocks to exactly match your University Mid-Term and Final Exam dates so the project timeline doesn't break during exam weeks.

---

## Phase 1: Project Initiation & Dataset Preparation (Days 1–45)
*Goal: Set up the environment and get all training data ready and annotated.*

| Task ID | Activity Description | Assigned To | Duration |
|:---:|---|---|:---:|
| 1.1 | Finalize Proposal, Architecture Draft, & Supervisor Docs | MR, FH, QM | 5 Days |
| 1.2 | Setup Git Repo, Python Envs, & CI/CD pipeline | FH, QM | 5 Days |
| 1.3 | Collect Weapon Dataset (~4000 images) & Negative Samples | MR, QM | 7 Days |
| 1.4 | Annotate/Format Weapon Dataset for YOLO (Split workload) | MR, FH, QM | 10 Days |
| 1.5 | Source UCF-Crime + RWF-2000 clips & record simulated data | MR, FH, QM | 12 Days |
| 1.6 | Exploratory Data Analysis & Augmentation Scripting | MR, FH | 6 Days |

> **🛡️ BUFFER 1: Mid-Term Exams / Slowdown Period (10 Days)**

---

## Phase 2: Weapon Detection AI Pipeline (Days 56–95)
*Goal: Train, tune, and optimize the YOLOv8 model for gun and knife detection.*

| Task ID | Activity Description | Assigned To | Duration |
|:---:|---|---|:---:|
| 2.1 | Setup YOLOv8-small training scripts & configuration files | FH, QM | 5 Days |
| 2.2 | Initial Training Runs & Hyperparameter Tuning on local/cloud GPU | MR, FH | 15 Days |
| 2.3 | Model Evaluation (mAP, Precision, Recall) & Error Analysis | MR, QM | 10 Days |
| 2.4 | Optimize Model for Real-Time FPS (Testing TensorRT/ONNX exports) | FH, QM | 10 Days |

---

## Phase 3: Pose Extraction & LSTM Action Classifier (Days 96–150)
*Goal: Build the temporal anomaly detection system for Fighting, Robbery, and Shooting.*

| Task ID | Activity Description | Assigned To | Duration |
|:---:|---|---|:---:|
| 3.1 | Implement YOLOv8-Pose keypoint extraction pipeline | MR, QM | 7 Days |
| 3.2 | Extract & format 51-dim sequence data from all video clips | FH, QM | 12 Days |
| 3.3 | Design 2-layer LSTM Architecture in PyTorch | MR, FH | 10 Days |
| 3.4 | Train LSTM on extracted sequences & handle overfitting/tuning | MR, FH, QM | 15 Days |
| 3.5 | Evaluate LSTM (F1-score, Confusion Matrix across 4 classes) | MR, QM | 10 Days |

> **🛡️ BUFFER 2: Final Exams / Slowdown Period (10 Days)**

---

## Phase 4: Backend System & Temporal Buffer (Days 161–210)
*Goal: Build the core engine that connects AI to the outside world.*

| Task ID | Activity Description | Assigned To | Duration |
|:---:|---|---|:---:|
| 4.1 | Develop OpenCV threaded video capture module (Webcam/IP) | FH, QM | 10 Days |
| 4.2 | Implement Temporal Consistency Buffer logic (Deque programming) | MR, FH | 8 Days |
| 4.3 | Build FastAPI async server & REST endpoints | MR, QM | 12 Days |
| 4.4 | Implement WebSocket streaming endpoint for processed frames | FH, QM | 10 Days |
| 4.5 | Database Schema setup (SQLite/SQLAlchemy) & ORM mapping | MR, FH | 10 Days |

---

## Phase 5: Frontend Dashboard & Alerting (Days 211–250)
*Goal: Create the operator UI for real-time monitoring and incident review.*

| Task ID | Activity Description | Assigned To | Duration |
|:---:|---|---|:---:|
| 5.1 | Initialize React + Vite SPA & Component Architecture | MR, FH, QM | 7 Days |
| 5.2 | Build WebSocket client to consume live video feed | FH, QM | 8 Days |
| 5.3 | Implement HTML5 Canvas for bounding box/skeleton overlays | MR, QM | 10 Days |
| 5.4 | Develop Incident History Table & Search/Filter functionality | MR, FH | 10 Days |
| 5.5 | Integrate Python SMTP email alerts & Dashboard Audio Siren | FH, QM | 5 Days |

> **🛡️ BUFFER 3: Mid-Term Exams / System Review Period (10 Days)**

---

## Phase 6: Integration, Testing & Deployment (Days 261–285)
*Goal: Stitch everything together and ensure it doesn't break in real-world scenarios.*

| Task ID | Activity Description | Assigned To | Duration |
|:---:|---|---|:---:|
| 6.1 | End-to-End Pipeline Connection (Camera → AI → DB → UI) | MR, FH, QM | 7 Days |
| 6.2 | System Load Testing & FPS Benchmarking on target hardware | MR, FH | 6 Days |
| 6.3 | False-Alarm Rate Testing & Buffer Tuning in Live Scenarios | MR, QM | 7 Days |
| 6.4 | Bug Fixing, Code Refactoring, & Code Commenting | MR, FH, QM | 5 Days |

---

## Phase 7: Final Documentation & Defense (Days 286–300)
*Goal: Prepare for grading and final presentations.*

| Task ID | Activity Description | Assigned To | Duration |
|:---:|---|---|:---:|
| 7.1 | Draft Final FYP Thesis/Report Document | MR, FH, QM | 8 Days |
| 7.2 | Prepare Defense Presentation (Slides & Pre-recorded Video Demo) | MR, QM | 5 Days |
| 7.3 | Dry-run presentations & Q/A preparation | MR, FH, QM | 2 Days |

---

## Task Distribution Summary
This WBS ensures that **no single person is stuck doing just one type of work**. 
- **Muddassir (MR)** works across Dataset Prep, AI Tuning, Database ORM, Canvas Rendering, and Testing.
- **Faiz (FH)** works across CI/CD, YOLO AI, LSTM Training, Video Capture Threads, WebSockets, and Alerting.
- **Qaim (QM)** works across Dataset sourcing, Inference Optimization, FastAPI, React Front-end, and Bug Fixing.

Everyone shares the burden of tedious tasks (like annotation and thesis writing) and everyone gets to code exciting parts (like AI and WebSockets).
