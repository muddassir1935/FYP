# SENTINEL — Individual Tasks (Phase & Milestone Based)

**Duration:** 10 Months (Aug 17, 2026 – Jun 15, 2027)
- **FYP-I (Fall 2026):** Aug 17 – Dec 6 (Classes) | Dec 14 – Jan 8 (Finals)
- **FYP-II (Spring 2027):** Jan 15 – Jun 15

**Members:** Muddassir Rizwan (MR), Faiz Hussain (FH), Qaim Mehdi (QM)

---

## Phase 1: Project Initiation & Research
**Milestone: Proposal Approved & Literature Review Complete**
**Duration: 28 days (Aug 17 – Sep 13)**

| Team Member | Activity | Tentative Date |
|---|---|---|
| Muddassir Rizwan (MR) | Literature review on weapon detection models (YOLO family) | Aug 17 – Sep 6 |
| Muddassir Rizwan (MR) | Draft project proposal & system architecture design | Sep 7 – Sep 13 |
| Faiz Hussain (FH) | Literature review on pose estimation & action recognition (LSTM) | Aug 17 – Sep 6 |
| Faiz Hussain (FH) | Setup GitHub repository, project structure & development environment | Sep 7 – Sep 13 |
| Qaim Mehdi (QM) | Literature review on temporal filtering & false-alarm suppression | Aug 17 – Sep 6 |
| Qaim Mehdi (QM) | Survey & shortlist benchmark datasets (UCF-Crime, RWF-2000, Kaggle) | Sep 7 – Sep 13 |

> 🛡️ *Midterm #1 Buffer: Sep 14 – Sep 27 (Weeks 5-6)*

---

## Phase 2: Dataset Collection & Preparation
**Milestone: Unified Weapon Dataset & Action Recognition Dataset Ready**
**Duration: 28 days (Sep 28 – Oct 25)**

| Team Member | Activity | Tentative Date |
|---|---|---|
| Muddassir Rizwan (MR) | Download & merge Kaggle weapon datasets into unified 2-class dataset (Gun, Knife) | Sep 28 – Oct 11 |
| Muddassir Rizwan (MR) | Verify & clean weapon annotations (YOLO format standardization) | Oct 12 – Oct 25 |
| Faiz Hussain (FH) | Download UCF-Crime dataset & filter relevant categories (Fighting, Robbery, Shooting) | Sep 28 – Oct 11 |
| Faiz Hussain (FH) | Download RWF-2000 dataset & map classes to unified 4-class schema | Oct 12 – Oct 25 |
| Qaim Mehdi (QM) | Prepare negative-sample images for weapon dataset (false-positive reduction) | Sep 28 – Oct 11 |
| Qaim Mehdi (QM) | Apply data augmentations & finalize train/val/test splits for all datasets | Oct 12 – Oct 25 |

> 🛡️ *Midterm #2 Buffer: Oct 26 – Nov 8 (Weeks 11-12)*

---

## Phase 3: Weapon Detection Model Training (YOLOv8s)
**Milestone: YOLOv8s Fine-tuned & Evaluated on Weapon Dataset**
**Duration: 28 days (Nov 9 – Dec 6)**

| Team Member | Activity | Tentative Date |
|---|---|---|
| Muddassir Rizwan (MR) | Fine-tune YOLOv8s on unified weapon dataset (Google Colab) | Nov 9 – Nov 22 |
| Muddassir Rizwan (MR) | Evaluate weapon model (mAP@0.5, Precision, Recall, Confusion Matrix) | Nov 23 – Dec 6 |
| Faiz Hussain (FH) | Implement YOLOv8m-Pose keypoint extraction pipeline | Nov 9 – Nov 22 |
| Faiz Hussain (FH) | Test pose estimation on sample CCTV footage & validate keypoint quality | Nov 23 – Dec 6 |
| Qaim Mehdi (QM) | Extract 51-dim keypoint sequences from UCF-Crime videos using YOLOv8m-Pose | Nov 9 – Nov 22 |
| Qaim Mehdi (QM) | Extract keypoint sequences from RWF-2000 & merge into unified training data | Nov 23 – Dec 6 |

> 🛡️ *Makeup Week + Final Exams: Dec 7, 2026 – Jan 14, 2027 (~5 weeks pause)*

---

## Phase 4: Action Recognition Model Training (LSTM)
**Milestone: LSTM Action Classifier Trained & Evaluated on 4 Classes**
**Duration: 35 days (Jan 15 – Feb 18)**

| Team Member | Activity | Tentative Date |
|---|---|---|
| Muddassir Rizwan (MR) | Design 2-layer LSTM architecture in PyTorch | Jan 15 – Jan 28 |
| Muddassir Rizwan (MR) | Hyperparameter tuning & experimentation (hidden units, dropout, window size) | Jan 29 – Feb 18 |
| Faiz Hussain (FH) | Train LSTM on extracted keypoint sequences (Google Colab GPU) | Jan 15 – Feb 4 |
| Faiz Hussain (FH) | Evaluate LSTM (F1-Score, Confusion Matrix across Fighting/Robbery/Shooting/Normal) | Feb 5 – Feb 18 |
| Qaim Mehdi (QM) | Implement sliding window data loader for 30-frame sequences | Jan 15 – Jan 28 |
| Qaim Mehdi (QM) | Cross-validation testing & overfitting analysis | Jan 29 – Feb 18 |

---

## Phase 5: Backend & API Development (FastAPI)
**Milestone: Working Backend with AI Pipeline, REST API & WebSocket Streaming**
**Duration: 35 days (Feb 19 – Mar 25)**

| Team Member | Activity | Tentative Date |
|---|---|---|
| Muddassir Rizwan (MR) | Implement Temporal Consistency Buffer logic (deque-based sliding window) | Feb 19 – Mar 4 |
| Muddassir Rizwan (MR) | Design & implement database schema (SQLite/SQLAlchemy) for incident logging | Mar 5 – Mar 25 |
| Faiz Hussain (FH) | Develop FastAPI async server with OpenCV threaded video capture | Feb 19 – Mar 4 |
| Faiz Hussain (FH) | Integrate trained AI models (YOLOv8s + YOLOv8m-Pose + LSTM) into inference pipeline | Mar 5 – Mar 18 |
| Faiz Hussain (FH) | Implement WebSocket server for real-time frame & alert broadcasting | Mar 19 – Mar 25 |
| Qaim Mehdi (QM) | Build REST API endpoints (incident retrieval, system status, configuration) | Feb 19 – Mar 4 |
| Qaim Mehdi (QM) | Implement automated SMTP email alert dispatch with incident snapshots | Mar 5 – Mar 25 |

---

## Phase 6: Frontend Dashboard Development (React + Vite)
**Milestone: Fully Functional Surveillance Dashboard with Real-Time Rendering**
**Duration: 28 days (Mar 26 – Apr 22)**

| Team Member | Activity | Tentative Date |
|---|---|---|
| Muddassir Rizwan (MR) | Build React + Vite SPA with component architecture & routing | Mar 26 – Apr 8 |
| Muddassir Rizwan (MR) | Implement audible alarm notification & visual alert banner | Apr 9 – Apr 22 |
| Faiz Hussain (FH) | Implement HTML5 Canvas overlay for bounding box & skeleton rendering | Mar 26 – Apr 8 |
| Faiz Hussain (FH) | Connect frontend to WebSocket for live frame streaming | Apr 9 – Apr 22 |
| Qaim Mehdi (QM) | Develop searchable incident history table with filters | Mar 26 – Apr 8 |
| Qaim Mehdi (QM) | UI polish (dark-mode theme, responsive layout, dashboard statistics) | Apr 9 – Apr 22 |

---

## Phase 7: Integration & Testing
**Milestone: Complete End-to-End Pipeline Working with Live Webcam Demo**
**Duration: 28 days (Apr 23 – May 20)**

| Team Member | Activity | Tentative Date |
|---|---|---|
| Muddassir Rizwan (MR) | End-to-end pipeline integration (Camera → YOLO → Pose → LSTM → Buffer → Dashboard) | Apr 23 – May 6 |
| Muddassir Rizwan (MR) | FPS benchmarking & performance profiling on target hardware | May 7 – May 20 |
| Faiz Hussain (FH) | System testing with pre-recorded crime videos (UCF-Crime test set) | Apr 23 – May 6 |
| Faiz Hussain (FH) | Live webcam testing with simulated scenarios | May 7 – May 20 |
| Qaim Mehdi (QM) | False-alarm rate testing & buffer threshold tuning | Apr 23 – May 6 |
| Qaim Mehdi (QM) | Bug fixing, edge case handling & final code cleanup | May 7 – May 20 |

---

## Phase 8: Documentation & Final Defense
**Milestone: FYP Thesis Submitted & Defense Presentation Delivered**
**Duration: 26 days (May 21 – Jun 15)**

| Team Member | Activity | Tentative Date |
|---|---|---|
| Muddassir Rizwan (MR) | Write methodology & results chapters of FYP thesis | May 21 – Jun 4 |
| Muddassir Rizwan (MR) | Prepare final defense presentation slides | Jun 5 – Jun 15 |
| Faiz Hussain (FH) | Write system architecture & implementation chapters of FYP thesis | May 21 – Jun 4 |
| Faiz Hussain (FH) | Record system demo video for defense | Jun 5 – Jun 15 |
| Qaim Mehdi (QM) | Write introduction, literature review & conclusion chapters of FYP thesis | May 21 – Jun 4 |
| Qaim Mehdi (QM) | Compile evaluation results & prepare Q/A responses for defense | Jun 5 – Jun 15 |
