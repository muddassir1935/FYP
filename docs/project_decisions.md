# 🎯 SENTINEL — Project Decisions & Architectural Choices

> **Document Type**: Technical Decision Log & Trade-Off Matrix  
> **Project**: SENTINEL (AI-Powered Intelligent Surveillance System)  
> **Target Audience**: FYP Team (3 Members), Project Supervisor  
> **Status**: Finalized (Aligned with Submitted FYP-I Proposal)

---

## 📌 Document Ka Maqsad (Purpose)

Ye document tamam critical decisions, technical trade-offs, scope boundaries, aur finalized architecture ko ek jagah consolidate karta hai. Proposal approve aur submit ho chuka hai, is liye ab in decisions par implementation start hogi.

---

## 1. 🚨 Critical Strategic Decisions (Finalized)

---

### Decision 1.1: Anomaly Detection Architecture
*Surveillance video me abnormal/violent behavior detect karne ke liye kaunsa method use karein?*

**Final Choice: YOLOv8-Pose + Temporal Classifier (LSTM)**
* **Mechanism**: Pehle insaan ke 17 skeleton keypoints extract honge, phir keypoints ka temporal sequence classify hoga.
* **Compute / GPU Need**: Low to Medium. Laptops pe bhi inference fast hoga (30+ FPS).
* **Pros**: Background clutter, lighting, clothing se immune; lightweight; explainable.
* **Rationale**: FYP timeline aur real-time demo ke liye pose-based classification sab se viable aur reliable hai. Heavy 3D-CNNs (Option B) ko unki extreme compute requirements ki wajah se drop kar diya gaya hai.

---

### Decision 1.2: Target Anomaly / Crime Classes
*Finalized classes jo system detect aur classify karega:*

| # | Anomaly / Crime Class | Description |
|---|---|---|
| **1** | **Fighting (Assault)** | 2 ya zyada afraad ke darmiyan haatha-paayi, mukke/laatein maarna. |
| **2** | **Robbery / Snatching** | Sudden interaction followed by fast running / mugging. |
| **3** | **Shooting (Weapon + Pose)** | Insaan ke haath me weapon + shooting stance. |
| **4** | **Normal Activity** | Benign daily activities for baseline. |

---

### Decision 1.3: Face Recognition & Person Tracking

> [!IMPORTANT]
> **Consensus Update**:
> Face Recognition aur Cross-Camera Re-ID ko scope se **bahar (Out of Scope)** rakha gaya hai. Core system sirf Threat Detection (Weapons + Behavior) par concentrate karega kyunke CCTV me faces blur aur small hotey hain, aur real-time FPS maintain karna primary goal hai.

---

### Decision 1.4: False Alarm Mitigation vs System Latency (FPS)

**Final Choice: Pure Temporal Consistency Buffer**
* **Mechanism**: Single frame detection par alert trigger nahi hoga. Threat ko multi-frame sliding window me persist karna hoga.
* **Latency Impact**: Zero Extra Compute Overhead (Maintains ≥25 FPS).
* **Rationale**: Secondary CNN Dual-Verification ko drop kiya gaya hai kyunke wo real-time performance (FPS) gira deta hai. Temporal buffer fast aur efficient false-alarm suppression provide karta hai.

---

## 2. 💻 Tech Stack Decisions & Trade-Off Matrix

### 2.1 AI / Deep Learning Models

| Component | Final Choice | Wajah (Rationale) |
|---|---|---|
| **Weapon Detection** | **Ultralytics YOLOv8s (Fine-Tuned)** | Handguns, Rifles, Knives support karega. Mature, stable, aur fast inference. |
| **Pose & Activity** | **YOLOv8m-Pose + 2-Layer LSTM** | Keypoint-based detection fast hai, multi-person support built-in hai. |
| **DL Framework** | **PyTorch 2.x** | YOLOv8 native PyTorch me hai, academic standard hai. |

---

### 2.2 Backend & Video Ingestion

| Component | Final Choice | Wajah (Rationale) |
|---|---|---|
| **Backend API** | **FastAPI (Python)** | Async support, ultra-fast, zero serialization bottleneck. |
| **Video Capture** | **OpenCV** | Simple, stable, Webcam aur live CCTV feeds handle karta hai. |
| **Live Streaming to UI** | **WebSocket / Canvas** | Surveillance ke liye simple aur ultra-low latency. |

---

### 2.3 Frontend & Storage

| Component | Final Choice | Wajah (Rationale) |
|---|---|---|
| **Frontend** | **React + Vite** | Instant build, reusable UI components, lightweight SPA dashboard. |
| **Styling** | **Vanilla CSS** | Custom dynamic designs aur animations ke liye (Tailwind strictly avoid kiya gaya hai). |
| **Database** | **SQLite** | Zero configuration; incident logs store karne ke liye kaafi hai. |

---

## 3. 🛡️ Scope Boundaries (Strict MoSCoW)

Scope creep se bachne ke liye boundaries bilkul clear hain:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SENTINEL SCOPE BOUNDARIES                       │
├────────────────────────────────────────────────────────────────────────┤
│  ✅ IN SCOPE (Commitment — Must Have)                                  │
│  • Single-Stream Video Ingestion (CCTV feed / RTSP / Video file)       │
│  • Weapon Detection (Handguns, Rifles, Knives)                         │
│  • Behavioral Crime Detection (Fighting, Robbery, Shooting)            │
│  • Temporal Consistency Buffer (False alarm filter)                    │
│  • Real-Time Web Dashboard (Live Video + Canvas Bounding Boxes)        │
│  • Incident Logging Database                                           │
│  • Visual/Audio Alerts on Dashboard                                    │
├────────────────────────────────────────────────────────────────────────┤
│  ❌ OUT OF SCOPE (Strictly Won't Have)                                 │
│  • Multi-camera network & cross-camera tracking                        │
│  • Face Recognition / Suspect Watchlist                                │
│  • Concealed weapon detection under clothes                            │
│  • Audio-based gunshot detection                                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 👥 Team Work Distribution

| Member | Core Responsibilities |
|---|---|
| **Muddassir Rizwan** | Model Development: Weapon & Action Models (YOLOv8 fine-tuning, Pose extraction, LSTM temporal anomaly classification) |
| **Faiz Hussain** | Backend & Systems: OpenCV video capture, Temporal Consistency Buffer, FastAPI server, WebSockets |
| **Qaim Mehdi** | Frontend & UI: React + Vite surveillance dashboard, Canvas overlay, Alert Banner, Incident logs UI |
